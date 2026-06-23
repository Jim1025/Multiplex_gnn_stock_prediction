"""
multiplex_gnn.py — MAGNET 主類別
MAGNET: Multiplex ADR-Guided Network for Equity Trading

Corresponds to IMPLEMENTATION_SPEC §0 / §8.1

三階段架構：
  Phase 1: Dual-Market Graph Encoding
           SharedLSTM → GAT_L1/GAT_L2 → Proj_L1/Proj_L2
  Phase 2: Gated Cross-Market Fusion
           CrossLayerFusion (per-node attention + gate)
  Phase 3: Multi-Objective Return Prediction
           PredictionHead (MLP)

使用方式：
    import yaml
    from src.models.multiplex_gnn import MAGNET

    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)

    model = MAGNET(cfg)
    y_hat, extras = model(batch)
"""

from __future__ import annotations

import yaml
import torch
import torch.nn as nn
from torch import Tensor

from src.models.encoders import SharedLSTM, GATEncoder, TypeProjection
from src.models.fusion import CrossLayerFusion
from src.models.prediction_head import PredictionHead, CombinedLoss


class MAGNET(nn.Module):
    """
    MAGNET — Multiplex ADR-Guided Network for Equity Trading

    Corresponds to IMPLEMENTATION_SPEC §0 (one-page overview) & §8.1

    Args:
        cfg (dict): 完整的 base.yaml 解析結果（含 model / loss_weights / align_loss）

    Shapes（以 batch 維度為例）：
        x_seq_L1   : [B, T, n, F]  ADR T 步歷史特徵序列
        x_seq_L2   : [B, T, n, F]  TW  T 步歷史特徵序列
        edge_index_L1 : [2, E1]
        edge_attr_L1  : [E1, 1]
        edge_index_L2 : [2, E2]
        edge_attr_L2  : [E2, 1]
        y             : [B, n]     TW(t+1) log_return（訓練時提供）

    Note:
        - A12 對角矩陣已在 graph_builder 中確保正確性（T12 測試守護），
          此處不需顯式傳入——CrossLayerFusion 直接對齊相同 index 的節點即可。
        - 推論時 batch 可以是單張快照（squeeze B 維度），forward 同樣有效。
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        m_cfg   = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        gat_cfg  = m_cfg["gat"]
        proj_cfg = m_cfg["projection"]
        fuse_cfg = m_cfg["fusion"]
        head_cfg = m_cfg["prediction_head"]

        # M6 Stage 0 ablation: 切斷 ADR → TW 跨層訊號（保留 fusion 結構）
        # 啟用時將 h_L1 在進 fusion 前歸零，h_fused = (1 - gate) * h_L2
        self.disable_a12 = bool(m_cfg.get("disable_a12", False))

        d_prime   = proj_cfg["d_prime"]
        H_lstm    = lstm_cfg["hidden_dim"]
        H_gat     = gat_cfg["hidden_dim"]   # 最後一層 concat=False → H_gat

        # ── Phase 1 ───────────────────────────────────────────────────
        # SharedLSTM：L1 / L2 共用同一份權重
        self.lstm = SharedLSTM(lstm_cfg)

        # GAT：L1 / L2 各自獨立（multiplex ≠ siamese）
        self.gat_L1 = GATEncoder(gat_cfg)
        self.gat_L2 = GATEncoder(gat_cfg)

        # Projection：L1 / L2 各自獨立
        self.proj_L1 = TypeProjection(proj_cfg, in_dim=H_gat)
        self.proj_L2 = TypeProjection(proj_cfg, in_dim=H_gat)

        # ── Phase 2 ───────────────────────────────────────────────────
        self.fusion = CrossLayerFusion(fuse_cfg, d_prime=d_prime)

        # ── Phase 3 ───────────────────────────────────────────────────
        self.head = PredictionHead(head_cfg, d_prime=d_prime)

        # 損失函數（訓練時使用）
        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

        # LSTM 輸入維度斷言
        assert lstm_cfg["input_dim"] == 9, (
            f"SharedLSTM input_dim 應與 TECH_FEATURE_COLS 維度一致（9），"
            f"當前為 {lstm_cfg['input_dim']}"
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        """
        Args:
            batch (dict):
                "x_seq_L1"      : [B, T, n, F]   ADR 歷史序列
                "x_seq_L2"      : [B, T, n, F]   TW  歷史序列
                "edge_index_L1" : list[Tensor[2, E_b]] 長度 B（每張快照邊數不同）
                                  或單一 Tensor [2, E1]（會自動廣播給所有 batch）
                "edge_attr_L1"  : list[Tensor[E_b, 1]] 同上對應
                "edge_index_L2" : list[Tensor[2, E_b]]
                "edge_attr_L2"  : list[Tensor[E_b, 1]]
                "y"             : [B, n]  (選填，推論時可不傳)

        Returns:
            y_hat   : [B, n]  預測 log_return
            extras  : dict{
                          "h_L1"   : [B, n, d']  ADR 投影後表示
                          "h_L2"   : [B, n, d']  TW  投影後表示
                          "h_fused": [B, n, d']  融合後表示
                          "alpha"  : [B, n, 1]   per-node attention
                          "gate"   : [B, n, d']  per-node per-dim gate
                      }
        """
        x_L1 = batch["x_seq_L1"]          # [B, T, n, F]
        x_L2 = batch["x_seq_L2"]          # [B, T, n, F]
        ei_L1 = batch["edge_index_L1"]    # list[Tensor] 或 Tensor
        ea_L1 = batch["edge_attr_L1"]
        ei_L2 = batch["edge_index_L2"]
        ea_L2 = batch["edge_attr_L2"]

        B = x_L1.size(0)
        n = x_L1.size(2)

        # ── Phase 1: LSTM 時序編碼 ────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §3.1
        h_lstm_L1 = self.lstm(x_L1)   # [B, n, H_lstm]
        h_lstm_L2 = self.lstm(x_L2)   # [B, n, H_lstm]（共用同一份 LSTM）

        # ── Phase 1: GAT 圖編碼 ───────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §3.2
        # 每張快照獨立跑 GAT（edge_index 已是單張快照的局部索引）
        h_gat_L1 = self._apply_gat_batched(self.gat_L1, h_lstm_L1, ei_L1, ea_L1)  # [B, n, H_gat]
        h_gat_L2 = self._apply_gat_batched(self.gat_L2, h_lstm_L2, ei_L2, ea_L2)  # [B, n, H_gat]

        # ── Phase 1: 投影至 d' 維 ─────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §3.3
        h_L1 = self.proj_L1(h_gat_L1)  # [B, n, d']
        h_L2 = self.proj_L2(h_gat_L2)  # [B, n, d']

        # ── Phase 2: 跨市場融合 ───────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §4
        # M6 Stage 0 ablation: disable_a12=True 時將 ADR 訊號零化（保留 fusion 結構）
        h_L1_in = torch.zeros_like(h_L1) if self.disable_a12 else h_L1
        h_fused, alpha, gate = self.fusion(h_L1_in, h_L2)  # [B, n, d']

        # ── Phase 3: 預測 ─────────────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §5.1
        y_hat = self.head(h_fused)  # [B, n]

        extras = {
            "h_L1":    h_L1,
            "h_L2":    h_L2,
            "h_fused": h_fused,
            "alpha":   alpha,
            "gate":    gate,
        }
        return y_hat, extras

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_gat_batched(
        gat: GATEncoder,
        h:   Tensor,                                # [B, n, H_lstm]
        edge_index: Tensor | list[Tensor],          # 每張快照的邊（list）或共用張量
        edge_attr:  Tensor | list[Tensor],
    ) -> Tensor:
        """
        對 batch 中每張快照逐一跑 GAT。每張快照的 edge_index/edge_attr
        可能不同（list 形式，來自 multiplex_collate）；
        也支援所有快照共用同一組邊（單張量形式，例如手動構造的測試用 batch）。

        後續若需更高 throughput，可改用 torch_geometric.data.Batch 一次性處理。

        Returns:
            out : [B, n, H_gat]
        """
        B = h.size(0)
        is_list_form = isinstance(edge_index, (list, tuple))

        outs = []
        for b in range(B):
            ei = edge_index[b] if is_list_form else edge_index
            ea = edge_attr[b]  if is_list_form else edge_attr
            outs.append(gat(h[b], ei, ea))  # [n, H_gat]
        return torch.stack(outs, dim=0)     # [B, n, H_gat]

    # ------------------------------------------------------------------
    # 便利方法：計算損失
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        y_hat:  Tensor,
        y:      Tensor,
        extras: dict,
    ) -> tuple[Tensor, dict]:
        """
        計算 CombinedLoss。

        Args:
            y_hat  : [B, n]
            y      : [B, n]
            extras : forward 回傳的 extras dict

        Returns:
            loss       : scalar
            components : dict{"mse", "rank", "align"}
        """
        return self.criterion(
            y_hat=y_hat,
            y=y,
            h_L1=extras.get("h_L1"),
            h_L2=extras.get("h_L2"),
        )

    # ------------------------------------------------------------------
    # 工廠方法：從 YAML 路徑建立模型
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str = "configs/base.yaml") -> "MAGNET":
        """從 YAML 路徑建立 MAGNET 實例。"""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)
