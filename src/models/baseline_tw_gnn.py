"""
baseline_tw_gnn.py — Stage 0 Ablation: TW-only single-layer GNN（脊椎 baseline）

對應 M6 Plan Stage 0 #2 (opt_p18_baseline_tw_gnn)
目的：論文 ablation table 的脊椎——驗證「multiplex 設計（L1 + L2 + A12 跨層）
是否真的優於單純的 TW 端 GNN」。
若 MAGNET test_IC ≈ baseline_tw_gnn test_IC，代表 ADR 層幾乎沒貢獻，
MAGNET 退化成「複雜版 TW GNN」，學術主張無法成立。

架構：
    x_seq_L2 [B, T, n, F]
        → SharedLSTM        → [B, n, H_lstm]
        → GATEncoder (L2)   → [B, n, H_gat]    (使用 edge_index_L2 + edge_attr_L2)
        → TypeProjection    → [B, n, d']
        → PredictionHead    → [B, n]

注意：
  - 僅使用 TW 端 (L2) 序列 + L2 圖；完全不看 ADR 序列與 L1 圖
  - 參數量比 MAGNET 少約一半（去掉 GAT_L1、Proj_L1、CrossLayerFusion）
  - forward 介面與 MAGNET 對齊（接同一個 batch dict，回傳 (y_hat, extras)）
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.encoders import SharedLSTM, GATEncoder, TypeProjection
from src.models.prediction_head import PredictionHead, CombinedLoss


class BaselineTWGNN(nn.Module):
    """TW-only 單層 GNN baseline（無 ADR、無跨層融合）。"""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m_cfg    = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        gat_cfg  = m_cfg["gat"]
        proj_cfg = m_cfg["projection"]
        head_cfg = m_cfg["prediction_head"]

        d_prime = proj_cfg["d_prime"]
        H_gat   = gat_cfg["hidden_dim"]

        self.lstm = SharedLSTM(lstm_cfg)
        self.gat  = GATEncoder(gat_cfg)
        self.proj = TypeProjection(proj_cfg, in_dim=H_gat)
        self.head = PredictionHead(head_cfg, d_prime=d_prime)

        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

        assert lstm_cfg["input_dim"] == 9, (
            f"SharedLSTM input_dim 應與 TECH_FEATURE_COLS 維度一致（9），"
            f"當前為 {lstm_cfg['input_dim']}"
        )

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        x_L2 = batch["x_seq_L2"]            # [B, T, n, F]
        ei   = batch["edge_index_L2"]       # list[Tensor] 或 Tensor
        ea   = batch["edge_attr_L2"]

        h_lstm = self.lstm(x_L2)            # [B, n, H_lstm]
        h_gat  = self._apply_gat_batched(self.gat, h_lstm, ei, ea)  # [B, n, H_gat]
        h      = self.proj(h_gat)           # [B, n, d']
        y_hat  = self.head(h)               # [B, n]

        extras = {
            "h_L1":    h,    # 對齊 MAGNET 簽名（不用於 loss）
            "h_L2":    h,
            "h_fused": h,
            "alpha":   torch.zeros(*h.shape[:-1], 1, device=h.device),
            "gate":    torch.zeros_like(h),
        }
        return y_hat, extras

    @staticmethod
    def _apply_gat_batched(
        gat: GATEncoder,
        h:   Tensor,                                # [B, n, H_lstm]
        edge_index: Tensor | list[Tensor],
        edge_attr:  Tensor | list[Tensor],
    ) -> Tensor:
        """與 MAGNET._apply_gat_batched 同樣的逐張處理；保留 list-or-tensor 雙模式。"""
        B = h.size(0)
        is_list_form = isinstance(edge_index, (list, tuple))

        outs = []
        for b in range(B):
            ei = edge_index[b] if is_list_form else edge_index
            ea = edge_attr[b]  if is_list_form else edge_attr
            outs.append(gat(h[b], ei, ea))
        return torch.stack(outs, dim=0)

    def compute_loss(
        self,
        y_hat:  Tensor,
        y:      Tensor,
        extras: dict,
    ) -> tuple[Tensor, dict]:
        return self.criterion(
            y_hat=y_hat,
            y=y,
            h_L1=extras.get("h_L1"),
            h_L2=extras.get("h_L2"),
        )
