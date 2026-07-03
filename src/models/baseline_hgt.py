"""
baseline_hgt.py — M7 External Baseline #4: HGT (Heterogeneous Graph Transformer)

Reference:
    Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020).
    Heterogeneous Graph Transformer.
    Proceedings of The Web Conference 2020 (WWW 2020), pp. 2704-2710.
    Paper:  https://arxiv.org/abs/2003.01332
    Author: https://github.com/acbull/pyHGT
    PyG:    torch_geometric.nn.HGTConv

原論文為通用異質圖架構（node type + edge type 各自獨立 attention），
在 OAG / Amazon 等推薦系統資料集發表。本檔適配到 ADR-TW 跨市場預測，
測試「HGT 學跨層 relation」能否勝過「MAGNET 用結構保證 A12 對角」。

架構：
    x_seq_L1 [B, T, n, F]  → SharedLSTM → h_L1 [B, n, H_lstm]  (ADR 節點初始特徵)
    x_seq_L2 [B, T, n, F]  → SharedLSTM → h_L2 [B, n, H_lstm]  (TW  節點初始特徵)

    per-snapshot 建構異質圖：
        node types  = {'adr', 'tw'}
        edge types  = {
            ('adr', 'intra', 'adr')  : L1 intra-market correlation edges
            ('tw',  'intra', 'tw')   : L2 intra-market correlation edges
            ('adr', 'cross', 'tw')   : A12 對角（每張快照都是 identity mapping）
            ('tw',  'cross', 'adr')  : A12 反向（讓 attention 雙向流動）
        }
        → HGTConv × 2 (含 meta-relation attention)
        → 取 'tw' node 輸出 → Projection → PredictionHead → [B, n]

與 MAGNET 的對比：
    MAGNET: A12 用「兩層節點順序對齊 + fusion」隱式保證（structural inductive bias）
    HGT   : A12 用「顯式 edge type + relation-specific attention」讓模型自己學

若 HGT test_IC < MAGNET，即證明「結構保證跨市場配對 > 讓模型從資料學跨市場關係」，
這是 MAGNET 論文核心主張之一。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv

from src.models.encoders import SharedLSTM
from src.models.prediction_head import PredictionHead, CombinedLoss


NODE_TYPES = ["adr", "tw"]
EDGE_TYPES = [
    ("adr", "intra", "adr"),
    ("tw",  "intra", "tw"),
    ("adr", "cross", "tw"),
    ("tw",  "cross", "adr"),
]
METADATA = (NODE_TYPES, EDGE_TYPES)


class BaselineHGT(nn.Module):
    """HGT baseline: heterogeneous graph transformer with meta-relation attention."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m_cfg    = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        gat_cfg  = m_cfg["gat"]
        proj_cfg = m_cfg["projection"]
        head_cfg = m_cfg["prediction_head"]

        H_lstm  = lstm_cfg["hidden_dim"]
        H_hgt   = gat_cfg["hidden_dim"]
        d_prime = proj_cfg["d_prime"]
        num_layers = gat_cfg["num_layers"]
        heads   = gat_cfg["num_heads"]

        # Shared LSTM 兩市場共用（跟 MAGNET 相同的 domain 動機：跨市場時序語意一致）
        self.lstm = SharedLSTM(lstm_cfg)

        # HGT 堆疊
        self.hgt_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = H_lstm if i == 0 else H_hgt
            self.hgt_layers.append(
                HGTConv(
                    in_channels=in_dim,
                    out_channels=H_hgt,
                    metadata=METADATA,
                    heads=heads,
                )
            )
        self.hgt_dropout = nn.Dropout(p=gat_cfg.get("dropout", 0.1))

        # 投影 + 預測（只用 'tw' node 輸出做預測，因為 target 是 TW next-day return）
        self.proj = nn.Sequential(
            nn.Linear(H_hgt, d_prime),
            nn.GELU(),
            nn.LayerNorm(d_prime),
        )
        self.predictor = PredictionHead(head_cfg, d_prime=d_prime)

        # 統一 loss
        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

        assert lstm_cfg["input_dim"] == 9, (
            f"SharedLSTM input_dim 應與 TECH_FEATURE_COLS 維度一致（9），"
            f"當前為 {lstm_cfg['input_dim']}"
        )

    def _build_a12_edges(self, n: int, device: torch.device) -> Tensor:
        """A12 對角邊：ADR node i ↔ TW node i（identity mapping）。"""
        idx = torch.arange(n, device=device, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)   # [2, n]

    def _apply_hgt(
        self,
        h_L1_b: Tensor,                                       # [n, H_lstm]
        h_L2_b: Tensor,                                       # [n, H_lstm]
        ei_L1_b: Tensor,
        ei_L2_b: Tensor,
        a12_ei: Tensor,
    ) -> Tensor:
        """對單張快照跑一次完整 HGT stack，回傳 TW node 表示 [n, H_hgt]。"""
        x_dict = {
            "adr": h_L1_b,
            "tw":  h_L2_b,
        }
        edge_index_dict = {
            ("adr", "intra", "adr"): ei_L1_b,
            ("tw",  "intra", "tw"):  ei_L2_b,
            ("adr", "cross", "tw"):  a12_ei,
            ("tw",  "cross", "adr"): a12_ei.flip(0),
        }

        for i, hgt in enumerate(self.hgt_layers):
            x_dict = hgt(x_dict, edge_index_dict)
            if i < len(self.hgt_layers) - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                x_dict = {k: self.hgt_dropout(v) for k, v in x_dict.items()}
        return x_dict["tw"]   # [n, H_hgt]

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        x_L1 = batch["x_seq_L1"]                              # [B, T, n, F]
        x_L2 = batch["x_seq_L2"]
        ei_L1 = batch["edge_index_L1"]
        ei_L2 = batch["edge_index_L2"]

        B, T, n, F_ = x_L1.shape

        # Phase 1: LSTM encoding（shared）
        h_L1 = self.lstm(x_L1)                                # [B, n, H_lstm]
        h_L2 = self.lstm(x_L2)                                # [B, n, H_lstm]

        # A12 對角邊（每張快照相同：identity mapping）
        a12_ei = self._build_a12_edges(n, device=x_L1.device)  # [2, n]

        # 逐張快照跑 HGT（因為每張 snapshot 的 L1/L2 邊都不同）
        is_list_form_L1 = isinstance(ei_L1, (list, tuple))
        is_list_form_L2 = isinstance(ei_L2, (list, tuple))
        outs = []
        for b in range(B):
            ei1 = ei_L1[b] if is_list_form_L1 else ei_L1
            ei2 = ei_L2[b] if is_list_form_L2 else ei_L2
            outs.append(self._apply_hgt(h_L1[b], h_L2[b], ei1, ei2, a12_ei))
        h_tw = torch.stack(outs, dim=0)                       # [B, n, H_hgt]

        # 投影 + 預測
        h_proj = self.proj(h_tw)                              # [B, n, d']
        y_hat  = self.predictor(h_proj)                       # [B, n]

        # 對齊 MAGNET 簽名
        extras = {
            "h_L1":    h_proj,
            "h_L2":    h_proj,
            "h_fused": h_proj,
            "alpha":   torch.zeros(*h_proj.shape[:-1], 1, device=h_proj.device),
            "gate":    torch.zeros_like(h_proj),
        }
        return y_hat, extras

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
