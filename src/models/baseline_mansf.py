"""
baseline_mansf.py — M7 External Baseline #3: MAN-SF (no-text variant)

Reference:
    Sawhney, R., Agarwal, S., Wadhwa, A., & Shah, R. R. (2020).
    Deep Attentive Learning for Stock Movement Prediction from
    Social Media Text and Company Correlations.
    EMNLP 2020, pp. 8415-8426.
    ACL:    https://aclanthology.org/2020.emnlp-main.676/
    Author: https://github.com/midas-research/man-sf-emnlp

原論文融合三種模態：(a) stock price GRU、(b) hierarchical tweet attention、
(c) correlation GCN。因專案無 Twitter 對應資料，本檔採 **no-text variant**：
disable tweet 分支，保留 price + correlation 兩種模態的 multimodal attention。
論文中應標為 MAN-SF (no-text) 以避免公平性爭議。

實作偏離：原論文 correlation graph 用 GCNConv (Kipf & Welling 2017)，
本檔改用 GATv2Conv 因 PyTorch MPS backend 對 GCNConv 內部
edge_index masking 有 uninitialized-memory bug。GAT 是 GCN 的
attention 擴展，對 correlation-weighted graph 的建模能力等價或更強，
論文附錄應標為「MAN-SF (no-text) with GATv2 backbone」。

架構：
    x_seq_L2 [B, T, n, F]  ← 只用 TW 端序列（單市場 baseline）
        → per-stock GRU                        → [B, n, H_gru]     (price 模態)
        → 用 GRU 輸出當節點特徵 + L2 correlation graph
            → 2-layer GCNConv                  → [B, n, H_gcn]     (graph 模態)
        → project both to d'                   → [B, n, d'] × 2
        → Modality attention (2-way softmax)   → [B, n, d']        (fused)
        → PredictionHead                       → [B, n]

與 MAGNET 差異：
    - GRU 而非 LSTM（原論文設計）
    - GCNConv 而非 GATv2Conv（原論文設計）
    - Modality attention 融「不同資訊源」（price vs graph）而非「不同市場」
    - 完全不使用 ADR / L1 圖

只使用 TW 端資料（batch["x_seq_L2"] / edge_L2）。用途：驗證「single-market
+ heterogeneous attention」對比 MAGNET 的「cross-market + gated fusion」。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv

from src.models.prediction_head import PredictionHead, CombinedLoss


class BaselineMANSF(nn.Module):
    """
    MAN-SF (no-text variant): GRU price + GCN correlation + modality attention.

    單市場（只用 TW L2）、disable tweet branch。
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m_cfg    = cfg["model"]
        lstm_cfg = m_cfg["lstm"]     # sizes 復用（H_gru = H_lstm）
        gat_cfg  = m_cfg["gat"]      # H_gcn = H_gat
        proj_cfg = m_cfg["projection"]
        head_cfg = m_cfg["prediction_head"]

        input_dim = lstm_cfg["input_dim"]
        H_gru     = lstm_cfg["hidden_dim"]
        H_gcn     = gat_cfg["hidden_dim"]
        d_prime   = proj_cfg["d_prime"]
        gcn_layers = gat_cfg["num_layers"]
        dropout    = gat_cfg.get("dropout", 0.1)

        # ── 模態 1：Price GRU（per-stock） ──────────────────────────
        # 原論文用 GRU，忠實移植
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=H_gru,
            num_layers=lstm_cfg["num_layers"],
            batch_first=True,
            bidirectional=False,
            dropout=lstm_cfg.get("dropout", 0.0) if lstm_cfg["num_layers"] > 1 else 0.0,
        )

        # ── 模態 2：Correlation Graph（原用 GCNConv，MPS bug workaround 改 GATv2） ──
        # 每層 GATv2 用 1 head + concat=False 讓輸出維度乾淨 = H_gcn
        # edge_attr [E, 1] 進 attention 作為邊權
        self.gcn_layers = nn.ModuleList()
        for i in range(gcn_layers):
            in_dim = H_gru if i == 0 else H_gcn
            self.gcn_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=H_gcn,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    edge_dim=1,
                    add_self_loops=True,
                )
            )
        self.gcn_dropout = nn.Dropout(p=dropout)

        # ── 兩模態投影至共同維度 d' ─────────────────────────────────
        self.proj_price = nn.Linear(H_gru, d_prime)
        self.proj_graph = nn.Linear(H_gcn, d_prime)

        # ── Modality attention (2-way) ─────────────────────────────
        # 原論文 Eq. 8-10：對每個 stock、每個模態算 attention score
        # score = v^T tanh(W [price ; graph])
        attn_hidden = m_cfg.get("fusion", {}).get("attention_hidden", 64)
        self.mod_attn = nn.Sequential(
            nn.Linear(2 * d_prime, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 2),   # 2 個模態
        )

        # ── Predictor ──────────────────────────────────────────────
        self.predictor = PredictionHead(head_cfg, d_prime=d_prime)

        # ── 統一 loss ──────────────────────────────────────────────
        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

        assert input_dim == 9, (
            f"GRU input_dim 應與 TECH_FEATURE_COLS 維度一致（9），"
            f"當前為 {input_dim}"
        )

    def _encode_price(self, x_seq: Tensor) -> Tensor:
        """
        Args:
            x_seq: [B, T, n, F]
        Returns:
            h_price: [B, n, H_gru]   (每檔 stock 最後一步 GRU hidden)
        """
        B, T, n, F_ = x_seq.shape
        x = x_seq.permute(0, 2, 1, 3).reshape(B * n, T, F_)  # [B*n, T, F]
        _, h_n = self.gru(x)                                  # h_n: [num_layers, B*n, H_gru]
        h_last = h_n[-1]                                      # [B*n, H_gru]
        return h_last.reshape(B, n, -1)                       # [B, n, H_gru]

    def _apply_gcn(
        self,
        h:          Tensor,                                   # [B, n, H_gru]
        edge_index: Tensor | list[Tensor],
        edge_attr:  Tensor | list[Tensor],
    ) -> Tensor:
        """
        對 batch 內每張快照跑一次 correlation graph attention。

        Returns:
            out: [B, n, H_gcn]
        """
        B = h.size(0)
        is_list_form = isinstance(edge_index, (list, tuple))
        outs = []
        for b in range(B):
            ei = edge_index[b] if is_list_form else edge_index
            ea = edge_attr[b]  if is_list_form else edge_attr
            x = h[b]                                          # [n, H_gru]
            for i, conv in enumerate(self.gcn_layers):
                x = conv(x, ei, edge_attr=ea)
                if i < len(self.gcn_layers) - 1:
                    x = F.relu(x)
                    x = self.gcn_dropout(x)
            outs.append(x)                                    # [n, H_gcn]
        return torch.stack(outs, dim=0)                       # [B, n, H_gcn]

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        x_L2 = batch["x_seq_L2"]                              # [B, T, n, F]
        ei   = batch["edge_index_L2"]
        ea   = batch["edge_attr_L2"]

        # 模態 1：Price GRU
        h_price = self._encode_price(x_L2)                    # [B, n, H_gru]

        # 模態 2：Correlation GCN（用 GRU output 當節點特徵）
        h_graph = self._apply_gcn(h_price, ei, ea)            # [B, n, H_gcn]

        # 投影至 d'
        p_proj = self.proj_price(h_price)                     # [B, n, d']
        g_proj = self.proj_graph(h_graph)                     # [B, n, d']

        # Modality attention (2-way softmax over modalities)
        concat = torch.cat([p_proj, g_proj], dim=-1)          # [B, n, 2*d']
        scores = self.mod_attn(concat)                        # [B, n, 2]
        alpha  = F.softmax(scores, dim=-1)                    # [B, n, 2]
        fused  = alpha[..., 0:1] * p_proj + alpha[..., 1:2] * g_proj  # [B, n, d']

        y_hat = self.predictor(fused)                         # [B, n]

        # 對齊 MAGNET 簽名（extras 不用於 loss）
        extras = {
            "h_L1":    fused,
            "h_L2":    fused,
            "h_fused": fused,
            "alpha":   alpha.mean(dim=-1, keepdim=True),      # per-node modality attention
            "gate":    torch.zeros_like(fused),
            # 供分析
            "modality_alpha": alpha,   # [B, n, 2] price vs graph weight
            "h_price":        p_proj,
            "h_graph":        g_proj,
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
