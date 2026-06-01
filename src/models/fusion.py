"""
fusion.py — Phase 2 跨市場融合模組
Corresponds to IMPLEMENTATION_SPEC §4

CrossLayerFusion：用 per-node attention + gated fusion
將 ADR(t) 的領先訊號注入到 TW(t+1) 預測。

注意：
  - Attention 是 per-node（每家公司獨立），對應 A12 的對角設計。
  - Gate g 是 per-node、per-dim（shape = [n, d']），非純量。
  - 回傳 (h_fused, α, g) 以供注意力權重分析。
  - Adaptive weight by volatility regime（SPEC §4.3）在 MVP 版本跳過，
    確認 baseline 收斂後再加。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CrossLayerFusion(nn.Module):
    """
    跨層融合模組（Gated Cross-Market Fusion）。

    Corresponds to IMPLEMENTATION_SPEC §4.1 / §4.2

    Args:
        cfg (dict): base.yaml 中的 model.fusion 區塊，包含：
            attention_hidden (int): attn MLP 中間維度（預設 64）
            gate_activation  (str): gate 激活函數，固定 "sigmoid"
        d_prime (int): 共同潛在維度（來自 model.projection.d_prime）

    Shapes:
        forward input  h_L1, h_L2 : [n, d'] 或 [B, n, d']
        forward output h_fused     : [n, d'] 或 [B, n, d']
                       alpha        : [n, 1] 或 [B, n, 1]  (per-node attention score)
                       gate         : [n, d'] 或 [B, n, d'] (per-node per-dim gate)
    """

    def __init__(self, cfg: dict, d_prime: int) -> None:
        super().__init__()
        attn_hidden = cfg.get("attention_hidden", 64)

        # Step 1: per-node attention score
        # input: [*, 2*d'], output: [*, 1]
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * d_prime, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 1),
        )

        # Step 2: per-node per-dim gate
        # input: [*, 2*d'], output: [*, d']
        self.gate_mlp = nn.Linear(2 * d_prime, d_prime)

    def forward(self, h_L1: Tensor, h_L2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            h_L1 : [..., n, d']  ADR 投影後表示
            h_L2 : [..., n, d']  TW 投影後表示

        Returns:
            h_fused : [..., n, d']  融合後表示
            alpha   : [..., n, 1]  per-node attention score（供分析）
            gate    : [..., n, d'] per-node per-dim gate（供分析）
        """
        # [..., n, 2*d']
        concat = torch.cat([h_L1, h_L2], dim=-1)

        # Step 1: attention（per-node）
        alpha = torch.sigmoid(self.attn_mlp(concat))   # [..., n, 1]

        # Step 2: gated fusion（per-node, per-dim）
        gate = torch.sigmoid(self.gate_mlp(concat))    # [..., n, d']
        h_fused = gate * h_L1 + (1.0 - gate) * h_L2   # [..., n, d']

        return h_fused, alpha, gate
