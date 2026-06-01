"""
prediction_head.py — Phase 3 預測頭與多目標損失
Corresponds to IMPLEMENTATION_SPEC §5

PredictionHead  : MLP，輸出每家公司的 log_return 預測 ŷ ∈ ℝ^n
CombinedLoss    : ℒ = ℒ_MSE + λ_rank·ℒ_rank + λ_align·ℒ_align
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# PredictionHead
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """
    預測頭（MLP）。

    Corresponds to IMPLEMENTATION_SPEC §5.1

    Architecture:
        Linear(d', H) → ReLU → Dropout → Linear(H, 1) → squeeze(-1)

    Args:
        cfg    (dict): base.yaml 中的 model.prediction_head 區塊：
            hidden_dim (int)  : MLP 隱藏維度（預設 64）
            dropout    (float): dropout 機率（預設 0.2）
        d_prime (int): 輸入維度（= model.projection.d_prime）

    Shapes:
        forward input  h_fused : [n, d'] 或 [B, n, d']
        forward output          : [n] 或 [B, n]
    """

    def __init__(self, cfg: dict, d_prime: int) -> None:
        super().__init__()
        hidden_dim = cfg.get("hidden_dim", 64)
        dropout = cfg.get("dropout", 0.2)
        self.mlp = nn.Sequential(
            nn.Linear(d_prime, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_fused: Tensor) -> Tensor:
        """
        Args:
            h_fused : [..., n, d']

        Returns:
            y_hat : [..., n]
        """
        return self.mlp(h_fused).squeeze(-1)


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    多目標損失函數。

    Corresponds to IMPLEMENTATION_SPEC §5.2

    ℒ = ℒ_MSE + λ_rank · ℒ_rank + λ_align · ℒ_align

    ℒ_MSE  : 均方誤差（報酬數值準確性）
    ℒ_rank : RankNet pairwise ranking loss（報酬排序，對 IC 有幫助）
    ℒ_align: InfoNCE contrastive loss（同公司 ADR/TW 表示對齊）

    Args:
        loss_cfg  (dict): base.yaml 的 loss_weights 區塊：
            mse   (float): λ_MSE（預設 1.0）
            rank  (float): λ_rank（預設 0.1）
            align (float): λ_align（預設 0.1）
        align_cfg (dict): base.yaml 的 align_loss 區塊：
            enabled     (bool) : 是否啟用 align loss（預設 True）
            temperature (float): InfoNCE 溫度（預設 0.1）
    """

    def __init__(self, loss_cfg: dict, align_cfg: dict) -> None:
        super().__init__()
        self.lambda_mse   = loss_cfg.get("mse",   1.0)
        self.lambda_rank  = loss_cfg.get("rank",  0.1)
        self.lambda_align = loss_cfg.get("align", 0.1)
        self.align_enabled = align_cfg.get("enabled", True)
        self.temperature  = align_cfg.get("temperature", 0.1)
        self.mse = nn.MSELoss()

    # ------------------------------------------------------------------
    # ℒ_rank : RankNet pairwise loss
    # ------------------------------------------------------------------
    @staticmethod
    def _rank_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        """
        RankNet pairwise loss（對所有 y_i > y_j 的配對）：
            ℒ_rank = Σ log(1 + exp(-(ŷ_i - ŷ_j)))

        Shapes:
            y_hat, y : [n] 或 [B, n]

        Returns:
            scalar loss
        """
        # 統一升成 [..., n]
        # 計算所有配對差
        # diff_hat[..., i, j] = ŷ_i - ŷ_j
        diff_hat = y_hat.unsqueeze(-1) - y_hat.unsqueeze(-2)   # [..., n, n]
        diff_y   = y.unsqueeze(-1)     - y.unsqueeze(-2)       # [..., n, n]
        # 只取 y_i > y_j 的配對（上三角，差 > 0）
        mask = (diff_y > 0).float()
        loss = F.softplus(-diff_hat)   # log(1 + exp(-x))，等同 RankNet
        return (loss * mask).sum() / (mask.sum().clamp(min=1.0))

    # ------------------------------------------------------------------
    # ℒ_align : InfoNCE contrastive loss
    # ------------------------------------------------------------------
    def _align_loss(self, h_L1: Tensor, h_L2: Tensor) -> Tensor:
        """
        InfoNCE 對比損失。
        正樣本 = 同公司 (i, i)，負樣本 = 同 batch 其他公司 (i, j≠i)。

        Shapes:
            h_L1 : [n, d'] 或 [B, n, d']  (已假設 batch 維度已 flatten 或為單一快照)
            h_L2 : 同上

        Returns:
            scalar loss
        """
        # 支援有無 batch 維度
        if h_L1.dim() == 3:
            # [B, n, d'] → [B*n, d'] 以 B 個快照的 n 節點作為 batch
            B, n, d = h_L1.shape
            h1 = h_L1.reshape(B * n, d)
            h2 = h_L2.reshape(B * n, d)
        else:
            h1 = h_L1   # [n, d']
            h2 = h_L2

        # L2 normalize
        h1 = F.normalize(h1, dim=-1)
        h2 = F.normalize(h2, dim=-1)

        # 相似度矩陣 [N, N]
        sim = torch.matmul(h1, h2.T) / self.temperature
        # 正樣本在對角線
        N = h1.size(0)
        labels = torch.arange(N, device=h1.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        y_hat: Tensor,
        y:     Tensor,
        h_L1:  Tensor | None = None,
        h_L2:  Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """
        Args:
            y_hat : [n] 或 [B, n]  預測 log_return
            y     : [n] 或 [B, n]  真實 log_return
            h_L1  : [..., n, d']   ADR 投影後表示（可選，用於 align loss）
            h_L2  : [..., n, d']   TW 投影後表示

        Returns:
            total_loss : scalar
            components : dict{"mse", "rank", "align"}  各分量（供 logging）
        """
        l_mse  = self.mse(y_hat, y)
        l_rank = self._rank_loss(y_hat, y)

        l_align = torch.tensor(0.0, device=y.device)
        if self.align_enabled and h_L1 is not None and h_L2 is not None:
            l_align = self._align_loss(h_L1, h_L2)

        total = (
            self.lambda_mse   * l_mse
            + self.lambda_rank  * l_rank
            + self.lambda_align * l_align
        )
        return total, {"mse": l_mse.item(), "rank": l_rank.item(), "align": l_align.item()}
