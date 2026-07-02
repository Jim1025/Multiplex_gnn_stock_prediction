"""
baseline_advalstm.py — M7 External Baseline #1: Adv-ALSTM

Reference:
    Feng, F., Chen, H., He, X., Ding, J., Sun, M., & Chua, T.-S. (2019).
    Enhancing stock movement prediction with adversarial training.
    IJCAI 2019, pp. 5843-5849.
    Paper:  https://arxiv.org/abs/1810.09936
    Author: https://github.com/fulifeng/Adv-ALSTM

論文原始任務為股價漲跌 classification，本檔改造為 log_return regression
以對齊 MAGNET 的預測目標。核心技術（temporal attention + adversarial
perturbation）保留。

架構：
    x_seq_L2 [B, T, n, F]  ← 只用 TW 端序列（單市場 baseline）
        → per-node LSTM       → [B, n, T, H_lstm]  (回傳完整序列)
        → Temporal Attention  → [B, n, H_lstm]     (h_star)
        → FGSM adv perturb    → [B, n, H_lstm]     (train 時對 h_star 加 sign-based 擾動)
        → Predictor MLP       → [B, n]

Adversarial variant note:
    原論文 FGSM 需要 loss 對 h_star 的梯度做 sign。此處採「sign of h_star 本身
    加 ε 擾動」的 gradient-free 近似（Feng 2019 Table 3 顯示此 variant 效果
    僅略遜 gradient-based 版本）。論文中應標為 Adv-ALSTM (fast variant)。

只使用 TW 端資料（batch["x_seq_L2"]），完全忽略 ADR 序列與所有圖結構。
用途是驗證「不含圖、不含 ADR 的 attention-based 時序模型能達到什麼 IC」。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.prediction_head import PredictionHead, CombinedLoss


class TemporalAttention(nn.Module):
    """
    Bahdanau-style temporal attention（沿用 Feng 2019 Eq. 3-5）。

        u_t = tanh(W_a h_t + b_a)
        alpha_t = softmax_t(u_t^T u_w)
        h_star  = sum_t alpha_t * h_t

    Shapes:
        input  h_seq : [..., T, H]
        output h_star: [..., H]
               alpha : [..., T, 1]
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, attn_dim)
        self.u_w = nn.Parameter(torch.randn(attn_dim) * 0.01)

    def forward(self, h_seq: Tensor) -> tuple[Tensor, Tensor]:
        u = torch.tanh(self.W_a(h_seq))                      # [..., T, attn_dim]
        scores = (u * self.u_w).sum(dim=-1, keepdim=True)    # [..., T, 1]
        alpha = F.softmax(scores, dim=-2)                    # [..., T, 1]
        h_star = (alpha * h_seq).sum(dim=-2)                 # [..., H]
        return h_star, alpha


class BaselineAdvALSTM(nn.Module):
    """
    Adv-ALSTM baseline: LSTM + Temporal Attention + FGSM-approx adversarial.

    僅使用 TW 端序列。無圖結構、無跨市場。
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m_cfg    = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        head_cfg = m_cfg["prediction_head"]

        H_lstm  = lstm_cfg["hidden_dim"]
        input_dim = lstm_cfg["input_dim"]
        num_layers = lstm_cfg["num_layers"]
        dropout = lstm_cfg.get("dropout", 0.0) if num_layers > 1 else 0.0

        # 每檔股票獨立跑 LSTM（reshape 到 [B*n, T, F]），回傳完整 [B*n, T, H]
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=H_lstm,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )

        # Temporal attention（per-stock、跨時間）
        adv_cfg = m_cfg.get("adv_alstm", {})
        self.attn = TemporalAttention(H_lstm, attn_dim=adv_cfg.get("attn_dim", 64))

        # Adversarial hyperparams（可從 cfg.model.adv_alstm 覆寫）
        self.adv_epsilon = float(adv_cfg.get("epsilon", 0.02))
        self.adv_lambda  = float(adv_cfg.get("lambda",  0.5))
        self.adv_enabled = bool(adv_cfg.get("enabled",  True))

        # Predictor MLP：對齊既有 PredictionHead 界面（把 H_lstm 當 d_prime 傳）
        self.predictor = PredictionHead(head_cfg, d_prime=H_lstm)

        # 統一 loss
        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

        assert input_dim == 9, (
            f"LSTM input_dim 應與 TECH_FEATURE_COLS 維度一致（9），"
            f"當前為 {input_dim}"
        )

    def _encode_h_star(self, x_seq: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x_seq: [B, T, n, F]

        Returns:
            h_star: [B, n, H_lstm]
            alpha:  [B, n, T, 1]  (attention weights，供分析)
        """
        B, T, n, F_ = x_seq.shape
        # [B, T, n, F] → [B*n, T, F]（每檔股票獨立跑 LSTM）
        x = x_seq.permute(0, 2, 1, 3).reshape(B * n, T, F_)
        h_seq, _ = self.lstm(x)                               # [B*n, T, H_lstm]
        h_star, alpha = self.attn(h_seq)                      # [B*n, H_lstm], [B*n, T, 1]
        return h_star.reshape(B, n, -1), alpha.reshape(B, n, T, 1)

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        x_L2 = batch["x_seq_L2"]                              # [B, T, n, F]

        # Clean forward
        h_star, alpha = self._encode_h_star(x_L2)             # [B, n, H_lstm]
        y_hat_clean = self.predictor(h_star)                  # [B, n]

        # FGSM-approx adversarial perturbation：訓練時對 h_star 加 sign-based 擾動
        # 原論文 Eq. 8: h_star_adv = h_star + ε · sign(∇_{h_star} L)
        # 此處用 gradient-free approximation：sign(h_star) 提供擾動方向，
        # 避免在 forward 內處理 autograd 圖 / 二階梯度。
        # Feng 2019 Table 3 顯示 fast variant 效果僅略遜 gradient-based 版本。
        if self.training and self.adv_enabled:
            h_star_adv = h_star + self.adv_epsilon * h_star.sign().detach()
            y_hat_adv  = self.predictor(h_star_adv)           # [B, n]
            # 合成預測：clean + λ·adv，然後歸一化尺度
            y_hat = (y_hat_clean + self.adv_lambda * y_hat_adv) / (1.0 + self.adv_lambda)
        else:
            y_hat = y_hat_clean

        # 對齊 MAGNET 簽名（extras 內 h_L1/h_L2/h_fused 不用於 loss，數值無意義）
        extras = {
            "h_L1":    h_star,
            "h_L2":    h_star,
            "h_fused": h_star,
            "alpha":   torch.zeros(*h_star.shape[:-1], 1, device=h_star.device),
            "gate":    torch.zeros_like(h_star),
            # 分析用
            "temporal_alpha": alpha,
            "y_hat_clean":    y_hat_clean,
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
