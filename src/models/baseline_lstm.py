"""
baseline_lstm.py — Stage 0 Ablation: LSTM-only baseline（無圖、無 ADR）

對應 M6 Plan Stage 0 #1 (opt_p17_baseline_lstm)
目的：驗證「圖結構是否真的提供額外訊號」。
若 MAGNET test_IC >> baseline_lstm test_IC，則證明 GNN 有實質貢獻。

架構：
    x_seq_L2 [B, T, n, F]
        → SharedLSTM        → [B, n, H_lstm]
        → TypeProjection    → [B, n, d']
        → PredictionHead    → [B, n]

注意：
  - 僅使用 TW 端 (L2) 序列輸入，完全不看 ADR；屬於「圖 + 跨市場資訊」都拿掉的純時序 baseline
  - forward 介面與 MAGNET 對齊（接同一個 batch dict，回傳 (y_hat, extras)）
  - extras 含 h_L2，h_L1 = h_fused = h_L2（讓既有 evaluator / loss 介面通用）
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.encoders import SharedLSTM, TypeProjection
from src.models.prediction_head import PredictionHead, CombinedLoss


class BaselineLSTM(nn.Module):
    """LSTM-only baseline（無圖、無跨市場融合）。"""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m_cfg    = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        proj_cfg = m_cfg["projection"]
        head_cfg = m_cfg["prediction_head"]

        d_prime = proj_cfg["d_prime"]
        H_lstm  = lstm_cfg["hidden_dim"]

        self.lstm = SharedLSTM(lstm_cfg)
        self.proj = TypeProjection(proj_cfg, in_dim=H_lstm)
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

        h = self.lstm(x_L2)                 # [B, n, H_lstm]
        h = self.proj(h)                    # [B, n, d']
        y_hat = self.head(h)                # [B, n]

        extras = {
            "h_L1":    h,    # 對齊 MAGNET 簽名（讓 align_loss/evaluator 不爆，數值無意義）
            "h_L2":    h,
            "h_fused": h,
            "alpha":   torch.zeros(*h.shape[:-1], 1, device=h.device),
            "gate":    torch.zeros_like(h),
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
