"""
baseline_dense_fusion.py — 稠密 early fusion 對照組

存在理由：
    baseline_early_fusion 是「稀疏」early fusion——每檔台股只接**自己配對的
    那一檔 ADR**。tw50 下這代表 30 檔美股裡有 23 檔資訊源完全沒被用到，
    而且 43 檔無配對台股的美股半邊是零向量。

    這個限制的代價量得出來（k=50、246 測試日）：

        ridge 只用 7 檔 ADR       test IC +0.049
        baseline_early_fusion     test IC +0.051   ← 幾乎相同
        ridge 用全部 30 檔        test IC +0.076

    兩者吃的是同一條資訊通道，差距完全來自那 23 檔沒被用到的資訊源。

    本檔把全部 n1 檔美股的特徵接到每一檔台股上，是 ridge 的神經網路對應版。
    它把兩件事分開：

        稠密 EF ≈ +0.076   差距純粹來自輸入涵蓋範圍，圖結構與兩級耦合多餘
        稠密 EF <  +0.076   差距來自神經網路本身的負擔（容量、目標函數）
        稠密 EF >  +0.076   非線性有價值，MAGNET 的方向沒錯只是實作要修

架構：
    x_seq_L2 [B, T, n2, F]                      ─┐
                                                 ├─ concat → [B, T, n2, F + n1*F]
    x_seq_L1 [B, T, n1, F] 攤平後廣播給每個 j  ─┘
        → SharedLSTM(input_dim = F + n1*F) → [B, n2, H]
        → TypeProjection                    → [B, n2, d']
        → PredictionHead                    → [B, n2]

與稀疏版的唯一差別是美股半邊取哪些節點：稀疏版 gather p(j)，本檔取全部。
不涉及圖、閘門、兩級耦合，與稀疏版同屬 early fusion。

注意：
    LSTM 輸入維度隨 n1 線性成長（k7 為 9+63=72，tw50 為 9+270=279），
    參數量因此遠大於稀疏版。做比較時必須揭露——若本檔勝出，要先排除
    「只是模型變大」的解釋，這正是容量掃描要處理的問題。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models._universe import universe_from_cfg
from src.models.encoders import SharedLSTM, TypeProjection
from src.models.prediction_head import PredictionHead, CombinedLoss


class BaselineDenseFusion(nn.Module):
    """稠密 early fusion：每檔台股接全部美股節點的特徵。"""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m_cfg    = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        proj_cfg = m_cfg["projection"]
        head_cfg = m_cfg["prediction_head"]

        F_base  = lstm_cfg["input_dim"]
        d_prime = proj_cfg["d_prime"]
        H_lstm  = lstm_cfg["hidden_dim"]

        assert F_base == 9, (
            f"SharedLSTM input_dim 應與 TECH_FEATURE_COLS 維度一致（9），"
            f"當前為 {F_base}"
        )

        u = universe_from_cfg(cfg)
        self.n_l1, self.n_l2 = u.n_l1, u.n_l2

        # 拼接後維度為 F + n1*F——架構推導，非可調超參
        fused_dim = F_base + self.n_l1 * F_base
        self.lstm = SharedLSTM({**lstm_cfg, "input_dim": fused_dim})
        self.proj = TypeProjection(proj_cfg, in_dim=H_lstm)
        self.head = PredictionHead(head_cfg, d_prime=d_prime)

        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        x_L1 = batch["x_seq_L1"]            # [B, T, n1, F]
        x_L2 = batch["x_seq_L2"]            # [B, T, n2, F]
        B, T, n1, F_ = x_L1.shape
        n2 = x_L2.size(2)
        if (n1, n2) != (self.n_l1, self.n_l2):
            raise ValueError(
                f"batch 節點數 L1={n1} / L2={n2} 與 cfg 的 universe"
                f"（L1={self.n_l1} / L2={self.n_l2}）不符。"
            )

        # 攤平全部美股節點後廣播給每一個台股節點
        us = x_L1.reshape(B, T, 1, n1 * F_).expand(B, T, n2, n1 * F_)
        x = torch.cat([x_L2, us], dim=-1)   # [B, T, n2, F + n1*F]

        h = self.lstm(x)                    # [B, n2, H]
        h = self.proj(h)                    # [B, n2, d']
        y_hat = self.head(h)                # [B, n2]

        extras = {
            "h_L1":    h,   # 對齊 MAGNET 簽名（數值無意義，本模型無獨立 L1 表示）
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
