"""
baseline_early_fusion.py — Early-fusion baseline（輸入層特徵拼接，無圖、無閘門）

存在理由（回應口試提問「為什麼不做 early fusion」）：
    既有三個 Stage 0 ablation（baseline_lstm / baseline_tw_gnn / magnet_no_a12）
    全部都是「拿掉 ADR」，沒有一個是「用最笨的方式使用 ADR」。
    使用 ADR 最簡單的做法就是把配對美股的特徵直接接在台股特徵後面，
    跑一個普通 LSTM——若這樣就贏過 MAGNET，兩級耦合架構即失去正當性。
    本檔提供該對照組。

架構：
    x_seq_L2 [B, T, n2, F]  ─┐
                             ├─ concat → [B, T, n2, 2F]
    x_seq_L1 gather p(j)  ───┘   （無配對節點的 ADR 半邊補零）
        → SharedLSTM(input_dim=2F) → [B, n2, H_lstm]
        → TypeProjection            → [B, n2, d']
        → PredictionHead            → [B, n2]

與 MAGNET 的關鍵差異：
  - 融合發生在「輸入層」而非潛在空間，故無法表達「這條連結是已知且固定的」——
    模型學到的是 [x_TW ; x_ADR] 上的稠密權重，等同全自由學習
  - 沒有層間邊的概念，因此沒有恆等/候選兩級之分，也沒有 g / B_eff 可讀出
  - 沒有任何圖結構（層內、層間皆無）

注意：
  - forward 介面與 MAGNET 對齊（同一個 batch dict，回傳 (y_hat, extras)）
  - p(j) 由 cfg 的 universe 推得，不假設 L1/L2 索引對齊；
    universe 擴充後多數 TW 節點無配對，ADR 半邊即為零向量
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models._universe import universe_from_cfg
from src.models.encoders import SharedLSTM, TypeProjection
from src.models.prediction_head import PredictionHead, CombinedLoss


class BaselineEarlyFusion(nn.Module):
    """Early-fusion baseline（輸入層拼接 ADR 特徵）。"""

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

        # 拼接後特徵維度為 2F——這是架構推導，非可調超參
        fused_lstm_cfg = {**lstm_cfg, "input_dim": F_base * 2}

        self.lstm = SharedLSTM(fused_lstm_cfg)
        self.proj = TypeProjection(proj_cfg, in_dim=H_lstm)
        self.head = PredictionHead(head_cfg, d_prime=d_prime)

        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

        # p(j)：TW 節點 j → 其配對 ADR 在 L1 的索引；無配對為 -1
        # E6：改由 cfg 的 universe 取得。原本從模組層級的 k7 常數推導，
        # 擴充後會產出長度 7 的索引去切 50 節點的張量。
        pair_index = universe_from_cfg(cfg).pair_index
        self.register_buffer(
            "pair_src",
            torch.tensor([max(i, 0) for i in pair_index], dtype=torch.long),
        )
        self.register_buffer(
            "has_pair",
            torch.tensor([i >= 0 for i in pair_index], dtype=torch.bool),
        )

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        x_L1 = batch["x_seq_L1"]            # [B, T, n1, F]
        x_L2 = batch["x_seq_L2"]            # [B, T, n2, F]

        # 依 p(j) 取出每個 TW 節點對應的 ADR 序列；無配對者補零
        # （pair_src 已把 -1 填成 0，實際由 has_pair 遮掉）
        adr = x_L1.index_select(dim=2, index=self.pair_src)      # [B, T, n2, F]
        adr = adr * self.has_pair.view(1, 1, -1, 1).to(adr.dtype)

        x = torch.cat([x_L2, adr], dim=-1)  # [B, T, n2, 2F]

        h = self.lstm(x)                    # [B, n2, H_lstm]
        h = self.proj(h)                    # [B, n2, d']
        y_hat = self.head(h)                # [B, n2]

        extras = {
            "h_L1":    h,    # 對齊 MAGNET 簽名（align_loss/evaluator 通用，數值無意義）
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
