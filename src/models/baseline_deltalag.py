"""
baseline_deltalag.py — M7 External Baseline #5: DeltaLag

Reference:
    Zhou, W. et al. (2025).
    DeltaLag: Learning Dynamic Lead-Lag Patterns in Financial Markets.
    Proceedings of the 6th ACM International Conference on AI in Finance
    (ICAIF 2025).
    Paper:  https://arxiv.org/abs/2511.00390
    ACM:    https://dl.acm.org/doi/10.1145/3768292.3770421

DeltaLag 是首個 end-to-end learning 動態 lead-lag pattern 的方法，透過
sparsified cross-attention 每日自適應識別 leader-lagger 配對 + 對應
lag values，並用 leader 的 lag-aligned raw features 預測 lagger 的
未來 return。

架構（照原論文 §3）：
    Phase 1: Temporal encoding
        X_{u,t} [L, F] → LSTM → X'_{u,t} [L, N]     每支股票獨立跑

    Phase 2: Cross-attention
        Query (target u):    q_{u,t} = X'_{u,t}[-1, :] · W^Q         ∈ ℝ^N
        Key   (leader v):    K_{v,t} = X'_{v,t}[-l_max:, :] · W^K    ∈ ℝ^{l_max × N}
        Attn:                A_{u,v,t} = q · K^T                     ∈ ℝ^{l_max}
        Stack across candidates: A_{u,t} ∈ ℝ^{(n_cand) × l_max}

    Phase 3: TopK sparsification
        {(i_m, j_m)}_{m=1..k} = TopK(A_{u,t})
        leader = candidate v_{i_m}, lag τ_m = l_max - j_m

    Phase 4: Lag-aligned raw feature extraction (關鍵：用 raw 而非 encoded)
        z_{leader,m} = x_{v_{i_m}, t - τ_m, :}   ∈ ℝ^F
        weights = softmax(topk_scores)
        z_{u,t} = Σ weights_m · z_{leader,m}

    Phase 5: Prediction head
        ŷ_{u,t+1} = MLP(z_{u,t})

適配到 ADR-TW 資料集的三個關鍵決策：
    1. Candidate pool: 每個 TW target 有 13 candidates = 7 ADR + 6 其他 TW
       這給 DeltaLag 公平機會動態學到「ADR_i 是 TW_i 最佳 leader、lag=1」
    2. Target: 只預測 TW nodes（因為 ground truth y 是 TW 隔日 return）
    3. l_max = 5：涵蓋 ADR-TW 自然 1-day lag，並留餘裕做動態探索

與 MAGNET 的對照：
    MAGNET: 結構性 leader-lagger pair（ADR_i↔TW_i，domain-verified）+ 結構性 lag=1
    DeltaLag: 動態學習 leader-lagger + 動態學習 lag value

若 MAGNET 勝：驗證「domain-verified structural pairing > learned dynamic pairing」
              為 §6 Discussion 提供第二個「structural vs learned」證據
              （第一個是 MAGNET vs HGT）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.prediction_head import PredictionHead, CombinedLoss


class BaselineDeltaLag(nn.Module):
    """
    DeltaLag baseline: sparsified cross-attention + dynamic lead-lag pair selection.

    Candidate pool 包含 ADR + TW，讓模型有機會動態學到 ADR→TW 是最佳跨市場 leader。
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m_cfg    = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        head_cfg = m_cfg["prediction_head"]

        input_dim = lstm_cfg["input_dim"]
        N         = lstm_cfg["hidden_dim"]
        num_layers = lstm_cfg["num_layers"]

        # DeltaLag hyperparams（可從 cfg.model.delta_lag 覆寫）
        dl_cfg      = m_cfg.get("delta_lag", {})
        self.l_max  = int(dl_cfg.get("l_max", 5))   # 最大 lag 探索範圍
        self.top_k  = int(dl_cfg.get("top_k", 2))   # 每個 target 選 top-k leader-lag pair

        # Phase 1: Temporal encoder（每股獨立 LSTM，回傳完整序列）
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=N,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_cfg.get("dropout", 0.0) if num_layers > 1 else 0.0,
        )

        # Phase 2: Cross-attention 投影（Q for target, K for candidates）
        self.W_Q = nn.Linear(N, N, bias=False)
        self.W_K = nn.Linear(N, N, bias=False)

        # Phase 5: Prediction head 吃 raw feature 空間（原論文 §4：raw yields superior）
        # 加 LayerNorm 適配我們的異質特徵尺度（原論文用 intraday price ratios 全部同尺度，
        # 我們的 9 維含 log_return / RSI 0-100 / MACD signed 等異質尺度，需先正規化）
        self.feature_norm = nn.LayerNorm(input_dim)
        self.predictor = PredictionHead(head_cfg, d_prime=input_dim)

        # 統一 loss（沿用 opt_p2 配方以確保與其他 baseline 公平）
        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

        assert input_dim == 9, (
            f"LSTM input_dim 應與 TECH_FEATURE_COLS 維度一致（9），"
            f"當前為 {input_dim}"
        )

    def _encode_all(self, x_all: Tensor) -> Tensor:
        """
        Args:
            x_all: [B, T, n_all, F]  合併後的 candidate + target 序列
        Returns:
            h_seq: [B, n_all, T, N]  每股 T 步 LSTM 隱狀態
        """
        B, T, n_all, F_ = x_all.shape
        x = x_all.permute(0, 2, 1, 3).reshape(B * n_all, T, F_)  # [B*n_all, T, F]
        h_seq, _ = self.lstm(x)                                   # [B*n_all, T, N]
        return h_seq.reshape(B, n_all, T, -1)                     # [B, n_all, T, N]

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        x_L1 = batch["x_seq_L1"]                                  # [B, T, n, F]  ADR
        x_L2 = batch["x_seq_L2"]                                  # [B, T, n, F]  TW

        B, T, n, F_ = x_L1.shape
        n_all = 2 * n                                             # candidate pool = ADR + TW

        # 合併兩市場序列：候選 pool 前 n 個是 ADR，後 n 個是 TW
        x_all = torch.cat([x_L1, x_L2], dim=2)                    # [B, T, 2n, F]

        # ── Phase 1: Temporal encoding ─────────────────────────
        h_seq = self._encode_all(x_all)                           # [B, 2n, T, N]

        # ── Phase 2: Cross-attention ───────────────────────────
        # Target = TW nodes（後 n 個），indices [n, 2n)
        h_targets = h_seq[:, n:, -1, :]                           # [B, n, N]  last-step for each target
        q = self.W_Q(h_targets)                                   # [B, n, N]

        # Candidates = 全體 2n 個，取最後 l_max 步作 keys
        h_cand_window = h_seq[:, :, -self.l_max:, :]              # [B, 2n, l_max, N]
        k = self.W_K(h_cand_window)                               # [B, 2n, l_max, N]

        # Attention scores: [B, n_target=n, n_cand=2n, l_max]
        # A[b, u, v, j] = q[b, u] · k[b, v, j]
        # 用 einsum 一次算完
        A = torch.einsum("buN,bvjN->buvj", q, k)                  # [B, n, 2n, l_max]

        # Mask self-loops：TW_i 不能是 TW_i 自己的 leader（indices n+i in candidate pool）
        # ADR_i (candidate index i) 可以是 TW_i (target index i) 的 leader → 不 mask
        for i in range(n):
            A[:, i, n + i, :] = float("-inf")                     # target i 對應 candidate n+i (自己)

        # ── Phase 3: TopK sparsification ────────────────────────
        # 攤平 (candidate, lag) 兩個維度後選 top-k
        A_flat = A.reshape(B, n, n_all * self.l_max)              # [B, n, 2n*l_max]
        topk_scores, topk_idx = A_flat.topk(self.top_k, dim=-1)   # [B, n, k]

        # 解碼 flat idx → (candidate_idx, lag_j)
        cand_idx = topk_idx // self.l_max                         # [B, n, k]  in [0, 2n)
        lag_j    = topk_idx %  self.l_max                         # [B, n, k]  in [0, l_max)

        # 轉為原論文 τ = l_max - j（j=l_max-1 對應 τ=1，即昨天）
        # feature 時間 = T - 1 - τ = T - 1 - (l_max - lag_j) = T - 1 - l_max + lag_j
        feat_time = T - 1 - self.l_max + lag_j                    # [B, n, k]

        # ── Phase 4: Lag-aligned raw feature extraction ─────────
        # 對 (b, u, m) 取 x_all[b, feat_time[b,u,m], cand_idx[b,u,m], :]
        # 用 advanced indexing 向量化
        b_idx = torch.arange(B, device=x_all.device).view(B, 1, 1).expand(B, n, self.top_k)
        z_leaders = x_all[b_idx, feat_time, cand_idx]             # [B, n, k, F]

        # Softmax weights over top-k
        weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)    # [B, n, k, 1]
        z = (weights * z_leaders).sum(dim=2)                      # [B, n, F]

        # ── Phase 5: Prediction ────────────────────────────────
        z = self.feature_norm(z)                                  # 正規化異質特徵尺度
        y_hat = self.predictor(z)                                 # [B, n]

        # 對齊 MAGNET 簽名（extras 不用於 loss；額外欄位供分析）
        extras = {
            "h_L1":    z,
            "h_L2":    z,
            "h_fused": z,
            "alpha":   weights.squeeze(-1).mean(dim=-1, keepdim=True),  # [B, n, 1]
            "gate":    torch.zeros_like(z),
            # 分析用：學到的 leader 分佈（看 DeltaLag 是否選中 ADR）
            "topk_cand_idx": cand_idx,                            # [B, n, k]
            "topk_lag":      self.l_max - lag_j,                  # [B, n, k]  (τ values)
            "topk_scores":   topk_scores,                         # [B, n, k]
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
