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
        self.lambda_mse      = loss_cfg.get("mse",      1.0)
        self.lambda_rank     = loss_cfg.get("rank",     0.1)
        self.lambda_align    = loss_cfg.get("align",    0.1)
        # M4.5 Phase 2: variance penalty 對抗 prediction collapse
        # 罰「ŷ 標準差遠小於 y 標準差」的塌縮模式
        self.lambda_variance = loss_cfg.get("variance", 0.0)
        self.align_enabled = align_cfg.get("enabled", True)
        self.temperature  = align_cfg.get("temperature", 0.1)
        self.mse = nn.MSELoss()

        # rank_normalize：配對比較前先把 ŷ 逐日橫截面標準化。
        #
        # 為什麼需要：softplus(-Δŷ) 吃的是原始報酬的差。實測已訓練模型的
        # ŷ 橫截面 std 只有 0.0039，典型 Δŷ ≈ 0.005，落在 softplus 的線性區——
        # 排對的配對梯度 -0.4986、排錯的 -0.5，幾乎沒有差別。RankNet 該有的
        # 「已排對的配對飽和、把力氣集中到難配對」完全沒發生，損失值全程
        # 停在 ln(2)=0.6931（實測 0.692986，差 0.00016），而它佔總損失 99.94%。
        # 標準化後 Δ 變成 O(1)，且與 IC 一樣尺度不變，訓練目標才與評估指標對齊。
        #
        # 預設 False：開啟會改變所有既有 run 的數值，凍結基準與 e7_acceptance
        # 的位元確定路徑必須維持不變。要用請在 config 的 loss_weights 下明示。
        self.rank_normalize = bool(loss_cfg.get("rank_normalize", False))

    # ------------------------------------------------------------------
    # ℒ_rank : RankNet pairwise loss
    # ------------------------------------------------------------------
    def _rank_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        RankNet pairwise loss（對所有 y_i > y_j 的配對）：
            ℒ_rank = Σ log(1 + exp(-(ŷ_i - ŷ_j)))

        rank_normalize=True 時先把 ŷ 逐日橫截面標準化（見 __init__ 說明）。
        y 不需標準化——配對遮罩用的是 diff_y > 0，本來就尺度不變。

        Shapes:
            y_hat, y : [n] 或 [B, n]

        Returns:
            scalar loss
        """
        if self.rank_normalize:
            y_hat = ((y_hat - y_hat.mean(dim=-1, keepdim=True))
                     / (y_hat.std(dim=-1, keepdim=True) + 1e-8))

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
        # E6：正樣本定義為「第 i 列的 h_L1 與 h_L2 是同一家公司」，
        # 這在 n1 == n2 且全配對時才成立。擴充後 L1 有 30 個節點、L2 有 50 個，
        # 只有 7 對是同公司，對角線不再是正樣本集合——這需要換一個
        # 對比損失的設計（只在有配對的 7 對上算），不是形狀對齊就能解決。
        # 目前 base.yaml 是 align_loss.enabled=false / align=0.0，故先擋住，
        # 不要讓它靜默算出一個意義錯誤的數字。
        if h_L1.shape != h_L2.shape:
            raise ValueError(
                f"align loss 需要 h_L1 與 h_L2 同形狀（正樣本在對角線），"
                f"當前為 {tuple(h_L1.shape)} vs {tuple(h_L2.shape)}。"
                f"不對稱 universe 下請維持 align_loss.enabled=false。"
            )

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
    # ℒ_variance : 預測標準差對齊真實標準差（M4.5 Phase 2）
    # ------------------------------------------------------------------
    @staticmethod
    def _variance_loss(y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Variance alignment penalty。

        動機：純 MSE 鼓勵「預測接近平均值」的塌縮解（prediction collapse）。
        在 M4.5 Phase 1 觀察到 std(ŷ)/std(y) ≈ 0.14，模型實質上輸出近常數。

        定義：對每個 batch 中的每個 sample（一張快照），計算 ŷ 和 y 跨 n 檔
        股票的 std，懲罰兩者差距。

            ℒ_var = E_b [ (std_n(ŷ_b) - std_n(y_b))² ]

        Shapes:
            y_hat, y : [n] 或 [B, n]

        Returns:
            scalar loss（已對 batch 平均）
        """
        # 統一升維到 [B, n]
        if y_hat.dim() == 1:
            y_hat = y_hat.unsqueeze(0)
            y     = y.unsqueeze(0)
        # 對 n 維算 std（即一張快照內 7 檔股票的橫截面 std）
        std_yh = y_hat.std(dim=-1, unbiased=False)   # [B]
        std_y  = y.std(dim=-1,     unbiased=False)   # [B]
        return ((std_yh - std_y) ** 2).mean()

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
            components : dict{"mse", "rank", "align", "variance"}  各分量（供 logging）
        """
        l_mse  = self.mse(y_hat, y)
        l_rank = self._rank_loss(y_hat, y)
        l_var  = self._variance_loss(y_hat, y)

        l_align = torch.tensor(0.0, device=y.device)
        if self.align_enabled and h_L1 is not None and h_L2 is not None:
            l_align = self._align_loss(h_L1, h_L2)

        total = (
            self.lambda_mse      * l_mse
            + self.lambda_rank     * l_rank
            + self.lambda_align    * l_align
            + self.lambda_variance * l_var
        )
        return total, {
            "mse":      l_mse.item(),
            "rank":     l_rank.item(),
            "align":    l_align.item(),
            "variance": l_var.item(),
        }
