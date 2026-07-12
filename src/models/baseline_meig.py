"""
baseline_meig.py — M7 External Baseline #6: MEIG-core

Reference:
    Bukhari, M., Maqsood, M., & Sattar, A. (2025).
    A novel inter-intra graph neural networks for stock price forecasting
    modeling cross-border relationships.
    Expert Systems with Applications, 286, 127907.
    https://doi.org/10.1016/j.eswa.2025.127907

MEIG（Macro-Event Driven Inter-Intra Graph）是原生跨市場的 stock-level
GNN：每個市場建 intra-graph、跨市場建 inter-graph（相關係數門檻篩邊，
原文 Eq. 6），三條平行 GCN branch 的輸出由 CGAT（Cross-Graph Attention
Layer）加權混合。CGAT 的實際機制（原文 Eq. 21-23）是 3 個自由可學純量
經 softmax 後對三條 branch 輸出做加權和 —— graph-level、與輸入無關。

架構（照原文 §3，pipeline 順序為 GCN 先、LSTM 後）：
    Phase 1: 相關圖構建（Eq. 6）
        corr(u, v) > θ 則建邊；intra 圖各市場一張、inter 圖只含跨市場邊
    Phase 2: 三條平行 spectral GCN（Eq. 13: H' = ReLU(A_hat H W)）
        Z_L1 = GCN_intra_L1(X)，Z_L2 = GCN_intra_L2(X)，Z_x = GCN_inter(X)
        每張圖為同一 2n 節點集上的不同 adjacency（intra 圖對他市場節點
        只留 self-loop），以 dense 正規化鄰接矩陣實作（14 節點不需稀疏化，
        且貼近原文譜式 GCN 形式、迴避 PyG 稀疏算子的 MPS 相容問題）
    Phase 3: CGAT（Eq. 21-23）
        alpha = Softmax(w)，w ∈ R^3 自由參數
        Z = alpha_1·Z_L1 + alpha_2·Z_L2 + alpha_3·Z_x
    Phase 4: LSTM 逐節點處理時間維度（§3.5）
    Phase 5: Prediction head（只預測 TW 節點的隔日報酬）

適配到 ADR-TW 資料集的關鍵決策（皆需在論文中 disclose）：
    1. 節點特徵僅用 9 維技術指標：原文的總經指標（GDP 等 5 項）與
       42.7M tweets 事件情緒無法取得，且為維持 9 模型輸入完全一致而
       移除。原文 Table 5-8 自帶 Macro=x 的 ablation rows 可引為前例
       （同 MAN-SF no-text 的處理方式）。
    2. 預測目標由「價格水準 + MSE」改為「隔日報酬 + 共用 CombinedLoss」，
       與其他 baseline 完全相同的訓練配方以確保公平。
    3. 相關係數在輸入窗（T=20 天）內以 log_return 現算：原文未揭露
       相關係數的估計窗口，本實作選擇無 lookahead 的窗內估計，且用
       報酬相關而非價格水準相關（短窗價格相關受趨勢污染）。
    4. 門檻值 theta 原文未揭露，預設 0.3（可由 cfg.model.meig 覆寫）。
    5. LSTM 沿用共用 lstm 配置（hidden 64）取代原文的 64/32 雙層堆疊，
       維持與其他 baseline 的容量一致性。

與 MAGNET 的對照：
    MEIG:   跨市場邊 = 統計門檻（易受 20 天窗雜訊影響）
            混合權重 = 3 個 graph-level 純量（訓練後固定、與輸入無關）
    MAGNET: 跨市場邊 = 雙掛牌身分（結構事實、零估計誤差）
            混合權重 = per-node per-dim gate（由當下表徵即時算出）

若 MAGNET 勝：驗證「statistical edge discovery + coarse graph-level
mixing < structural pairing + fine-grained input-dependent gate」，
補上 baseline suite 中唯一原生跨市場方法的直接對比。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.prediction_head import PredictionHead, CombinedLoss


class BaselineMEIG(nn.Module):
    """
    MEIG-core baseline: inter-intra correlation graphs + 3-branch GCN + CGAT.

    只實作 graph + CGAT 核心（modality-reduced）：總經與事件情緒節點特徵
    因資料不可得且需維持輸入一致性而移除。
    """

    # log_return 在 TECH_FEATURE_COLS 的位置（用於窗內相關係數估計）
    CORR_FEATURE_IDX = 0

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m_cfg    = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        head_cfg = m_cfg["prediction_head"]

        input_dim  = lstm_cfg["input_dim"]
        lstm_hidden = lstm_cfg["hidden_dim"]
        num_layers  = lstm_cfg["num_layers"]

        # MEIG hyperparams（可從 cfg.model.meig 覆寫）
        meig_cfg = m_cfg.get("meig", {})
        self.theta_intra = float(meig_cfg.get("theta_intra", 0.3))
        self.theta_inter = float(meig_cfg.get("theta_inter", 0.3))
        gcn_hidden       = int(meig_cfg.get("gcn_hidden", 64))
        # 原文 §3.6「three parallel graph convolution layers」= 每 branch 1 層
        gcn_layers       = int(meig_cfg.get("gcn_layers", 1))

        # Phase 2: 三條平行 GCN branch 的權重（每層一個 Linear）
        def _branch() -> nn.ModuleList:
            dims = [input_dim] + [gcn_hidden] * gcn_layers
            return nn.ModuleList(
                nn.Linear(dims[i], dims[i + 1]) for i in range(gcn_layers)
            )
        self.gcn_intra_L1 = _branch()
        self.gcn_intra_L2 = _branch()
        self.gcn_inter    = _branch()

        # Phase 3: CGAT —— 3 個自由可學純量（Eq. 22: alpha = Softmax(w)）
        self.cgat_w = nn.Parameter(torch.zeros(3))

        # Phase 4: LSTM 在 GCN 之後處理時間維度（原文 §3.5 的順序）
        self.lstm = nn.LSTM(
            input_size=gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_cfg.get("dropout", 0.0) if num_layers > 1 else 0.0,
        )

        # Phase 5: Prediction head（共用 head 配置，吃 LSTM 最後 hidden）
        self.predictor = PredictionHead(head_cfg, d_prime=lstm_hidden)

        # 統一 loss（沿用 opt_p2 配方以確保與其他 baseline 公平）
        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

        assert input_dim == 9, (
            f"LSTM input_dim 應與 TECH_FEATURE_COLS 維度一致（9），"
            f"當前為 {input_dim}"
        )

        # 節點區塊 mask（前 n 個 = ADR，後 n 個 = TW）延後到第一次 forward
        # 依實際 n 建立並 cache（register_buffer 需知 n，資料層固定 n=7）
        self._masks_built_for_n: int | None = None

    # ------------------------------------------------------------------
    # 圖構建 helpers
    # ------------------------------------------------------------------

    def _build_block_masks(self, n: int, device: torch.device) -> None:
        """建立三張圖的節點區塊 mask（[2n, 2n] bool，不含對角線）。"""
        n_all = 2 * n
        idx = torch.arange(n_all, device=device)
        is_adr = idx < n                                          # [2n]
        eye = torch.eye(n_all, dtype=torch.bool, device=device)

        intra1 = is_adr.unsqueeze(0) & is_adr.unsqueeze(1) & ~eye   # ADR 區塊
        intra2 = ~is_adr.unsqueeze(0) & ~is_adr.unsqueeze(1) & ~eye # TW 區塊
        inter  = is_adr.unsqueeze(0) ^ is_adr.unsqueeze(1)          # 跨區塊

        self.register_buffer("intra1_mask", intra1, persistent=False)
        self.register_buffer("intra2_mask", intra2, persistent=False)
        self.register_buffer("inter_mask",  inter,  persistent=False)
        self._masks_built_for_n = n

    @staticmethod
    def _normalize_adj(edges: Tensor) -> Tensor:
        """
        Eq. 13 的對稱正規化：A_hat = D^{-1/2} (A + I) D^{-1/2}。

        Args:
            edges: [B, 2n, 2n] bool，門檻篩過的邊（不含 self-loop）
        Returns:
            [B, 2n, 2n] float 正規化 adjacency
        """
        n_all = edges.size(-1)
        A = edges.float() + torch.eye(n_all, device=edges.device)
        deg = A.sum(dim=-1)                                       # [B, 2n]
        d_inv_sqrt = deg.pow(-0.5)                                # deg >= 1（有 self-loop）
        return d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)

    def _gcn_branch(self, layers: nn.ModuleList, A_hat: Tensor, x: Tensor) -> Tensor:
        """
        單條 branch 的 spectral GCN：H' = ReLU(A_hat H W)，逐時步套用。

        Args:
            A_hat: [B, 2n, 2n]  正規化 adjacency
            x:     [B, T, 2n, F]
        Returns:
            [B, T, 2n, H]
        """
        h = x
        for lin in layers:
            h = lin(h)                                            # [B, T, 2n, H]
            h = torch.einsum("buv,btvh->btuh", A_hat, h)
            h = F.relu(h)
        return h

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        x_L1 = batch["x_seq_L1"]                                  # [B, T, n, F]  ADR
        x_L2 = batch["x_seq_L2"]                                  # [B, T, n, F]  TW

        B, T, n, _ = x_L1.shape
        x_all = torch.cat([x_L1, x_L2], dim=2)                    # [B, T, 2n, F]

        if self._masks_built_for_n != n:
            self._build_block_masks(n, x_all.device)

        # ── Phase 1: 窗內相關圖（Eq. 6: corr > theta 建邊） ─────────
        r = x_all[..., self.CORR_FEATURE_IDX]                     # [B, T, 2n] log_return
        r = r - r.mean(dim=1, keepdim=True)
        r = r / (r.std(dim=1, keepdim=True, unbiased=False) + 1e-8)
        corr = torch.einsum("btu,btv->buv", r, r) / T             # [B, 2n, 2n] Pearson

        A1 = self._normalize_adj((corr > self.theta_intra) & self.intra1_mask)
        A2 = self._normalize_adj((corr > self.theta_intra) & self.intra2_mask)
        Ax = self._normalize_adj((corr > self.theta_inter) & self.inter_mask)

        # ── Phase 2: 三條平行 GCN branch（Eq. 21） ──────────────────
        Z1 = self._gcn_branch(self.gcn_intra_L1, A1, x_all)       # [B, T, 2n, H]
        Z2 = self._gcn_branch(self.gcn_intra_L2, A2, x_all)
        Zx = self._gcn_branch(self.gcn_inter,    Ax, x_all)

        # ── Phase 3: CGAT graph-level 混合（Eq. 22-23） ─────────────
        alpha = torch.softmax(self.cgat_w, dim=0)                 # [3]
        Z = alpha[0] * Z1 + alpha[1] * Z2 + alpha[2] * Zx         # [B, T, 2n, H]

        # ── Phase 4: LSTM 逐節點處理時間（§3.5） ────────────────────
        n_all = 2 * n
        z_seq = Z.permute(0, 2, 1, 3).reshape(B * n_all, T, -1)   # [B*2n, T, H]
        h_seq, _ = self.lstm(z_seq)                               # [B*2n, T, N]
        h = h_seq[:, -1, :].reshape(B, n_all, -1)                 # [B, 2n, N]

        # ── Phase 5: Prediction（只出 TW 節點） ─────────────────────
        h_adr, h_tw = h[:, :n, :], h[:, n:, :]
        y_hat = self.predictor(h_tw)                              # [B, n]

        # 對齊 MAGNET 簽名（extras 不用於 loss；額外欄位供分析）
        extras = {
            "h_L1":    h_adr,
            "h_L2":    h_tw,
            "h_fused": h_tw,
            "alpha":   alpha[2].expand(B, n, 1),                  # α_inter（分析用）
            "gate":    torch.zeros_like(h_tw),
            # 分析用：CGAT 三權重與 inter 圖的邊密度
            "cgat_alpha": alpha,                                  # [3] (L1, L2, inter)
            "inter_edge_density": (
                ((corr > self.theta_inter) & self.inter_mask).float().mean()
            ),
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
