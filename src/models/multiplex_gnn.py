"""
multiplex_gnn.py — MAGNET 主類別
MAGNET: Multiplex ADR-Guided Network for Equity Trading

Corresponds to IMPLEMENTATION_SPEC §0 / §8.1

三階段架構：
  Phase 1: Dual-Market Graph Encoding
           SharedLSTM → GAT_L1/GAT_L2 → Proj_L1/Proj_L2
  Phase 2: Gated Cross-Market Fusion
           CrossLayerFusion (per-node attention + gate)
  Phase 3: Multi-Objective Return Prediction
           PredictionHead (MLP)

使用方式：
    import yaml
    from src.models.multiplex_gnn import MAGNET

    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)

    model = MAGNET(cfg)
    y_hat, extras = model(batch)
"""

from __future__ import annotations

import yaml
import torch
import torch.nn as nn
from torch import Tensor

from src.models._universe import universe_from_cfg
from src.models.encoders import (SharedLSTM, GATEncoder, TypeProjection,
                                 FeatureNorm)
from src.models.fusion import CrossLayerFusion
from src.models.prediction_head import PredictionHead, CombinedLoss


# _FeatureNorm 已移到 encoders.py（SharedLSTM 的 input_norm 也要用同一個），
# 這裡保留原名以免既有 import 與測試失效。
_FeatureNorm = FeatureNorm


class MAGNET(nn.Module):
    """
    MAGNET — Multiplex ADR-Guided Network for Equity Trading

    Corresponds to IMPLEMENTATION_SPEC §0 (one-page overview) & §8.1

    Args:
        cfg (dict): 完整的 base.yaml 解析結果（含 model / loss_weights / align_loss）

    Shapes（以 batch 維度為例）：
        x_seq_L1   : [B, T, n1, F]  ADR T 步歷史特徵序列
        x_seq_L2   : [B, T, n2, F]  TW  T 步歷史特徵序列
        edge_index_L1 : [2, E1]
        edge_attr_L1  : [E1, 1]
        edge_index_L2 : [2, E2]
        edge_attr_L2  : [E2, 1]
        y             : [B, n2]     TW(t+1) log_return（訓練時提供）

    Note:
        - 推論時 batch 可以是單張快照（squeeze B 維度），forward 同樣有效。

    ─────────────────────────────────────────────────────────────────────
    E6 變更說明（2026-08-09）

    原實作把 A12 當成對角線：h_L1 與 h_L2 都是 [B, n, d']，直接送進
    CrossLayerFusion 靠共用索引對齊，等於硬編碼 p(j) = j。這隱含兩件事：
    n1 == n2，且每個 TW 節點都有配對。擴充後（US 30 / TW 50、配對率 14%）
    兩者皆不成立。

    本次改為顯式 gather：融合前先把 L1 對齊到 L2 的索引空間

        ĥ1_j = h1_{p(j)} + Σ_i B_eff[i,j]·h1_i        [B, n2, d']

    p(j) < 0（無配對）時第一項為零向量——論文式子裡「無配對節點無恆等項」
    的一般化形式，這裡才真正實例化。B_eff 亦由 [n, n] 改為 [n1, n2]，
    候選遮罩由 pair_index 排除恆等邊，不再用 torch.eye。

    k7 下 p(j) = j 且每個節點都有配對，gather 退化為恆等、遮罩退化為
    ~eye——舊行為是新結構的特例，故 k=7 的輸出不變（已用兩條位元確定的
    凍結 run 驗證 SHA-256 相同）。
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        m_cfg   = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        gat_cfg  = m_cfg["gat"]
        proj_cfg = m_cfg["projection"]
        fuse_cfg = m_cfg["fusion"]
        head_cfg = m_cfg["prediction_head"]

        # M6 Stage 0 ablation: 切斷 ADR → TW 跨層訊號（保留 fusion 結構）
        # 啟用時將 h_L1 在進 fusion 前歸零，h_fused = (1 - gate) * h_L2
        self.disable_a12 = bool(m_cfg.get("disable_a12", False))

        # M8 Hierarchical A12: 弱連結第二層邊（strong = 身分配對，維持不動）
        #   mode = None       : 現行 MAGNET（無弱連結）
        #   mode = "free"     : ADR_i → TW_j (i≠j) 全部 42 條候選自由學習
        #   mode = "industry" : 候選僅限同產業跨公司（PAIR_MAP industry），12 條
        # 機制：beta 為靜態可學邊權（init 0 → 起點即現行 MAGNET），
        #   h_L1_aug_j = h_L1_j + Σ_i beta_eff[i,j]·h_L1_i，弱訊號先併入
        #   ADR 側再過既有 gate——gate 維持唯一的跨市場閘門，
        #   強弱不共用 softmax（HGT 教訓）、lag 固定為 1 不學（DeltaLag 教訓）、
        #   L1 稀疏懲罰讓資料不支持的邊留在 0（MEIG 教訓）。
        # 節點結構：p(j) 與兩層節點數皆由 universe 決定（E6）
        u = universe_from_cfg(cfg)
        self.universe_name = u.name
        self.n_l1, self.n_l2 = u.n_l1, u.n_l2
        # pair_src[j] = TW 節點 j 的配對 ADR 索引（無配對者先填 0，靠 has_pair 遮掉）
        # persistent=False：這兩個 buffer 完全由 cfg 的 universe 決定，不是學到的
        # 狀態，不該進 state_dict。若進了，E6 之前存的 checkpoint 會因為缺這兩個
        # key 而載不進來（m8_epoch_trajectory.py 讀 M9 的逐 epoch 權重時炸過）。
        # 與 baseline_hats / baseline_meig 的既有慣例一致。
        self.register_buffer(
            "pair_src",
            torch.tensor([max(i, 0) for i in u.pair_index], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "has_pair",
            torch.tensor([i >= 0 for i in u.pair_index], dtype=torch.bool),
            persistent=False,
        )

        # 階段 A-7：耦合前的橫截面去均值（cross-sectional demeaning）
        #
        # 量到的問題：訓練後 L1 的 30 個美股表示彼此餘弦相似度 +0.9997、
        # L2 的 50 個台股表示 +0.9995——兩層的節點表示都指向同一個方向，
        # 橫截面上分不出彼此，而排序任務要的正是橫截面差異。逐維扣掉當日
        # 的節點平均後，L1 降到 +0.1616、L2 降到 +0.0621（T1F9 s42，10 天平均）。
        #
        # 為什麼扣掉的是共同成分而不是訊號：ridge 階梯的 M30r（30 檔美股
        # 減去等權平均）test IC +0.0912，對照完整 R2 的 +0.1020——去均值後
        # 的殘差保留了 89% 的訊號。共同成分主要是市場方向，對橫截面排序
        # 本來就沒有貢獻（所有節點同加一個常數，名次不變）。
        #
        # 只去均值，不要再除以橫截面標準差：實測除完 L1 反而回到 +0.9275，
        # 因為 std 小的維度會把主導殘差方向放大。
        #
        # 位置：proj 之後、跨層耦合之前。與 projection.layer_norm 正交
        # ——後者在特徵維度上正規化單一節點，這裡在節點維度上對齊全體。
        #
        # 預設 False：開啟會改變所有既有 run 的數值，k7 凍結基準與
        # e7_acceptance 的位元確定路徑必須維持不變。
        cs_cfg = m_cfg.get("cs_demean", {}) or {}
        self.cs_demean_l1 = bool(cs_cfg.get("l1", False))
        self.cs_demean_l2 = bool(cs_cfg.get("l2", False))

        # 階段 A-1+3：稠密可學耦合矩陣 A（per-node 載荷）
        #
        # 現行的跨層耦合是「固定權重 1 的恆等邊 + init 0 的候選邊」，等價於
        # 一個結構被寫死的 A：只有 7 個位置是 1，其餘由 weak_beta 從 0 出發。
        # 三個已量測到的後果：
        #   1. 43/50 節點的跨市場輸入位元為零（結構可達 + 擾動兩法一致）
        #   2. lambda=1e-3 讓候選邊只送出台股自身訊號的 0.08%（假 null 的來源）
        #   3. 模型沒有任何 per-node 參數 —— 台積電與台塑走同一組權重，
        #      而跨市場曝險本來就是個股專屬的
        #
        # 本模式把兩條通道合併成單一可學矩陣：
        #     h_in[:, j] = sum_i A[i, j] * h_L1[:, i]
        # A[:, j] 就是台股 j 的載荷向量，30 x 50 = 1,500 個節點專屬參數，
        # 與 [24] 的 per-target 迴歸係數數量相同。
        #
        # 初始化 A[p(j), j] = init_paired（預設 1，等於保留恆等邊的起點），
        # 其餘 = init_other（預設 1/n1）。取 1/n1 的理由：sum_i (1/n1) h_L1[i]
        # 就是等權平均，而 ridge 階梯的 M1（美股等權平均 1 維）單獨 test IC
        # +0.0736——50 個節點在第 0 步就拿到目前只有 7 個節點拿得到的東西。
        # 對照 init 0：lambda=0 訓練後 beta 只長到 2.965e-03，是 1/30 的 1/11。
        #
        # 不加 L1 稀疏懲罰：lambda=1e-3 已證實會把邊凍住；[24] 的 t 檢定篩邊
        # 在 30x50 尺度下保留 92% 的配對（tau=2 vs tau=0 差 +0.0013, p=0.59），
        # 這個規模不需要選邊。
        #
        # 上限（先行測過）：以現行編碼器的 h_L1 做 per-target 線性映射，
        # test IC +0.0435，對照現況 +0.0269。可回收約 +0.017，但到不了
        # 原始報酬的 +0.1020——那段損失在編碼器裡（特徵稀釋 -0.031、
        # LSTM -0.023、GAT_L1 -0.005），不是耦合層能修的。
        #
        # 預設 None：既有 run 與 k7 凍結基準的數值必須維持不變。
        # 階段 A-2a：原始特徵到耦合點的跳接（raw skip）
        #
        # 量到的問題：以 per-target 線性映射能從各位置抽出的 test IC
        #     raw    原始特徵（F3 設定，3 x 30 = 90 維）   +0.0874
        #     h_L1   現行耦合點（960 維）                  +0.0448   <- 編碼器丟掉 49%
        #     concat [h_L1 ; raw]                        +0.0666
        # 模型實際做到 +0.0269，是自身耦合點上限的 60%。也就是說耦合層還能
        # 榨出 +0.018，而把 raw 接進耦合點可以再多 +0.022——後者是目前所有
        # 未動的改動裡最大的一塊。
        #
        # 這一項不是為了「拆掉表示塌縮」：那個假說已被三個實驗推翻
        # （cs_demean 拿掉共同成分 IC 掉到 -0.0021；dense A 在無約束下自己把
        # 稠密邊縮到 13%；拿掉 RSI_14+BB_pos 後 IC 掉 45%）。塌縮是訊號本身。
        # 這裡修的是另一件事：編碼器在抵達耦合點前丟掉了一半可線性抽取的訊號。
        #
        # 形式：h_L1 <- proj_L1(...) + skip_L1(x_raw[:, -1])，形狀不變，
        # fusion 與 head 都不用改，三階段順序也不變（故仍屬機制層改動）。
        # 跳接吃的特徵與 LSTM 完全相同（共用 feature_subset），避免把
        # 「多給了資訊」誤算成「跳接有效」。
        #
        # 預設關閉：開啟會改變所有既有 run 與 k7 凍結基準的數值。
        skip_cfg = m_cfg.get("raw_skip", {}) or {}
        self.raw_skip_l1 = bool(skip_cfg.get("l1", False))
        self.raw_skip_l2 = bool(skip_cfg.get("l2", False))
        self.raw_skip_zero_init = bool(skip_cfg.get("zero_init", False))
        # mode = "add"（A-2a，已測，無效）| "concat"（A-2a'）
        #
        # 為什麼要有 concat：2a 的相加把原始特徵壓進一個已被編碼器佔滿的 d' 維
        # 空間，兩者重疊不可分。實測耦合點可抽取的資訊完全沒動（+0.0448 ->
        # +0.0446），而拼接的同一組表示是 +0.0662。差別在算子，不在資訊。
        #
        # concat 模式下 proj 的輸出維度讓出 concat_dim 維給原始特徵，
        # 兩者拼起來仍是 d'——所以 fusion / head 的介面完全不用改，
        # 也不要求 L1 與 L2 同時開啟。
        self.raw_skip_mode = skip_cfg.get("mode", "add")
        if self.raw_skip_mode not in ("add", "concat"):
            raise ValueError(
                f"model.raw_skip.mode 只支援 'add' 或 'concat'，"
                f"當前為 {self.raw_skip_mode!r}"
            )
        self.raw_concat_dim = skip_cfg.get("concat_dim")   # None -> 取 lstm.in_size
        # norm：跳接前的正規化。
        #   "batchnorm"（預設）逐特徵跨節點對齊尺度，節點差異保留
        #   "none"          不正規化，資訊零損失但線性層條件數差
        #   "layernorm"     逐節點跨特徵——**已證實有害**，僅供重現舊 run
        #
        # 變更揭露：2026-08-16 之前的 raw_skip run（tag tw50_T1F3skip* 與
        # tw50_T1F3cat*）用的是 layernorm，其 config_snapshot 沒有 norm 欄位。
        # 那批的結論（跳接無效）不成立，因為原始特徵在進耦合點前已被壓掉 64%。
        self.raw_skip_norm = skip_cfg.get("norm", "batchnorm")
        if self.raw_skip_norm not in ("batchnorm", "none", "layernorm"):
            raise ValueError(
                f"model.raw_skip.norm 需為 batchnorm / none / layernorm，"
                f"當前為 {self.raw_skip_norm!r}"
            )

        # ── 層內圖的消融（A₁ / A₂）─────────────────────────────
        # 存在理由：跨層邊（恆等 + 候選）已在 §26 消融過，
        # 但層內由 |rho| > tau 建的相關係數圖從未單獨檢驗。
        #
        #   empty_*  只留 self-loop：保留 GAT 的參數，只拿掉訊息傳遞。
        #            比「直接繞過 GAT」乾淨，後者會同時少掉一批參數。
        #   rand_*   保留邊數與 edge_attr 的分布，只打亂端點：
        #            檢驗「相關係數挑出來的結構」是否有資訊，
        #            而不只是「有沒有圖」。
        #
        # 預設 none：既有 run 數值不變。
        gat_cfg_ab = m_cfg.get("gat", {}) or {}
        self.graph_ablate = gat_cfg_ab.get("graph_ablate", "none")
        _ok = {"none", "empty_l1", "empty_l2", "empty_both",
               "rand_l1", "rand_l2", "rand_both"}
        if self.graph_ablate not in _ok:
            raise ValueError(
                f"model.gat.graph_ablate 需為 {sorted(_ok)} 之一，"
                f"當前為 {self.graph_ablate!r}")
        self.graph_ablate_seed = int(gat_cfg_ab.get("graph_ablate_seed", 42))

        coup_cfg = m_cfg.get("coupling", {}) or {}
        self.coupling_mode = coup_cfg.get("mode")          # None | "dense"
        if self.coupling_mode not in (None, "dense"):
            raise ValueError(
                f"model.coupling.mode 只支援 None 或 'dense'，"
                f"當前為 {self.coupling_mode!r}"
            )

        weak_cfg = m_cfg.get("weak_links", {}) or {}
        self.weak_mode   = weak_cfg.get("mode")            # None | "free" | "industry"
        self.weak_lambda = float(weak_cfg.get("lambda_sparse", 1e-3))
        if self.weak_mode is not None:
            self.register_buffer(
                "weak_mask", self._build_weak_mask(self.weak_mode, u)
            )
            # beta_rank：候選邊矩陣 B 的秩約束。
            #   0（預設）= 全秩，B 直接是 [n1, n2]，n1xn2 = 1,500 個自由參數
            #   r > 0    = B = U Vᵀ，U [n1, r] / V [n2, r]，(n1+n2)xr 個參數
            #
            # 存在理由（§37）：穩定性的損失可以完全歸因到 ③ 這條路徑——
            # 不含 ③ 的設定前後半落差 <= 0.0012，含 ③ 的 >= 0.0231，無例外。
            # 而在資料上直接估 C（③ 對應的殘差結構）時，**同一段期間內部
            # 拆兩半的相關只有 0.13–0.43**，對照 beta（50 參數）的 0.53–0.65。
            # 問題不是市場狀態改變，是 1,500 個兩兩載重在 246 天 x 50 檔的
            # 樣本下估不出來。降秩直接對應這個診斷；訊號本身只有 1–3 維（§21）。
            #
            # 初始化沿用 LoRA 的作法：U 隨機、V 全零 -> B 起點仍為 0
            # （與全秩版相同，訓練起點即「只有恆等邊」），但 V 拿得到梯度。
            self.beta_rank = int(weak_cfg.get("beta_rank", 0))
            if self.beta_rank < 0:
                raise ValueError("beta_rank 不可為負")
            if self.beta_rank > 0:
                r = self.beta_rank
                self.weak_U = nn.Parameter(torch.randn(u.n_l1, r) / (u.n_l1 ** 0.5))
                self.weak_V = nn.Parameter(torch.zeros(u.n_l2, r))
            else:
                self.weak_beta = nn.Parameter(torch.zeros(u.n_l1, u.n_l2))

        # ── beta 層（階段 P1）──────────────────────────────────────
        # 存在理由（回應「43 檔無配對台股拿不到跨市場訊號」）：
        #
        # 量到的事實：
        #   (1) 跨市場訊號 72% 是 rank-1——每天一個純量（美股市場報酬）
        #       乘上每檔台股一個 beta，就拿到最好線性 baseline 的 72%
        #       （+0.0738 / +0.1020）；rank-3 拿到 87%。
        #   (2) h_L1 有 99.86% 的能量在共同成分 c 上，特異成分只有 0.14%。
        #   (3) 恆等邊給 7 檔滿量級 5.61、候選邊給 43 檔 0.0042，差 1,345 倍。
        #   (4) 恆等邊 60% 的價值來自「有沒有配對」這個二元分組 x 市場因子，
        #       只有 40% 是 ADR 特有的——把恆等邊換成美股均值仍保留 60%。
        #
        # 統一設計原則（§24.2）：任何耦合路徑，若其輸出對 50 檔台股是同一個
        # 向量，對橫截面 IC 的貢獻恰好為零，不論量級多大。實測：全部 50 檔
        # 都給 h̄₁ -> IC +0.0036（≈ 0）。所以價值全在**跨股票的離散度**。
        #
        # 形式：
        #   ĥ₁ⱼ = α_j·h₁_p(j)·[paired] + γ_j·h̄₁ + Σᵢ B[i,j]·(h₁ᵢ − h̄₁)
        #         └ ADR 路徑，可學權重  └ 因子暴露   └ 殘差聚合（去掉共同成分）
        #
        # 三項各司其職，不重複計算共同成分：γ_j 承載 beta 暴露（現行設計只有
        # 「配對 = 1 / 無配對 = 0」兩級，這裡升級成 50 級連續），
        # B 改吃殘差所以不再與 γ 競爭同一個方向。
        #
        # 退化保證：alpha_mean=1、gamma_mean=0、init_std=0、B=0 時
        # 逐位元等於現行 MAGNET，故既有 run 不受影響（tests 有驗）。
        #
        # 初始化必須分散（P0）：零初始化在飽和加法通道裡長不起來
        # （實測 B 從 0 出發，訓練後 |B|max 仍只有 0.0005）。
        self.beta_layer = bool(weak_cfg.get("beta_layer", False))
        if self.beta_layer:
            if self.weak_mode is None:
                raise ValueError(
                    "model.weak_links.beta_layer 需要 weak_links.mode 不為 None"
                )
            std = float(weak_cfg.get("beta_init_std", 0.3))
            a_mu = float(weak_cfg.get("beta_alpha_mean", 1.0))
            g_mu = float(weak_cfg.get("beta_gamma_mean", 0.0))
            b_std = float(weak_cfg.get("beta_b_init_std", 0.0))
            # 三項的開關（消融用）。預設全開＝完整 beta 層。
            #   use_identity ①  α_j·h₁_p(j)·[paired]   7 檔的 ADR 特異訊號
            #   use_factor   ②  γ_j·h̄₁                 50 檔的市場因子暴露
            #   use_residual ③  Σᵢ B[i,j]·(h₁ᵢ − h̄₁)   扣掉共同成分後的個股結構
            self.beta_use_identity = bool(weak_cfg.get("beta_use_identity", True))
            self.beta_use_factor = bool(weak_cfg.get("beta_use_factor", True))
            self.beta_use_residual = bool(weak_cfg.get("beta_use_residual", True))
            if not any((self.beta_use_identity, self.beta_use_factor,
                        self.beta_use_residual)):
                raise ValueError("beta 層的三項不可同時關閉")
            if std < 0 or b_std < 0:
                raise ValueError("beta_init_std / beta_b_init_std 不可為負")
            # 因子個數 k。k=1 時完全走原路徑（不建立 beta_factor_w），
            # 既有 run 位元不變。k>1 時 ② 改為 Σ_f γ_{j,f}·m_f，
            # 其中 m_f = Σᵢ w_{f,i}·h₁ᵢ 是學出來的因子方向。
            #
            # 動機：訊號本身的低秩分解（§21）—— rank-1 等權市場因子拿到 72%、
            # rank-3 拿到 87%。現行 ② 用的是「等權平均」這一個固定方向；
            # 多因子讓方向本身也可學，且第二、三個因子有機會抓到那 15%。
            # 這一格的 headroom 量在資料上，不是量在探針上（§34 的教訓）。
            self.beta_n_factors = int(weak_cfg.get("beta_n_factors", 1))
            if self.beta_n_factors < 1:
                raise ValueError("beta_n_factors 必須 >= 1")
            self.beta_alpha = nn.Parameter(
                torch.randn(u.n_l2) * std + a_mu)
            if self.beta_n_factors == 1:
                self.beta_gamma = nn.Parameter(
                    torch.randn(u.n_l2) * std + g_mu)
            else:
                k = self.beta_n_factors
                self.beta_gamma = nn.Parameter(
                    torch.randn(u.n_l2, k) * std + g_mu)
                # 第 1 個因子固定初始化為等權平均（= 現行 h̄₁），
                # 其餘分散初始化。同樣尺度（1/n₁）避免因子間量級失衡。
                w = torch.randn(k, u.n_l1) * (std / u.n_l1)
                w[0] = 1.0 / u.n_l1
                self.beta_factor_w = nn.Parameter(w)
            if b_std > 0:
                with torch.no_grad():
                    if self.beta_rank > 0:
                        # 低秩時對 V 施加離散度（U 已隨機），效果等價
                        self.weak_V.normal_(0.0, b_std)
                    else:
                        self.weak_beta.normal_(0.0, b_std)

        if self.coupling_mode == "dense":
            if self.weak_mode is not None:
                # dense 已經涵蓋全部 n1 x n2 條邊，再疊 weak_beta 等於同一組邊
                # 有兩份權重，梯度會在兩者之間任意分配，A 的數值不再可解讀。
                raise ValueError(
                    "model.coupling.mode='dense' 與 weak_links.mode 互斥："
                    "dense 的 A 已涵蓋所有跨層邊（含恆等邊），"
                    f"不可再啟用 weak_links（當前 mode={self.weak_mode!r}）。"
                )
            init_paired = float(coup_cfg.get("init_paired", 1.0))
            init_other = coup_cfg.get("init_other")
            init_other = 1.0 / u.n_l1 if init_other is None else float(init_other)
            A0 = torch.full((u.n_l1, u.n_l2), init_other, dtype=torch.float32)
            for j, i in enumerate(u.pair_index):
                if i >= 0:
                    A0[i, j] = init_paired
            self.coupling_A = nn.Parameter(A0)
            self.coupling_init = (init_paired, init_other)

        d_prime   = proj_cfg["d_prime"]
        H_lstm    = lstm_cfg["hidden_dim"]
        H_gat     = gat_cfg["hidden_dim"]   # 最後一層 concat=False → H_gat

        # ── Phase 1 ───────────────────────────────────────────────────
        # SharedLSTM：L1 / L2 共用同一份權重
        self.lstm = SharedLSTM(lstm_cfg)

        # GAT：L1 / L2 各自獨立（multiplex ≠ siamese）
        self.gat_L1 = GATEncoder(gat_cfg)
        self.gat_L2 = GATEncoder(gat_cfg)

        # 階段 A-2a'：concat 模式下，proj 讓出 concat_dim 維給原始特徵路徑。
        # 預設讓出 lstm.in_size 維——原始特徵有幾維就給幾維，剛好能無損通過，
        # 再多是浪費（線性層只是換基底），再少會壓縮。
        d_raw = (self.lstm.in_size if self.raw_concat_dim is None
                 else int(self.raw_concat_dim))
        self.raw_concat_dim = d_raw
        cat_l1 = self.raw_skip_l1 and self.raw_skip_mode == "concat"
        cat_l2 = self.raw_skip_l2 and self.raw_skip_mode == "concat"
        for flag, name in ((cat_l1, "L1"), (cat_l2, "L2")):
            if flag and not 0 < d_raw < d_prime:
                raise ValueError(
                    f"raw_skip.concat_dim 需落在 (0, d_prime={d_prime})，"
                    f"當前為 {d_raw}（{name}）"
                )
        # Projection：L1 / L2 各自獨立
        self.proj_L1 = TypeProjection(
            {**proj_cfg, "d_prime": d_prime - d_raw} if cat_l1 else proj_cfg,
            in_dim=H_gat)
        self.proj_L2 = TypeProjection(
            {**proj_cfg, "d_prime": d_prime - d_raw} if cat_l2 else proj_cfg,
            in_dim=H_gat)

        # ── Phase 2 ───────────────────────────────────────────────────
        self.fusion = CrossLayerFusion(fuse_cfg, d_prime=d_prime)

        # ── Phase 3 ───────────────────────────────────────────────────
        self.head = PredictionHead(head_cfg, d_prime=d_prime,
                                   n_nodes=self.n_l2)

        # 損失函數（訓練時使用）
        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

        # LSTM 輸入維度斷言
        assert lstm_cfg["input_dim"] == 9, (
            f"SharedLSTM input_dim 應與 TECH_FEATURE_COLS 維度一致（9），"
            f"當前為 {lstm_cfg['input_dim']}"
        )

        # 階段 A-2a：跳接模組。**刻意放在 __init__ 最後**——建立 nn.Linear 會
        # 消耗 RNG，若放在中間，開啟跳接會連帶位移 fusion 與 head 的初始化，
        # 那樣同一顆種子下「開/關」兩個模型就不只差一條跳接，效果無法歸因。
        # （這一點是 test_raw_skip_zero_init_matches_baseline 抓到的。）
        #
        # in_dim 取 SharedLSTM 實際吃到的維度（feature_subset 生效後的 in_size），
        # 確保兩條路徑的資訊集相同。
        # 正規化在前：原始特徵的尺度差到 5000 倍（RSI_14 約 50、
        # log_return 約 0.01），不處理的話線性層的條件數會被 RSI 主導。
        # 必須是逐特徵跨節點（_FeatureNorm），不可用 LayerNorm——後者
        # 正規化特徵軸，會把橫截面差異刪掉，見 _FeatureNorm 的說明。
        # 不用 bias：常數項對每個節點同加同減，對橫截面排序無作用。
        if self.raw_skip_l1 or self.raw_skip_l2:
            out_dim = (self.raw_concat_dim if self.raw_skip_mode == "concat"
                       else d_prime)

            def _mk_skip() -> nn.Module:
                lin = nn.Linear(self.lstm.in_size, out_dim, bias=False)
                if self.raw_skip_zero_init:
                    if self.raw_skip_mode == "concat":
                        # concat 下 proj 的輸出維度已經變了，權重歸零也回不到
                        # 原模型；留著這個選項只會給出一個假的「錨點」。
                        raise ValueError(
                            "raw_skip.zero_init 只在 mode='add' 下有意義："
                            "concat 會改變 proj 的輸出維度，歸零也無法還原原模型。"
                        )
                    nn.init.zeros_(lin.weight)     # 開啟時起點即現行模型
                norm: nn.Module = {
                    "batchnorm": lambda: _FeatureNorm(self.lstm.in_size),
                    "none":      nn.Identity,
                    "layernorm": lambda: nn.LayerNorm(self.lstm.in_size),
                }[self.raw_skip_norm]()
                return nn.Sequential(norm, lin)
            if self.raw_skip_l1:
                self.skip_L1 = _mk_skip()
            if self.raw_skip_l2:
                self.skip_L2 = _mk_skip()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        """
        Args:
            batch (dict):
                "x_seq_L1"      : [B, T, n1, F]  ADR 歷史序列
                "x_seq_L2"      : [B, T, n2, F]  TW  歷史序列
                "edge_index_L1" : list[Tensor[2, E_b]] 長度 B（每張快照邊數不同）
                                  或單一 Tensor [2, E1]（會自動廣播給所有 batch）
                "edge_attr_L1"  : list[Tensor[E_b, 1]] 同上對應
                "edge_index_L2" : list[Tensor[2, E_b]]
                "edge_attr_L2"  : list[Tensor[E_b, 1]]
                "y"             : [B, n2]  (選填，推論時可不傳)

        Returns:
            y_hat   : [B, n2]  預測 log_return
            extras  : dict{
                          "h_L1"   : [B, n1, d']  ADR 投影後表示（未對齊）
                          "h_L2"   : [B, n2, d']  TW  投影後表示
                          "h_fused": [B, n2, d']  融合後表示
                          "alpha"  : [B, n2, 1]   per-node attention
                          "gate"   : [B, n2, d']  per-node per-dim gate
                      }
        """
        x_L1 = batch["x_seq_L1"]          # [B, T, n1, F]
        x_L2 = batch["x_seq_L2"]          # [B, T, n2, F]
        ei_L1 = batch["edge_index_L1"]    # list[Tensor] 或 Tensor
        ea_L1 = batch["edge_attr_L1"]
        ei_L2 = batch["edge_index_L2"]
        ea_L2 = batch["edge_attr_L2"]

        B = x_L1.size(0)
        self._assert_node_counts(x_L1.size(2), x_L2.size(2))

        # ── Phase 1: LSTM 時序編碼 ────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §3.1
        h_lstm_L1 = self.lstm(x_L1)   # [B, n, H_lstm]
        h_lstm_L2 = self.lstm(x_L2, layer=1)   # [B, n, H_lstm]（共用同一份 LSTM；
        #                                    layer=1 只在 input_norm_scope
        #                                    ="per_layer" 時才切到第二組正規化）

        # ── Phase 1: GAT 圖編碼 ───────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §3.2
        # 每張快照獨立跑 GAT（edge_index 已是單張快照的局部索引）
        if self.graph_ablate != "none":
            ei_L1, ea_L1 = self._ablate_graph(ei_L1, ea_L1, x_L1.size(2), "l1")
            ei_L2, ea_L2 = self._ablate_graph(ei_L2, ea_L2, x_L2.size(2), "l2")
        h_gat_L1 = self._apply_gat_batched(self.gat_L1, h_lstm_L1, ei_L1, ea_L1)  # [B, n1, H_gat]
        h_gat_L2 = self._apply_gat_batched(self.gat_L2, h_lstm_L2, ei_L2, ea_L2)  # [B, n2, H_gat]

        # ── Phase 1: 投影至 d' 維 ─────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §3.3
        h_L1 = self.proj_L1(h_gat_L1)  # [B, n1, d']
        h_L2 = self.proj_L2(h_gat_L2)  # [B, n2, d']

        # 階段 A-2a：原始特徵跳接（見 __init__ 的說明）。
        # 取最後一步 x[:, -1]，並用 SharedLSTM 的 feat_idx 取同一組欄位。
        if self.raw_skip_l1:
            r1 = self.skip_L1(self._raw_last(x_L1))
            h_L1 = (torch.cat([h_L1, r1], dim=-1)
                    if self.raw_skip_mode == "concat" else h_L1 + r1)
        if self.raw_skip_l2:
            r2 = self.skip_L2(self._raw_last(x_L2))
            h_L2 = (torch.cat([h_L2, r2], dim=-1)
                    if self.raw_skip_mode == "concat" else h_L2 + r2)

        # 階段 A-7：橫截面去均值（見 __init__ 的說明）。
        # 兩個旗標預設皆為 False，關閉時完全不進入這段，既有 run 位元不變。
        if self.cs_demean_l1:
            h_L1 = self._cs_demean(h_L1)
        if self.cs_demean_l2:
            h_L2 = self._cs_demean(h_L2)

        # ── Phase 2: 跨市場融合 ───────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §4
        # 先把 L1 對齊到 L2 的索引空間（[B, n1, d'] → [B, n2, d']），
        # gate 之後才是逐節點對齊的運算。
        # M6 Stage 0 ablation: disable_a12=True 時將 ADR 訊號零化（保留 fusion 結構）
        # M8: weak links 先富化 ADR 側，再過 gate（gate 仍是唯一跨市場閘門）
        if self.disable_a12:
            h_L1_in = h_L1.new_zeros(B, self.n_l2, h_L1.size(-1))
        else:
            h_L1_in = self._augment_weak(h_L1)             # [B, n2, d']
        h_fused, alpha, gate = self.fusion(h_L1_in, h_L2)  # [B, n2, d']

        # ── Phase 3: 預測 ─────────────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §5.1
        y_hat = self.head(h_fused)  # [B, n2]

        extras = {
            "h_L1":    h_L1,
            "h_L2":    h_L2,
            "h_fused": h_fused,
            "alpha":   alpha,
            "gate":    gate,
        }
        if self.weak_mode is not None:
            extras["weak_beta"] = self._weak_beta_full() * self.weak_mask  # [n1, n2] 分析用
        if self.coupling_mode == "dense":
            extras["coupling_A"] = self.coupling_A                 # [n1, n2] 分析用
        return y_hat, extras

    def _raw_last(self, x_seq: Tensor) -> Tensor:
        """
        取序列最後一步的原始特徵，並套用與 SharedLSTM 相同的 feature_subset。

            x_seq : [B, T, n, F]  ->  [B, n, in_size]

        共用 feat_idx 是刻意的：跳接若看得到 LSTM 看不到的欄位，
        實驗就分不清是「跳接有效」還是「多給了資訊」。
        """
        x = x_seq[:, -1]                                   # [B, n, F]
        if self.lstm.feat_idx is not None:
            x = x.index_select(-1, self.lstm.feat_idx)
        return x

    @staticmethod
    def _cs_demean(h: Tensor) -> Tensor:
        """
        逐維橫截面去均值：對節點維度扣掉當日全體節點的平均。

            h'[b, i, k] = h[b, i, k] - mean_i h[b, i, k]

        節點維度是倒數第二維（[B, n, d'] 或 [n, d'] 皆適用），故用 dim=-2。
        """
        return h - h.mean(dim=-2, keepdim=True)

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------
    def _assert_node_counts(self, n1: int, n2: int) -> None:
        """
        batch 的節點數必須與 cfg 宣告的 universe 相符。

        不檢查的話，pair_src 會拿長度 n2_cfg 的索引去 gather 長度 n1_batch
        的張量——越界才會炸，訊息完全看不出真正原因；更糟的是若兩個
        universe 的節點數碰巧相同，連炸都不會炸。
        """
        if (n1, n2) != (self.n_l1, self.n_l2):
            raise ValueError(
                f"batch 節點數 L1={n1} / L2={n2} 與 cfg 的 "
                f"universe='{self.universe_name}'（L1={self.n_l1} / L2={self.n_l2}）不符。"
                f"請確認 base.yaml 的 data.universe 與 data.snapshot_dir 一致。"
            )

    @staticmethod
    def _build_weak_mask(mode: str, universe) -> Tensor:
        """
        建立弱連結候選遮罩 [n1, n2] float。mask[i, j] = 1 表示允許
        ADR_i → TW_j 的弱連結候選；恆等邊（i == p(j)）一律排除，
        因為那條邊屬於強連結層。

        mode = "free"     : 全部非恆等邊
        mode = "industry" : 僅同產業的非恆等邊

        E6：原本用 torch.eye 排除對角線。恆等邊是對角線只在 k7 成立，
        擴充後它是 pair_index 決定的稀疏映射。
        """
        if mode not in ("free", "industry"):
            raise ValueError(f"未知 weak_links.mode={mode!r}；可選：free | industry")
        n1, n2 = universe.n_l1, universe.n_l2

        ident = torch.zeros(n1, n2, dtype=torch.bool)
        for j, i in enumerate(universe.pair_index):
            if i >= 0:
                ident[i, j] = True

        if mode == "free":
            mask = ~ident
        else:
            us_ind = [universe.industry.get(t) for t in universe.us_nodes]
            tw_ind = [universe.industry.get(c) for c in universe.tw_nodes]
            same = torch.tensor(
                [[us_ind[i] is not None and us_ind[i] == tw_ind[j]
                  for j in range(n2)] for i in range(n1)],
                dtype=torch.bool,
            )
            mask = same & ~ident
            if not mask.any():
                # 兩側產業別來自不同分類法時字串永遠不相等，遮罩全零。
                # beta init 也是 0，於是 magnet_weak_industry 會退化成
                # 一般 MAGNET——訓練照跑、結果一模一樣，看起來像是
                # 「同產業弱連結沒有用」，實際上是一條候選邊都沒建。
                # 這需要一份 US↔TW 產業對照表（屬於 universe 定義），
                # 不是模型層能決定的，因此在這裡擋住。
                raise ValueError(
                    f"[{universe.name}] weak_links.mode='industry' 產生 0 條候選邊："
                    f"US 側產業別 {sorted({i for i in us_ind if i})} 與 "
                    f"TW 側 {sorted(set(tw_ind))} 沒有交集。"
                    f"需要先在 universe 定義中補上兩側共用的產業分類。"
                )
        return mask.float()

    def _weak_beta_full(self) -> Tensor:
        """回傳未遮罩的 B（全秩時即參數本身；低秩時為 U Vᵀ）。"""
        if getattr(self, "beta_rank", 0) > 0:
            return self.weak_U @ self.weak_V.t()          # [n1, n2]
        return self.weak_beta

    def _augment_weak(self, h_L1: Tensor) -> Tensor:
        """
        把 L1 對齊到 L2 的索引空間，並併入弱連結：

            ĥ1_j = h1_{p(j)} + Σ_i beta_eff[i,j]·h1_i          （預設）
            ĥ1_j = α_j·h1_{p(j)} + γ_j·h̄1 + Σ_i B[i,j]·(h1_i − h̄1)  （beta 層）

        第一項是恆等邊（權重固定 1，存在與否已知）；p(j) < 0 的節點
        沒有這一項，補零向量。第二項是候選邊，beta init 0
        → 未學習前恆等於「只有恆等邊」的現行 MAGNET（殘差式結構學習）。

        Args:
            h_L1 : [B, n1, d']
        Returns:
            [B, n2, d']
        """
        if self.coupling_mode == "dense":
            # 單一可學矩陣，恆等邊已折進 A 的初始值，不再另外相加
            return torch.einsum("ij,bid->bjd", self.coupling_A, h_L1)
        ident = h_L1.index_select(dim=1, index=self.pair_src)      # [B, n2, d']
        ident = ident * self.has_pair.view(1, -1, 1).to(ident.dtype)
        if self.weak_mode is None:
            return ident
        beta_eff = self._weak_beta_full() * self.weak_mask         # [n1, n2]
        if not self.beta_layer:
            h_weak = torch.einsum("ij,bid->bjd", beta_eff, h_L1)   # [B, n2, d']
            return ident + h_weak
        # beta 層：三項各司其職，見 __init__ 的說明
        #   ĥ₁ⱼ = α_j·h₁_p(j)·[paired] + γ_j·h̄₁ + Σᵢ B[i,j]·(h₁ᵢ − h̄₁)
        hbar = h_L1.mean(dim=1, keepdim=True)                      # [B, 1, d']
        out = torch.zeros_like(ident)
        if self.beta_use_identity:                                 # ①
            out = out + ident * self.beta_alpha.view(1, -1, 1)
        if self.beta_use_factor:                                   # ②
            if self.beta_n_factors == 1:
                out = out + self.beta_gamma.view(1, -1, 1) * hbar
            else:
                # m : [B, k, d']   每天 k 個因子；γ : [n2, k]
                m = torch.einsum("fi,bid->bfd", self.beta_factor_w, h_L1)
                out = out + torch.einsum("jf,bfd->bjd", self.beta_gamma, m)
        if self.beta_use_residual:                                 # ③
            out = out + torch.einsum("ij,bid->bjd", beta_eff, h_L1 - hbar)
        return out

    def _ablate_graph(self, edge_index, edge_attr, n_nodes: int, side: str):
        """層內圖的消融。side = "l1" | "l2"。

        empty：回傳空的 edge_index（GATv2Conv 的 add_self_loops 會補上
               self-loop，所以節點仍有自己的訊息，只是沒有鄰居）。
        rand ：保留邊數與 edge_attr 的值，只把端點重抽（不含自環）。
               用固定 seed，且以「快照序號」推進，讓每張快照的隨機圖不同
               但整體可重現。
        """
        mode = self.graph_ablate
        if mode == "none" or not (mode.endswith("both") or mode.endswith(side)):
            return edge_index, edge_attr
        is_list = isinstance(edge_index, (list, tuple))
        eis = list(edge_index) if is_list else [edge_index]
        eas = list(edge_attr) if is_list else [edge_attr]
        out_i, out_a = [], []
        for k, (ei, ea) in enumerate(zip(eis, eas)):
            if mode.startswith("empty"):
                out_i.append(torch.zeros(2, 0, dtype=torch.long, device=ei.device))
                out_a.append(ea.new_zeros((0,) + tuple(ea.shape[1:])))
                continue
            m = ei.size(1)
            g = torch.Generator(device="cpu").manual_seed(
                self.graph_ablate_seed * 1_000_003 + k)
            src = torch.randint(0, n_nodes, (m,), generator=g)
            dst = torch.randint(0, n_nodes, (m,), generator=g)
            bad = src == dst                       # 去掉自環，GAT 自己會加
            dst[bad] = (dst[bad] + 1) % n_nodes
            out_i.append(torch.stack([src, dst]).to(ei.device))
            out_a.append(ea)                       # 邊權的值分布保持不變
        if is_list:
            return out_i, out_a
        return out_i[0], out_a[0]

    @staticmethod
    def _apply_gat_batched(
        gat: GATEncoder,
        h:   Tensor,                                # [B, n, H_lstm]
        edge_index: Tensor | list[Tensor],          # 每張快照的邊（list）或共用張量
        edge_attr:  Tensor | list[Tensor],
    ) -> Tensor:
        """
        對 batch 中每張快照逐一跑 GAT。每張快照的 edge_index/edge_attr
        可能不同（list 形式，來自 multiplex_collate）；
        也支援所有快照共用同一組邊（單張量形式，例如手動構造的測試用 batch）。

        後續若需更高 throughput，可改用 torch_geometric.data.Batch 一次性處理。

        Returns:
            out : [B, n, H_gat]
        """
        B = h.size(0)
        is_list_form = isinstance(edge_index, (list, tuple))

        outs = []
        for b in range(B):
            ei = edge_index[b] if is_list_form else edge_index
            ea = edge_attr[b]  if is_list_form else edge_attr
            outs.append(gat(h[b], ei, ea))  # [n, H_gat]
        return torch.stack(outs, dim=0)     # [B, n, H_gat]

    # ------------------------------------------------------------------
    # 便利方法：計算損失
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        y_hat:  Tensor,
        y:      Tensor,
        extras: dict,
    ) -> tuple[Tensor, dict]:
        """
        計算 CombinedLoss。

        Args:
            y_hat  : [B, n2]
            y      : [B, n2]
            extras : forward 回傳的 extras dict

        Returns:
            loss       : scalar
            components : dict{"mse", "rank", "align", "variance"[, "weak_l1"]}
        """
        loss, comps = self.criterion(
            y_hat=y_hat,
            y=y,
            h_L1=extras.get("h_L1"),
            h_L2=extras.get("h_L2"),
        )
        # M8: 弱連結 L1 稀疏懲罰（僅訓練圖中生效；資料不支持的邊收縮回 0）
        if self.weak_mode is not None:
            weak_l1 = (self._weak_beta_full() * self.weak_mask).abs().sum()
            loss = loss + self.weak_lambda * weak_l1
            comps = {**comps, "weak_l1": float(weak_l1.detach())}
        return loss, comps

    # ------------------------------------------------------------------
    # 工廠方法：從 YAML 路徑建立模型
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str = "configs/base.yaml") -> "MAGNET":
        """從 YAML 路徑建立 MAGNET 實例。"""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)
