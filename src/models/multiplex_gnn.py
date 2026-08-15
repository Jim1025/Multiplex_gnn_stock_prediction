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
from src.models.encoders import SharedLSTM, GATEncoder, TypeProjection
from src.models.fusion import CrossLayerFusion
from src.models.prediction_head import PredictionHead, CombinedLoss


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

        weak_cfg = m_cfg.get("weak_links", {}) or {}
        self.weak_mode   = weak_cfg.get("mode")            # None | "free" | "industry"
        self.weak_lambda = float(weak_cfg.get("lambda_sparse", 1e-3))
        if self.weak_mode is not None:
            self.register_buffer(
                "weak_mask", self._build_weak_mask(self.weak_mode, u)
            )
            self.weak_beta = nn.Parameter(torch.zeros(u.n_l1, u.n_l2))

        d_prime   = proj_cfg["d_prime"]
        H_lstm    = lstm_cfg["hidden_dim"]
        H_gat     = gat_cfg["hidden_dim"]   # 最後一層 concat=False → H_gat

        # ── Phase 1 ───────────────────────────────────────────────────
        # SharedLSTM：L1 / L2 共用同一份權重
        self.lstm = SharedLSTM(lstm_cfg)

        # GAT：L1 / L2 各自獨立（multiplex ≠ siamese）
        self.gat_L1 = GATEncoder(gat_cfg)
        self.gat_L2 = GATEncoder(gat_cfg)

        # Projection：L1 / L2 各自獨立
        self.proj_L1 = TypeProjection(proj_cfg, in_dim=H_gat)
        self.proj_L2 = TypeProjection(proj_cfg, in_dim=H_gat)

        # ── Phase 2 ───────────────────────────────────────────────────
        self.fusion = CrossLayerFusion(fuse_cfg, d_prime=d_prime)

        # ── Phase 3 ───────────────────────────────────────────────────
        self.head = PredictionHead(head_cfg, d_prime=d_prime)

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
        h_lstm_L2 = self.lstm(x_L2)   # [B, n, H_lstm]（共用同一份 LSTM）

        # ── Phase 1: GAT 圖編碼 ───────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §3.2
        # 每張快照獨立跑 GAT（edge_index 已是單張快照的局部索引）
        h_gat_L1 = self._apply_gat_batched(self.gat_L1, h_lstm_L1, ei_L1, ea_L1)  # [B, n1, H_gat]
        h_gat_L2 = self._apply_gat_batched(self.gat_L2, h_lstm_L2, ei_L2, ea_L2)  # [B, n2, H_gat]

        # ── Phase 1: 投影至 d' 維 ─────────────────────────────────────
        # Corresponds to IMPLEMENTATION_SPEC §3.3
        h_L1 = self.proj_L1(h_gat_L1)  # [B, n1, d']
        h_L2 = self.proj_L2(h_gat_L2)  # [B, n2, d']

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
            extras["weak_beta"] = self.weak_beta * self.weak_mask  # [n1, n2] 分析用
        return y_hat, extras

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

    def _augment_weak(self, h_L1: Tensor) -> Tensor:
        """
        把 L1 對齊到 L2 的索引空間，並併入弱連結：

            ĥ1_j = h1_{p(j)} + Σ_i beta_eff[i,j]·h1_i

        第一項是恆等邊（權重固定 1，存在與否已知）；p(j) < 0 的節點
        沒有這一項，補零向量。第二項是候選邊，beta init 0
        → 未學習前恆等於「只有恆等邊」的現行 MAGNET（殘差式結構學習）。

        Args:
            h_L1 : [B, n1, d']
        Returns:
            [B, n2, d']
        """
        ident = h_L1.index_select(dim=1, index=self.pair_src)      # [B, n2, d']
        ident = ident * self.has_pair.view(1, -1, 1).to(ident.dtype)
        if self.weak_mode is None:
            return ident
        beta_eff = self.weak_beta * self.weak_mask                 # [n1, n2]
        h_weak = torch.einsum("ij,bid->bjd", beta_eff, h_L1)       # [B, n2, d']
        return ident + h_weak

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
            weak_l1 = (self.weak_beta * self.weak_mask).abs().sum()
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
