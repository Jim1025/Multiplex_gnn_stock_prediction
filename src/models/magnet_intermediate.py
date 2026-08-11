"""
magnet_intermediate.py — MAGNET 的 intermediate fusion 變體

存在理由（回應口試 Q7，並修正一個已量測到的結構缺陷）：

    一般 MAGNET 是 late fusion——L2 的圖跑在融合之前：

        h_gat_L2 = GAT_L2(h_lstm_L2, E2)      台股圖在這裡跑完
        h_fused  = g·h1_{p(j)} + (1-g)·h_L2   ADR 到這裡才進來，逐節點
        y_hat    = head(h_fused)              之後沒有任何傳播

    台股圖消耗的是只含台股特徵的 h_lstm_L2，等 ADR 資訊抵達時圖已經用完。
    融合是 per-node，融合之後直接進 head。後果實測如下（tw50、擾動整個
    x_seq_L1 後 y_hat 的變動）：

        配對 7 檔    mean|Δ| = 2.07e-02
        無配對 43 檔  mean|Δ| = 0.00e+00      ← 位元零

    ADR 資訊到不了 43 檔無配對台股，而且是結構性的：不存在前向路徑，
    連梯度路徑都沒有。gate 本身是開的（≈0.54），問題不是模型學會忽略 ADR，
    而是閘門後面接的是零向量。這代表「恆等邊注入、層內圖傳播」這句話的
    第二步從未被實例化——那張 1,304 條邊的台股圖從來不曾攜帶跨市場資訊。

    這個缺陷在 k=7 不可能被診斷：7 檔全部有配對，7/7 看起來像全覆蓋。

本檔的改動只有一件事：把融合移到 L2 圖傳播之前。

        z_L1   = proj_L1(GAT_L1(...))          [B, n1, H]
        z_L2   = proj_L2(h_lstm_L2)            [B, n2, H]   注意：圖之前
        ẑ_L1   = z_L1_{p(j)}（無配對補零）      [B, n2, H]
        h_seed = g·ẑ_L1 + (1-g)·z_L2           [B, n2, H]
        h_out  = GAT_L2(h_seed, E2)            [B, n2, H_gat]  ← 圖現在載著 ADR
        y_hat  = head(proj_out(h_out))         [B, n2]

    無配對節點 j 的 h_seed 仍然沒有 ADR（它沒有恆等邊，這點不變），但它在
    E2 裡的鄰居當中有配對股，GAT_L2 聚合之後 h_out[j] 就含有 ADR 資訊。
    L2 密度 53%、GAT 兩層 = 兩跳，一跳即幾乎覆蓋全圖。

不變的三件事（確保改動單一可歸因）：
    - gate 仍是唯一的跨市場閘門
    - 恆等邊權重仍固定 1（可學權重是另一個實驗）
    - p(j) < 0 仍補零向量

融合維度：
    在 H = gat.hidden_dim 融合，而非 late fusion 的 d'。GATEncoder 第一層的
    in_dim 就是 gat.hidden_dim，所以 GAT_L2 完全不用改。代價是「共同潛在空間」
    的維度由 d'=32 變成 H=64，型別投影仍在融合前各做一次，語意保留。
    投影到 d' 移到圖之後，供 head 使用。

    這使本變體的參數量與 late fusion 略有差異（proj 的輸出維度不同）。
    做架構比較時必須揭露，不可把參數量差異造成的效果算在融合位置上。

實作採繼承 MAGNET 再替換 Phase 2/3 模組：耦合邏輯（pair_src 對齊、弱連結、
逐快照 GAT、節點數守衛、compute_loss）與母類共用同一份，避免兩邊漂移。
副作用是 super().__init__() 會先建出用不到的 d' 版模組再被覆寫，消耗掉一段
RNG，故本變體與 MAGNET 在同一個 seed 下的初始權重不可比——架構不同，本來
就不該逐權重比較。
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.models.encoders import GATEncoder, TypeProjection
from src.models.fusion import CrossLayerFusion
from src.models.multiplex_gnn import MAGNET


class MAGNETIntermediate(MAGNET):
    """MAGNET，但融合發生在 L2 圖傳播之前。"""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        m_cfg    = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        gat_cfg  = m_cfg["gat"]
        proj_cfg = m_cfg["projection"]
        fuse_cfg = m_cfg["fusion"]

        H_lstm = lstm_cfg["hidden_dim"]
        H_gat  = gat_cfg["hidden_dim"]
        H      = H_gat                      # 融合維度 = GAT_L2 的 in_dim
        d_prime = proj_cfg["d_prime"]

        # 型別投影改為投到 H（融合維度），兩側各一份
        proj_cfg_H = {**proj_cfg, "d_prime": H}
        self.proj_L1 = TypeProjection(proj_cfg_H, in_dim=H_gat)    # GAT_L1 之後
        self.proj_L2 = TypeProjection(proj_cfg_H, in_dim=H_lstm)   # 圖之前，吃 LSTM 輸出
        self.fusion  = CrossLayerFusion(fuse_cfg, d_prime=H)

        # 圖之後才投影到 d' 供 head 使用
        self.proj_out = TypeProjection(proj_cfg, in_dim=H_gat)
        assert self.head.mlp[0].in_features == d_prime

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        x_L1 = batch["x_seq_L1"]          # [B, T, n1, F]
        x_L2 = batch["x_seq_L2"]          # [B, T, n2, F]
        B = x_L1.size(0)
        self._assert_node_counts(x_L1.size(2), x_L2.size(2))

        # ── Phase 1: 時序編碼 ─────────────────────────────────────
        h_lstm_L1 = self.lstm(x_L1)       # [B, n1, H_lstm]
        h_lstm_L2 = self.lstm(x_L2)       # [B, n2, H_lstm]

        # 美股側先自行做層內傳播（與 late fusion 相同）
        h_gat_L1 = self._apply_gat_batched(
            self.gat_L1, h_lstm_L1, batch["edge_index_L1"], batch["edge_attr_L1"])

        # ── Phase 2: 融合（提前到 L2 圖之前）─────────────────────
        z_L1 = self.proj_L1(h_gat_L1)     # [B, n1, H]
        z_L2 = self.proj_L2(h_lstm_L2)    # [B, n2, H]  台股圖尚未跑
        if self.disable_a12:
            z_L1_in = z_L1.new_zeros(B, self.n_l2, z_L1.size(-1))
        else:
            z_L1_in = self._augment_weak(z_L1)                     # [B, n2, H]
        h_seed, alpha, gate = self.fusion(z_L1_in, z_L2)           # [B, n2, H]

        # ── Phase 2b: 台股圖傳播——此處才是跨市場訊號的擴散步驟 ──
        h_out = self._apply_gat_batched(
            self.gat_L2, h_seed, batch["edge_index_L2"], batch["edge_attr_L2"])

        # ── Phase 3: 預測 ────────────────────────────────────────
        h_final = self.proj_out(h_out)    # [B, n2, d']
        y_hat = self.head(h_final)        # [B, n2]

        extras = {
            "h_L1":    z_L1,              # [B, n1, H]  未對齊，停留在 L1 索引空間
            "h_L2":    z_L2,              # [B, n2, H]  圖之前的台股表示
            "h_fused": h_seed,            # [B, n2, H]  融合後、傳播前
            "h_out":   h_final,           # [B, n2, d'] 傳播後（供分析）
            "alpha":   alpha,
            "gate":    gate,
        }
        if self.weak_mode is not None:
            extras["weak_beta"] = self.weak_beta * self.weak_mask
        return y_hat, extras
