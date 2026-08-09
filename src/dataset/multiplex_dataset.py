"""
multiplex_dataset.py — M3 訓練用 Dataset Wrapper
Corresponds to IMPLEMENTATION_SPEC §8.2 / §3.1 (LSTM 時序輸入組裝)

責任：
  - 讀取 graph_builder 產出的 .pt 圖快照
  - 從 features CSV 動態組裝過去 T 步歷史序列（給 SharedLSTM）
  - 嚴格 Look-ahead 守護：取得的最後一列日期 ≤ target_date - 1 trading day
  - Walk-forward 切分：train / val / test

設計決策（已與使用者確認）：
  1. T=20 步切片用各自市場日曆（ADR=NYSE / TW=XTAI）
  2. NaN 防禦性填 0（與 graph_builder 的 NAN_FILL_VALUE=0.0 一致）
  3. 暖機期快照完全排除（直接信任 graph_builder 的 1,643 張）
  4. 節點順序由 Universe 決定（與 graph_builder 用同一個定義，見下方 E5 說明）
  5. Sentiment 不在 M3 處理，留到 M6（不讀 data/raw/sentiment/）

─────────────────────────────────────────────────────────────────────────
E5 變更說明（經授權變更資料層，2026-08-09）

為什麼要改：
  本檔原本用三個模組層級常數（ADR_TICKERS / TW_CODES / N_NODES）當作
  唯一的節點定義，而它們都由 PAIR_MAP 推導，隱含 n_l1 == n_l2 == 7。
  E4 之後 graph_builder 已能產出 L1=30 / L2=50 的不對稱快照，但本檔仍會
  用 7 檔的清單去組 x_seq——序列與快照對到不同的 universe，維度不合會在
  模型內部才炸開，訊息也看不出真正原因。

  更危險的是「維度碰巧相容」的情形：只要兩個 universe 的節點數相同，
  整條流程不會有任何例外，訓練照跑，但 x_seq 的第 j 欄與 y 的第 j 欄
  是不同公司。這種錯誤在指標上看不出來，只會表現成 IC 偏低。

本次改動：
  1. __init__ 新增 universe 參數（可傳 Universe 物件或名稱字串）；
     未指定時讀 base.yaml 的 data.universe，仍預設 k7。
  2. 節點清單改為實例屬性 self.adr_tickers / self.tw_codes，
     兩層各自的長度為 self.n_l1 / self.n_l2。
  3. 建構時檢查第一張快照的節點數是否等於 universe 的節點數，
     不符直接拋錯——把上述靜默錯誤變成 fail fast。
  4. get_ticker_order() 由 staticmethod 改為實例方法。原本它無論實例
     持有哪個 universe 都回傳 k7 的順序，是個會給出錯誤答案的介面。

影響範圍：
  k7 下 universe 推導出的清單與 ADR_TICKERS / TW_CODES 逐項相同，
  故樣本內容不變。驗證：freeze_k7.py --verify 應維持 17/17 通過。
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor
from torch.utils.data import Dataset

from src.dataset.config import DEFAULT_UNIVERSE, PAIR_MAP, Universe, load_universe


# ---------------------------------------------------------------------------
# 9 維技術指標欄位（與 features.py 的 TECH_FEATURE_COLS 一致）
# ---------------------------------------------------------------------------
TECH_COLS: list[str] = [
    "log_return",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_pos",
    "MA5_dev", "MA20_dev",
    "log_volume_z",
]

F = len(TECH_COLS)   # 9

# ---------------------------------------------------------------------------
# k=7 相容常數
#
# 這三個名字散落在 evaluator / baseline_early_fusion / freeze_k7 與測試中。
# 它們的語意是「k7 這個 universe 的節點順序」，不是「本 Dataset 的節點順序」
# ——擴充後 n_l1 != n_l2，N_NODES 已無定義。新程式碼請改用實例屬性
# .adr_tickers / .tw_codes / .n_l1 / .n_l2。
#
# 刻意仍由 PAIR_MAP 獨立推導（而非 load_universe("k7")）：
# tests/test_universe.py 靠「兩條路徑各自算出同樣答案」來驗證 Universe API
# 沒有改動 k7 的節點順序；若改成從 Universe 取值，那個測試會變成恆真。
# ---------------------------------------------------------------------------
ADR_TICKERS: list[str] = list(PAIR_MAP.keys())                  # [TSM, UMC, ASX, CHT, IMOS, AUOTY, HNHPF]
TW_CODES:    list[str] = [v["tw"] for v in PAIR_MAP.values()]    # [2330, 2303, 3711, 2412, 8150, 2409, 2317]
N_NODES = len(ADR_TICKERS)                                       # 7


# ---------------------------------------------------------------------------
# MultiplexDataset
# ---------------------------------------------------------------------------

class MultiplexDataset(Dataset):
    """
    Multiplex GNN 訓練用 Dataset。

    Corresponds to IMPLEMENTATION_SPEC §8.2

    Args:
        snapshot_dir (str): 圖快照根目錄（預設 data/graphs/snapshots）
        features_dir (str): 特徵 CSV 根目錄（預設 data/features，含 adr/ tw/）
        T (int):           LSTM 回看步數（預設 20）
        split (str):       "train" | "val" | "test" | "all"
        config_path (str): base.yaml 路徑（讀取 data.split 與 data.universe）
        universe:          Universe 物件或名稱字串（"k7" | "tw50"）。
                           未指定時取 base.yaml 的 data.universe。
                           必須與 snapshot_dir 是同一個 universe 建的，
                           建構時會檢查。

    每筆樣本（dict）：
        x_seq_L1        : [T, n1, F]  Float32
        x_seq_L2        : [T, n2, F]  Float32
        edge_index_L1   : [2, E1]     Long
        edge_attr_L1    : [E1, 1]     Float32
        edge_index_L2   : [2, E2]     Long
        edge_attr_L2    : [E2, 1]     Float32
        y               : [n2]        Float32   (TW(t+1) log_return)
        target_date     : str
        is_long_gap_L1  : [n1]        Bool
        is_long_gap_L2  : [n2]        Bool

    註：n1（US 層）與 n2（TW 層）不必相等。k7 下兩者都是 7，
        tw50 下是 30 與 50。
    """

    def __init__(
        self,
        snapshot_dir: str = "data/graphs/snapshots",
        features_dir: str = "data/features",
        T: int = 20,
        split: str = "train",
        config_path: str = "configs/base.yaml",
        universe: Universe | str | None = None,
    ) -> None:
        super().__init__()

        assert split in {"train", "val", "test", "all"}, f"invalid split: {split}"

        self.snapshot_dir = snapshot_dir
        self.features_dir = features_dir
        self.T = T
        self.split = split

        # ── 讀取 config.yaml 的 split 設定 ─────────────────────────
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        split_cfg = cfg["data"]["split"]
        train_end = split_cfg["train_end"]
        val_end   = split_cfg["val_end"]

        # ── 決定 universe ─────────────────────────────────────────
        # 優先序：明確傳入的參數 > base.yaml 的 data.universe > DEFAULT_UNIVERSE
        # （名稱不在此硬編，與其他超參一致由 config 提供）
        if universe is None:
            universe = cfg["data"].get("universe", DEFAULT_UNIVERSE)
        if isinstance(universe, str):
            universe = load_universe(universe)

        self.universe    = universe
        self.adr_tickers = list(universe.us_nodes)
        self.tw_codes    = list(universe.tw_nodes)
        self.n_l1        = universe.n_l1
        self.n_l2        = universe.n_l2
        # pair_index[j] = TW 節點 j 對應的 US 索引；-1 代表無配對（E6 會用到）
        self.pair_index  = list(universe.pair_index)

        # ── 掃描所有快照檔，按檔名（含日期）排序 ─────────────────
        all_files = sorted(glob.glob(os.path.join(snapshot_dir, "graph_*.pt")))
        if not all_files:
            raise FileNotFoundError(
                f"找不到任何圖快照於 {snapshot_dir}/graph_*.pt"
            )

        # 依 split 切片（walk-forward，禁止隨機）
        if split == "train":
            self.snapshot_files = all_files[:train_end]
        elif split == "val":
            self.snapshot_files = all_files[train_end:val_end]
        elif split == "test":
            self.snapshot_files = all_files[val_end:]
        else:
            self.snapshot_files = all_files

        # ── 預載所有特徵 CSV 到記憶體 ────────────────────────────
        self._adr_dfs: dict[str, pd.DataFrame] = self._load_features("adr", self.adr_tickers)
        self._tw_dfs:  dict[str, pd.DataFrame] = self._load_features("tw",  self.tw_codes)

        # ── 快照與 universe 一致性檢查 ───────────────────────────
        if self.snapshot_files:
            self._assert_snapshot_matches_universe(self.snapshot_files[0])

        # 印出 split 摘要
        print(
            f"[MultiplexDataset] split={split:<5} | n_samples={len(self.snapshot_files):<5} "
            f"| T={T} | F={F} | universe={self.universe.name} "
            f"| n_l1={self.n_l1} n_l2={self.n_l2}"
        )

    # ------------------------------------------------------------------
    # 私有：一致性檢查
    # ------------------------------------------------------------------
    def _assert_snapshot_matches_universe(self, path: str) -> None:
        """
        確認快照的節點數與 universe 相符。

        x_seq 由 universe 的清單即時組裝，y 與邊則來自快照檔。兩者若源自
        不同 universe，只要節點數碰巧相同就不會有任何例外——訓練照跑，
        但每一欄對到的是不同公司。這裡在建構時就擋掉。
        """
        snap = torch.load(path, weights_only=False)
        n_l1 = int(snap["adr"].x.shape[0])
        n_l2 = int(snap["tw"].y.shape[0])
        if (n_l1, n_l2) != (self.n_l1, self.n_l2):
            raise ValueError(
                f"快照與 universe 不符：{os.path.basename(path)} 的節點數為 "
                f"L1={n_l1} / L2={n_l2}，但 universe='{self.universe.name}' 要求 "
                f"L1={self.n_l1} / L2={self.n_l2}。"
                f"請確認 snapshot_dir={self.snapshot_dir} 是用同一個 universe 建的。"
            )

    # ------------------------------------------------------------------
    # 私有：載入特徵
    # ------------------------------------------------------------------
    def _load_features(self, market: str, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """讀取一個市場的所有 ticker CSV，回傳 {ticker: DataFrame}，只保留 9 維 TECH_COLS。"""
        dfs: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            path = os.path.join(self.features_dir, market, f"{ticker}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到特徵檔 {path}")
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            missing = [c for c in TECH_COLS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{path} 缺少欄位 {missing}（應為 TECH_COLS 的子集）"
                )
            dfs[ticker] = df[TECH_COLS].astype(np.float32)
        return dfs

    # ------------------------------------------------------------------
    # Dataset 介面
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.snapshot_files)

    def __getitem__(self, idx: int) -> dict:
        path = self.snapshot_files[idx]
        snap = torch.load(path, weights_only=False)

        target_date_str: str = snap.target_date
        target_date = pd.Timestamp(target_date_str)

        # ── 動態組裝 T 步歷史序列（嚴格 Look-ahead 守護） ─────────
        x_seq_L1 = self._build_sequence(self._adr_dfs, self.adr_tickers, target_date)  # [T, n1, F]
        x_seq_L2 = self._build_sequence(self._tw_dfs,  self.tw_codes,    target_date)  # [T, n2, F]

        # ── 從快照讀邊資訊與標籤 ─────────────────────────────────
        edge_index_L1 = snap[("adr", "corr", "adr")].edge_index.long()       # [2, E1]
        edge_attr_L1  = snap[("adr", "corr", "adr")].edge_attr.float()       # [E1, 1]
        edge_index_L2 = snap[("tw",  "corr", "tw")].edge_index.long()        # [2, E2]
        edge_attr_L2  = snap[("tw",  "corr", "tw")].edge_attr.float()        # [E2, 1]

        y = snap["tw"].y.float()                                              # [n2]

        # 追溯欄位（M3 暫不用，留給 M4 loss masking）
        is_long_gap_L1 = (
            snap["adr"].is_long_gap.bool()
            if hasattr(snap["adr"], "is_long_gap")
            else torch.zeros(self.n_l1, dtype=torch.bool)
        )
        is_long_gap_L2 = (
            snap["tw"].is_long_gap.bool()
            if hasattr(snap["tw"], "is_long_gap")
            else torch.zeros(self.n_l2, dtype=torch.bool)
        )

        return {
            "x_seq_L1":       x_seq_L1,
            "x_seq_L2":       x_seq_L2,
            "edge_index_L1":  edge_index_L1,
            "edge_attr_L1":   edge_attr_L1,
            "edge_index_L2":  edge_index_L2,
            "edge_attr_L2":   edge_attr_L2,
            "y":              y,
            "target_date":    target_date_str,
            "is_long_gap_L1": is_long_gap_L1,
            "is_long_gap_L2": is_long_gap_L2,
        }

    # ------------------------------------------------------------------
    # 私有：組裝 T 步歷史序列
    # ------------------------------------------------------------------
    def _build_sequence(
        self,
        dfs:     dict[str, pd.DataFrame],
        tickers: list[str],
        target_date: pd.Timestamp,
    ) -> Tensor:
        """
        從各 ticker 的 CSV 截取 < target_date 的最後 T 列，組成 [T, len(tickers), F]。

        兩層各自呼叫一次；tickers 長度不必相同（tw50 下為 30 與 50）。

        Look-ahead 守護：
            csv.loc[csv.index < target_date].iloc[-T:]
            嚴格 `<`，絕不可碰 target_date 當天。

        異常處理：
            - 不足 T 列時左側 zero-pad（極少數早期快照才會發生）
            - NaN 防禦性填 0（與 graph_builder NAN_FILL_VALUE=0.0 一致）

        Returns:
            x : [T, len(tickers), F]  Float32 tensor
        """
        per_ticker_arrays = []
        for ticker in tickers:
            df = dfs[ticker]
            past = df.loc[df.index < target_date]
            window = past.iloc[-self.T:].to_numpy(dtype=np.float32)  # [t, F]，t ≤ T

            t = window.shape[0]
            if t < self.T:
                # 左側 zero-pad（極少數早期快照才會觸發）
                pad = np.zeros((self.T - t, F), dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)       # [T, F]
                warnings.warn(
                    f"[{ticker}] target_date={target_date.date()} 歷史不足 T={self.T}，"
                    f"左側 zero-pad {self.T - t} 列",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # NaN 防禦性填 0
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
            per_ticker_arrays.append(window)

        # stack [n, T, F] → transpose [T, n, F]
        x = np.stack(per_ticker_arrays, axis=0).transpose(1, 0, 2)
        return torch.from_numpy(x)                                    # Float32

    # ------------------------------------------------------------------
    # 便利方法
    # ------------------------------------------------------------------
    def get_ticker_order(self) -> tuple[list[str], list[str]]:
        """
        回傳本實例的 (adr_tickers, tw_codes) 順序，外部測試用以驗證 A12 對齊。

        E5 由 staticmethod 改為實例方法：原本無論實例持有哪個 universe
        都回傳 k7 的順序，是個會靜默給出錯誤答案的介面。
        """
        return list(self.adr_tickers), list(self.tw_codes)

    # ── 便利屬性 ──────────────────────────────────────────────────
    @property
    def n_nodes(self) -> tuple[int, int]:
        """(n_l1, n_l2)。刻意不提供單一 n——擴充後兩層節點數不相等。"""
        return self.n_l1, self.n_l2


# ---------------------------------------------------------------------------
# collate_fn — 處理每張快照邊數不同的 batching
# ---------------------------------------------------------------------------

def multiplex_collate(batch: list[dict]) -> dict:
    """
    Batch 整理函數。

    - x_seq_L1 → stack 成 [B, T, n1, F]；x_seq_L2 → [B, T, n2, F]
    - y                  → stack 成 [B, n2]
    - edge_index_*, edge_attr_* → list（每張快照邊數不同，
      由 MAGNET._apply_gat_batched 內部逐張處理）
    """
    out: dict = {
        "x_seq_L1": torch.stack([b["x_seq_L1"] for b in batch], dim=0),   # [B, T, n1, F]
        "x_seq_L2": torch.stack([b["x_seq_L2"] for b in batch], dim=0),   # [B, T, n2, F]
        "y":        torch.stack([b["y"]        for b in batch], dim=0),   # [B, n2]
        "is_long_gap_L1": torch.stack([b["is_long_gap_L1"] for b in batch], dim=0),
        "is_long_gap_L2": torch.stack([b["is_long_gap_L2"] for b in batch], dim=0),

        # 邊資訊每張快照不同，保持為 list
        "edge_index_L1": [b["edge_index_L1"] for b in batch],
        "edge_attr_L1":  [b["edge_attr_L1"]  for b in batch],
        "edge_index_L2": [b["edge_index_L2"] for b in batch],
        "edge_attr_L2":  [b["edge_attr_L2"]  for b in batch],

        "target_date": [b["target_date"] for b in batch],
    }
    return out
