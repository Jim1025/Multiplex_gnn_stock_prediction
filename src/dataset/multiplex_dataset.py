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
  4. 節點順序固定為 list(PAIR_MAP.keys())（與 graph_builder 對齊，A12 對角依賴）
  5. Sentiment 不在 M3 處理，留到 M6（不讀 data/raw/sentiment/）
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

from src.dataset.config import PAIR_MAP


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
        config_path (str): base.yaml 路徑（讀取 data.split）

    每筆樣本（dict）：
        x_seq_L1        : [T, n, F]  Float32
        x_seq_L2        : [T, n, F]  Float32
        edge_index_L1   : [2, E1]    Long
        edge_attr_L1    : [E1, 1]    Float32
        edge_index_L2   : [2, E2]    Long
        edge_attr_L2    : [E2, 1]    Float32
        y               : [n]        Float32   (TW(t+1) log_return)
        target_date     : str
        is_long_gap_L1  : [n]        Bool
        is_long_gap_L2  : [n]        Bool
    """

    def __init__(
        self,
        snapshot_dir: str = "data/graphs/snapshots",
        features_dir: str = "data/features",
        T: int = 20,
        split: str = "train",
        config_path: str = "configs/base.yaml",
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
        self._adr_dfs: dict[str, pd.DataFrame] = self._load_features("adr", ADR_TICKERS)
        self._tw_dfs:  dict[str, pd.DataFrame] = self._load_features("tw",  TW_CODES)

        # 印出 split 摘要
        print(
            f"[MultiplexDataset] split={split:<5} | n_samples={len(self.snapshot_files):<5} "
            f"| T={T} | F={F} | n_nodes={N_NODES}"
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
        x_seq_L1 = self._build_sequence(self._adr_dfs, ADR_TICKERS, target_date)  # [T, n, F]
        x_seq_L2 = self._build_sequence(self._tw_dfs,  TW_CODES,    target_date)  # [T, n, F]

        # ── 從快照讀邊資訊與標籤 ─────────────────────────────────
        edge_index_L1 = snap[("adr", "corr", "adr")].edge_index.long()       # [2, E1]
        edge_attr_L1  = snap[("adr", "corr", "adr")].edge_attr.float()       # [E1, 1]
        edge_index_L2 = snap[("tw",  "corr", "tw")].edge_index.long()        # [2, E2]
        edge_attr_L2  = snap[("tw",  "corr", "tw")].edge_attr.float()        # [E2, 1]

        y = snap["tw"].y.float()                                              # [n]

        # 追溯欄位（M3 暫不用，留給 M4 loss masking）
        is_long_gap_L1 = (
            snap["adr"].is_long_gap.bool()
            if hasattr(snap["adr"], "is_long_gap")
            else torch.zeros(N_NODES, dtype=torch.bool)
        )
        is_long_gap_L2 = (
            snap["tw"].is_long_gap.bool()
            if hasattr(snap["tw"], "is_long_gap")
            else torch.zeros(N_NODES, dtype=torch.bool)
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
        從各 ticker 的 CSV 截取 < target_date 的最後 T 列，組成 [T, n, F]。

        Look-ahead 守護：
            csv.loc[csv.index < target_date].iloc[-T:]
            嚴格 `<`，絕不可碰 target_date 當天。

        異常處理：
            - 不足 T 列時左側 zero-pad（極少數早期快照才會發生）
            - NaN 防禦性填 0（與 graph_builder NAN_FILL_VALUE=0.0 一致）

        Returns:
            x : [T, n, F]  Float32 tensor
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
    @staticmethod
    def get_ticker_order() -> tuple[list[str], list[str]]:
        """回傳 (adr_tickers, tw_codes) 順序，外部測試用以驗證 A12 對齊。"""
        return list(ADR_TICKERS), list(TW_CODES)


# ---------------------------------------------------------------------------
# collate_fn — 處理每張快照邊數不同的 batching
# ---------------------------------------------------------------------------

def multiplex_collate(batch: list[dict]) -> dict:
    """
    Batch 整理函數。

    - x_seq_L1, x_seq_L2 → stack 成 [B, T, n, F]
    - y                  → stack 成 [B, n]
    - edge_index_*, edge_attr_* → list（每張快照邊數不同，
      由 MAGNET._apply_gat_batched 內部逐張處理）
    """
    out: dict = {
        "x_seq_L1": torch.stack([b["x_seq_L1"] for b in batch], dim=0),   # [B, T, n, F]
        "x_seq_L2": torch.stack([b["x_seq_L2"] for b in batch], dim=0),
        "y":        torch.stack([b["y"]        for b in batch], dim=0),   # [B, n]
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
