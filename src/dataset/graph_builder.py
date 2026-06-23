"""
graph_builder.py — Multiplex 圖快照建構器（MVP v1.0）
========================================================
承接 features.py 產出的 9 維特徵 CSV，為每一個目標日期產出一張
PyTorch Geometric HeteroData 圖快照，作為下游 GNN 訓練的輸入。

對應論文 M2.1–M2.4：
  L1 ADR Graph：節點 = 7 ADR，邊 = |ρ| > threshold 的動態相關係數邊
  L2 TW Graph： 節點 = 7 TW，  邊 = |ρ| > threshold 的動態相關係數邊
                              （未來補上：靜態供應鏈邊）
  A12 Cross-layer：嚴格對角矩陣，1-to-1 配對防止跨公司洩漏

設計原則（4 個關鍵決策）：
  ① 邊權重 = |ρ|（保留相關強度，GAT 可學習）
  ② Sentiment 全域特徵留待 Phase 1 LSTM 編碼器階段（不在此處理）
  ③ 靜態供應鏈邊延後（先用純動態邊跑 baseline）
  ④ 相關係數閾值預設 0.3（7 節點時 0.5 會太稀疏）

時序對齊邏輯（核心防護）：
  對於目標日 target_date（即預測 TW(target_date) 報酬的那天）：
    回看視窗：[target_date - corr_window 工作日, target_date - 1]
    L1 節點特徵：使用 ADR 在 target_date - 1 日的 9 維特徵
    L2 節點特徵：使用 TW  在 target_date - 1 日的 9 維特徵
    預測標的：  TW 在 target_date 的 log_return
  
  關鍵：所有用於建圖的資料必須嚴格早於 target_date，
        防止 Look-ahead Bias。

執行方式：
    python src/dataset/graph_builder.py \\
        --start-date 2019-01-01 \\
        --end-date 2025-12-31

作為模組：
    from graph_builder import MultiplexGraphBuilder
    gb = MultiplexGraphBuilder(corr_window=60, corr_threshold=0.3)
    snapshot = gb.build_snapshot(pd.Timestamp("2024-01-15"))
"""

import os
import sys
import glob
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import HeteroData


# ════════════════════════════════════════════════════════════
# 常數
# ════════════════════════════════════════════════════════════

# 9 維節點特徵（與 features.py TECH_FEATURE_COLS 對齊）
NODE_FEATURE_COLS = [
    "log_return", "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_pos", "MA5_dev", "MA20_dev",
    "log_volume_z",
]
N_FEATURES = len(NODE_FEATURE_COLS)  # = 9

# 預設參數
DEFAULT_CORR_WINDOW    = 60    # 回看視窗
DEFAULT_CORR_THRESHOLD = 0.3   # |ρ| 閾值（7 節點下 0.5 過於稀疏）
DEFAULT_USE_ABS_CORR   = True  # |ρ| 取絕對值（同時保留正負相關邊）

# PyG HeteroData 邊類型命名（下游 GAT 會引用這些名稱）
EDGE_L1_TYPE     = ("adr", "corr",  "adr")
EDGE_L2_TYPE     = ("tw",  "corr",  "tw")
EDGE_A12_TYPE    = ("adr", "cross", "tw")

# 政策：long_gap 列的特徵為 NaN，不應進入訓練
NAN_FILL_VALUE = 0.0   # 為了讓 PyG tensor 不含 NaN，補 0 並設 mask


# ════════════════════════════════════════════════════════════
# 報告容器
# ════════════════════════════════════════════════════════════

@dataclass
class SnapshotMetadata:
    """單張圖快照的中繼資訊。"""
    target_date:     pd.Timestamp
    window_start:    pd.Timestamp
    window_end:      pd.Timestamp
    n_l1_nodes:      int = 0
    n_l2_nodes:      int = 0
    n_l1_edges:      int = 0
    n_l2_edges:      int = 0
    n_a12_edges:     int = 0
    l1_density:      float = 0.0
    l2_density:      float = 0.0
    has_imputed:     bool = False
    has_long_gap:    bool = False
    n_invalid_nodes: int = 0      # 因 NaN 等導致無效的節點數
    warnings:        List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_date"]  = self.target_date.strftime("%Y-%m-%d")
        d["window_start"] = self.window_start.strftime("%Y-%m-%d")
        d["window_end"]   = self.window_end.strftime("%Y-%m-%d")
        d["warnings"]     = "; ".join(d["warnings"])
        return d


# ════════════════════════════════════════════════════════════
# 主類別
# ════════════════════════════════════════════════════════════

class MultiplexGraphBuilder:
    """
    Multiplex 圖快照建構器。

    Parameters
    ----------
    pair_map : Dict[str, str]
        ADR ticker → TW code 的對應字典，例如 {"TSM": "2330", ...}
    adr_dir, tw_dir : str
        features.py 產出的 CSV 目錄
    corr_window : int
        相關係數回看視窗（工作日數，預設 60）
    corr_threshold : float
        |ρ| 閾值，超過此值才建邊（預設 0.3）
    use_abs_corr : bool
        是否取絕對值（True 同時保留正負相關，False 只保留正相關）
    feature_col_for_corr : str
        用於計算相關係數的欄位（預設 log_return）
    static_supply_chain_path : str, optional
        靜態供應鏈邊 CSV 路徑（未來支援，目前不使用）
    """

    def __init__(self,
                 pair_map:                Dict[str, str],
                 adr_dir:                 str = "data/features/adr",
                 tw_dir:                  str = "data/features/tw",
                 corr_window:             int = DEFAULT_CORR_WINDOW,
                 corr_threshold:          float = DEFAULT_CORR_THRESHOLD,
                 use_abs_corr:            bool = DEFAULT_USE_ABS_CORR,
                 feature_col_for_corr:    str = "log_return",
                 static_supply_chain_path: Optional[str] = None):

        self.pair_map             = dict(pair_map)
        self.adr_dir              = adr_dir
        self.tw_dir               = tw_dir
        self.corr_window          = corr_window
        self.corr_threshold       = corr_threshold
        self.use_abs_corr         = use_abs_corr
        self.feature_col_for_corr = feature_col_for_corr

        # 靜態供應鏈邊：MVP 暫不使用，預留介面
        self.static_supply_chain_path = static_supply_chain_path
        if static_supply_chain_path:
            warnings.warn(
                "static_supply_chain_path 已預留但 MVP 版本未實作，將忽略此參數",
                stacklevel=2,
            )

        # 固定節點順序（極為重要：A12 對角矩陣依賴此順序）
        self.adr_tickers = list(self.pair_map.keys())
        self.tw_codes    = list(self.pair_map.values())
        self.n_nodes     = len(self.adr_tickers)

        # 載入所有特徵資料（一次性載入，後續快取使用）
        self._adr_data: Dict[str, pd.DataFrame] = {}
        self._tw_data:  Dict[str, pd.DataFrame] = {}
        self._load_all_features()

    # ── 私有：資料載入 ────────────────────────────────────────

    def _load_all_features(self):
        """一次性載入所有 ticker 的 features CSV。"""
        for ticker in self.adr_tickers:
            path = Path(self.adr_dir) / f"{ticker}.csv"
            if not path.exists():
                raise FileNotFoundError(f"ADR features 檔案不存在：{path}")
            self._adr_data[ticker] = pd.read_csv(
                path, index_col="Date", parse_dates=True
            ).sort_index()

        for code in self.tw_codes:
            path = Path(self.tw_dir) / f"{code}.csv"
            if not path.exists():
                raise FileNotFoundError(f"TW features 檔案不存在：{path}")
            self._tw_data[code] = pd.read_csv(
                path, index_col="Date", parse_dates=True
            ).sort_index()

    # ── 公開：建構單一快照 ───────────────────────────────────

    def build_snapshot(self,
                       target_date: pd.Timestamp,
                       ) -> Tuple[Optional[HeteroData], SnapshotMetadata]:
        """
        為單一目標日期建立一張多層圖快照。

        Parameters
        ----------
        target_date : pd.Timestamp
            預測目標日（即「未來」的那一天，用於取 y 標籤）

        Returns
        -------
        data : HeteroData or None
            None 代表該日無法建立快照（資料不足等原因）
        metadata : SnapshotMetadata
            該快照的中繼資訊（即使建立失敗也會回傳）
        """
        target_date = pd.Timestamp(target_date).normalize()

        # ── Step 1：取得回看視窗（嚴格早於 target_date）──────
        # 用 TW 的交易日作為基準（因為預測目標是 TW）
        sample_tw = next(iter(self._tw_data.values()))
        tw_dates  = sample_tw.index

        # 找到 target_date 在 TW 索引中的位置
        # 若 target_date 不存在於 TW 索引，找最近的下一個交易日
        future_mask = tw_dates >= target_date
        if not future_mask.any():
            return None, SnapshotMetadata(
                target_date=target_date,
                window_start=target_date,
                window_end=target_date,
                warnings=["target_date 超出資料範圍"],
            )
        target_pos = future_mask.argmax()
        actual_target_date = tw_dates[target_pos]

        # 視窗：取 target_date 之前的 corr_window 個交易日
        if target_pos < self.corr_window:
            return None, SnapshotMetadata(
                target_date=actual_target_date,
                window_start=tw_dates[0],
                window_end=tw_dates[max(0, target_pos - 1)],
                warnings=[f"暖機期不足，需要至少 {self.corr_window} 個歷史交易日"],
            )

        window_start = tw_dates[target_pos - self.corr_window]
        window_end   = tw_dates[target_pos - 1]   # 嚴格早於 target_date

        meta = SnapshotMetadata(
            target_date=actual_target_date,
            window_start=window_start,
            window_end=window_end,
            n_l1_nodes=self.n_nodes,
            n_l2_nodes=self.n_nodes,
        )

        # ── Step 2：取得節點特徵（使用 window_end 那天的特徵）─
        l1_features, l1_imp_mask, l1_lg_mask = self._extract_node_features(
            self._adr_data, self.adr_tickers, window_end, meta, "L1"
        )
        l2_features, l2_imp_mask, l2_lg_mask = self._extract_node_features(
            self._tw_data, self.tw_codes, window_end, meta, "L2"
        )

        # ── Step 3：計算 L1 / L2 相關係數邊 ────────────────
        l1_edge_index, l1_edge_attr = self._build_correlation_edges(
            self._adr_data, self.adr_tickers,
            window_start, window_end, meta, "L1"
        )
        l2_edge_index, l2_edge_attr = self._build_correlation_edges(
            self._tw_data, self.tw_codes,
            window_start, window_end, meta, "L2"
        )

        # ── Step 4：建立 A12 對角矩陣 ──────────────────────
        a12_edge_index = torch.tensor(
            [list(range(self.n_nodes)), list(range(self.n_nodes))],
            dtype=torch.long,
        )
        meta.n_a12_edges = self.n_nodes

        # ── Step 5：取得預測目標 y（TW 在 target_date 的 log_return）
        y = self._extract_target_returns(actual_target_date, meta)

        # ── Step 6：封裝為 HeteroData ─────────────────────
        data = HeteroData()

        # 節點特徵
        data["adr"].x = l1_features
        data["tw" ].x = l2_features

        # 政策旗標（供下游訓練決定是否剔除）
        data["adr"].is_imputed  = l1_imp_mask
        data["adr"].is_long_gap = l1_lg_mask
        data["tw" ].is_imputed  = l2_imp_mask
        data["tw" ].is_long_gap = l2_lg_mask

        # L1 邊
        data[EDGE_L1_TYPE].edge_index = l1_edge_index
        data[EDGE_L1_TYPE].edge_attr  = l1_edge_attr

        # L2 邊
        data[EDGE_L2_TYPE].edge_index = l2_edge_index
        data[EDGE_L2_TYPE].edge_attr  = l2_edge_attr

        # A12 跨層邊
        data[EDGE_A12_TYPE].edge_index = a12_edge_index

        # 預測目標
        data["tw"].y = y

        # 中繼資訊（PyG 允許自訂屬性）
        data.target_date  = pd.Timestamp(actual_target_date).strftime("%Y-%m-%d")
        data.window_start = window_start.strftime("%Y-%m-%d")
        data.window_end   = window_end.strftime("%Y-%m-%d")
        data.lookback_days = self.corr_window

        # 統計
        meta.l1_density = (
            meta.n_l1_edges / max(self.n_nodes * (self.n_nodes - 1), 1)
        )
        meta.l2_density = (
            meta.n_l2_edges / max(self.n_nodes * (self.n_nodes - 1), 1)
        )
        meta.has_imputed  = bool(l1_imp_mask.any() or l2_imp_mask.any())
        meta.has_long_gap = bool(l1_lg_mask.any() or l2_lg_mask.any())

        return data, meta

    # ── 公開：批次建構序列 ───────────────────────────────────

    def build_sequence(self,
                       start_date: pd.Timestamp,
                       end_date:   pd.Timestamp,
                       out_dir:    str = "data/graphs/snapshots",
                       ) -> List[SnapshotMetadata]:
        """
        對 [start_date, end_date] 範圍內每個 TW 交易日產出快照。

        Parameters
        ----------
        start_date, end_date : pd.Timestamp
            目標日範圍（含端點）
        out_dir : str
            快照儲存目錄

        Returns
        -------
        all_meta : List[SnapshotMetadata]
            所有快照的中繼資訊（含成功與失敗）
        """
        os.makedirs(out_dir, exist_ok=True)

        start_date = pd.Timestamp(start_date).normalize()
        end_date   = pd.Timestamp(end_date).normalize()

        # 用 TW 交易日列表作為「目標日」候選
        sample_tw  = next(iter(self._tw_data.values()))
        target_dates = sample_tw.index[
            (sample_tw.index >= start_date) & (sample_tw.index <= end_date)
        ]

        print(f"\n[MultiplexGraphBuilder] {start_date.date()} ~ {end_date.date()}")
        print(f"  pair_map: {len(self.pair_map)} 組節點")
        print(f"  corr_window={self.corr_window}, threshold={self.corr_threshold}")
        print(f"  目標日候選：{len(target_dates)} 個")

        all_meta: List[SnapshotMetadata] = []
        n_success, n_skip = 0, 0

        for d in target_dates:
            data, meta = self.build_snapshot(d)
            all_meta.append(meta)

            if data is None:
                n_skip += 1
                continue

            # 序列化
            fname = f"graph_{d.strftime('%Y-%m-%d')}.pt"
            torch.save(data, Path(out_dir) / fname)
            n_success += 1

        # 寫批次報告
        report_path = Path(out_dir) / "graph_metadata.csv"
        pd.DataFrame([m.to_dict() for m in all_meta]).to_csv(
            report_path, index=False, encoding="utf-8-sig"
        )

        print(f"\n  產出快照：{n_success} 張")
        print(f"  — 跳過：{n_skip} 張（暖機期等原因）")
        print(f"  報告 → {report_path}")

        return all_meta

    # ── 私有：節點特徵提取 ───────────────────────────────────

    def _extract_node_features(self,
                               data_dict:    Dict[str, pd.DataFrame],
                               tickers:      List[str],
                               feature_date: pd.Timestamp,
                               meta:         SnapshotMetadata,
                               layer_label:  str,
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        取得 feature_date 那天的 9 維特徵。

        Returns
        -------
        features : torch.Tensor [n_nodes, N_FEATURES]
        is_imputed_mask : torch.BoolTensor [n_nodes]
        is_long_gap_mask : torch.BoolTensor [n_nodes]
        """
        n = len(tickers)
        feat_arr = np.full((n, N_FEATURES), NAN_FILL_VALUE, dtype=np.float32)
        imp_mask = np.zeros(n, dtype=bool)
        lg_mask  = np.zeros(n, dtype=bool)

        for i, ticker in enumerate(tickers):
            df = data_dict[ticker]

            # 找最接近 feature_date 但不晚於它的列
            valid_dates = df.index[df.index <= feature_date]
            if len(valid_dates) == 0:
                meta.n_invalid_nodes += 1
                meta.warnings.append(
                    f"{layer_label} {ticker} 無 {feature_date.date()} 之前的資料"
                )
                continue

            actual_date = valid_dates[-1]
            row = df.loc[actual_date]

            # 提取 9 維特徵
            try:
                vals = row[NODE_FEATURE_COLS].values.astype(np.float32)
            except KeyError as e:
                raise ValueError(
                    f"{layer_label} {ticker} 缺少特徵欄位：{e}。"
                    f"請確認已執行 features.py。"
                )

            # 處理 NaN（long_gap 列的特徵為 NaN）
            nan_count = int(np.isnan(vals).sum())
            if nan_count > 0:
                feat_arr[i] = np.nan_to_num(vals, nan=NAN_FILL_VALUE)
            else:
                feat_arr[i] = vals

            # 讀取追溯欄位
            if "is_imputed" in df.columns:
                imp_mask[i] = bool(row.get("is_imputed", False))
            if "is_long_gap" in df.columns:
                lg_mask[i] = bool(row.get("is_long_gap", False))

        return (
            torch.from_numpy(feat_arr),
            torch.from_numpy(imp_mask),
            torch.from_numpy(lg_mask),
        )

    # ── 私有：相關係數邊計算 ─────────────────────────────────

    def _build_correlation_edges(self,
                                 data_dict:    Dict[str, pd.DataFrame],
                                 tickers:      List[str],
                                 window_start: pd.Timestamp,
                                 window_end:   pd.Timestamp,
                                 meta:         SnapshotMetadata,
                                 layer_label:  str,
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算 [window_start, window_end] 視窗內各 ticker 兩兩 Pearson 相關係數，
        產出 |ρ| > threshold 的邊。

        Returns
        -------
        edge_index : torch.LongTensor [2, n_edges]，雙向邊（i→j 與 j→i 各一次）
        edge_attr  : torch.FloatTensor [n_edges, 1]，邊權重 = |ρ|
        """
        n = len(tickers)

        # 收集視窗內每個 ticker 的特徵序列
        series_list = []
        for ticker in tickers:
            df = data_dict[ticker]
            mask = (df.index >= window_start) & (df.index <= window_end)
            s = df.loc[mask, self.feature_col_for_corr]
            series_list.append(s)

        # 對齊到共同日期，組成矩陣
        df_window = pd.concat(series_list, axis=1, join="inner")
        df_window.columns = tickers

        if len(df_window) < self.corr_window // 2:
            meta.warnings.append(
                f"{layer_label} 視窗共同交易日 {len(df_window)} 過少，"
                f"邊數可能偏低"
            )

        # 計算相關矩陣
        corr_mat = df_window.corr(method="pearson").values  # [n, n]

        # 取絕對值（同時保留正負相關）
        if self.use_abs_corr:
            score_mat = np.abs(corr_mat)
        else:
            score_mat = corr_mat

        # 建邊：i ≠ j 且 |ρ| > threshold
        edges_src, edges_dst, edges_w = [], [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rho = score_mat[i, j]
                if np.isnan(rho):
                    continue
                if rho > self.corr_threshold:
                    edges_src.append(i)
                    edges_dst.append(j)
                    edges_w.append(float(np.abs(corr_mat[i, j])))

        if not edges_src:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, 1), dtype=torch.float32)
        else:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            edge_attr  = torch.tensor(edges_w, dtype=torch.float32).unsqueeze(1)

        # 統計（雙向邊：實際無向邊數 = edges / 2）
        if layer_label == "L1":
            meta.n_l1_edges = len(edges_src)
        else:
            meta.n_l2_edges = len(edges_src)

        return edge_index, edge_attr

    # ── 私有：預測目標提取 ───────────────────────────────────

    def _extract_target_returns(self,
                                target_date: pd.Timestamp,
                                meta:        SnapshotMetadata,
                                ) -> torch.Tensor:
        """
        取得 TW 各節點在 target_date 的 log_return（作為 y）。
        若該日 is_long_gap=True 或缺漏，回傳 NaN（下游訓練應剔除）。
        """
        y_arr = np.full(self.n_nodes, np.nan, dtype=np.float32)

        for i, code in enumerate(self.tw_codes):
            df = self._tw_data[code]
            if target_date not in df.index:
                meta.warnings.append(f"TW {code} 缺 {target_date.date()} 資料")
                continue
            row = df.loc[target_date]
            if "is_long_gap" in df.columns and bool(row.get("is_long_gap", False)):
                continue   # 長缺口列保持 NaN
            val = row.get("log_return", np.nan)
            if pd.notna(val):
                y_arr[i] = float(val)

        return torch.from_numpy(y_arr)


# ════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════

def _load_pair_map_from_config() -> Dict[str, str]:
    """從 src.dataset.config 載入 PAIR_MAP（與 test_no_lookahead.py 一致）。"""
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.dataset import config as cfg
    if hasattr(cfg, "get_pair_dict") and callable(cfg.get_pair_dict):
        return cfg.get_pair_dict()
    pm = cfg.PAIR_MAP
    first_val = next(iter(pm.values()))
    if isinstance(first_val, dict):
        return {k: v["tw"] for k, v in pm.items()}
    return dict(pm)


def main():
    parser = argparse.ArgumentParser(description="Multiplex 圖快照建構")
    parser.add_argument("--start-date", default="2019-01-01",
                        help="目標日範圍起始（含）")
    parser.add_argument("--end-date",   default="2025-12-31",
                        help="目標日範圍結束（含）")
    parser.add_argument("--corr-window", type=int, default=DEFAULT_CORR_WINDOW)
    parser.add_argument("--corr-threshold", type=float,
                        default=DEFAULT_CORR_THRESHOLD)
    parser.add_argument("--adr-dir", default="data/features/adr")
    parser.add_argument("--tw-dir",  default="data/features/tw")
    parser.add_argument("--out-dir", default="data/graphs/snapshots")
    args = parser.parse_args()

    pair_map = _load_pair_map_from_config()
    print(f"\n載入 {len(pair_map)} 組配對：{list(pair_map.items())}")

    gb = MultiplexGraphBuilder(
        pair_map=pair_map,
        adr_dir=args.adr_dir,
        tw_dir=args.tw_dir,
        corr_window=args.corr_window,
        corr_threshold=args.corr_threshold,
    )

    all_meta = gb.build_sequence(
        pd.Timestamp(args.start_date),
        pd.Timestamp(args.end_date),
        out_dir=args.out_dir,
    )

    # 摘要
    n_total   = len(all_meta)
    n_success = sum(1 for m in all_meta if m.n_l1_nodes > 0)
    n_skip    = n_total - n_success
    n_warned  = sum(1 for m in all_meta if m.warnings)
    avg_l1_e  = np.mean([m.n_l1_edges for m in all_meta if m.n_l1_nodes > 0]) if n_success else 0
    avg_l2_e  = np.mean([m.n_l2_edges for m in all_meta if m.n_l1_nodes > 0]) if n_success else 0

    print(f"\n摘要：")
    print(f"  目標日總數：{n_total}")
    print(f"  成功快照：  {n_success}")
    print(f"  跳過：      {n_skip}")
    print(f"  含警告：    {n_warned}")
    print(f"  平均 L1 邊數：{avg_l1_e:.1f}")
    print(f"  平均 L2 邊數：{avg_l2_e:.1f}")
    print(f"\n全部完成。")


if __name__ == "__main__":
    main()