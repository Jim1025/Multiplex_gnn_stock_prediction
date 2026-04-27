"""
pipeline.py — OHLCV 資料前處理管線（v2.0：符合 data_imputation_policy.md）
============================================================================
整合缺漏值填補與資料清洗為單一腳本，遵循「分級補值政策」。

符合政策原則：
  ① log_return 絕不直接 ffill；只對 Close 補值後「重新計算」log_return
  ② 用 NYSE / TWSE 官方交易日曆 reindex，區分結構性 vs 技術性缺漏
  ③ 每筆列均可追溯：is_imputed / gap_length / imputation_source_date
  ④ 長缺口樣本以 is_long_gap 旗標標記，供下游訓練依政策 §8.2 剔除

完整流程：
    data/raw/{market}/{ticker}.csv
        │
        ├─ Phase 0：交易日曆對齊（Level 0）
        │     依 market 對應 NYSE(ADR) / TWSE(TW) 交易日曆 reindex
        │     區分 L-A 結構性缺漏（非交易日，不補）與 L-C 技術性缺漏（應補）
        │
        ├─ Phase 1：分級補值（僅針對 L-C 技術性缺漏）
        │     缺口 ≤ 2 日  → Close ffill（標準補值）
        │     缺口 3-5 日  → Close ffill + is_imputed 旗標（降權訓練）
        │     缺口 > 5 日  → 不補值，標記 is_long_gap（整段樣本應剔除）
        │     Volume       → 填 0
        │
        └─ Phase 2：資料清洗
              Step 1  異常值 IQR Clip + 報酬率 Z-score Clip
              Step 2  OHLC 內部一致性修正
              Step 3  ADF 平穩性檢驗
              Step 4  對數報酬率計算（由清洗後的 Close 重算）
              Step 5  Rolling Z-score 標準化
        │
        ▼
    data/processed/{market}/{ticker}.csv
      欄位：Open, High, Low, Close, Volume, log_return, log_volume,
            log_return_z, log_volume_z,
            is_imputed, gap_length, imputation_source_date, is_long_gap
    data/processed/{market}/pipeline_report.csv

政策依據：見 docs/data_imputation_policy.md v1.0

直接執行（處理全部標的）：
    python pipeline.py

作為模組（單支股票）：
    from pipeline import DataPipeline
    pipe = DataPipeline()
    df, report = pipe.run("data/raw/adr/TSM.csv", out_path="data/processed/adr/TSM.csv")
"""

import os
import glob
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from statsmodels.tsa.stattools import adfuller


# ════════════════════════════════════════════════════════════
# 常數（所有預設值統一在此調整）
# ════════════════════════════════════════════════════════════

PRICE_COLS   = ["Open", "High", "Low", "Close"]
VOLUME_COLS  = ["Volume"]

# Phase 1：分級補值門檻（對齊 data_imputation_policy.md §4）
SAFE_FFILL_LIMIT = 2    # 缺口 ≤ 此值 → 標準 ffill（不加旗標）
SOFT_FFILL_LIMIT = 5    # 缺口 ≤ 此值 → ffill 並標記 is_imputed（降權訓練）
                        # 缺口 > 此值 → 不補值，標記 is_long_gap

# 追溯欄位（政策 §8.1）
TRACE_COLS = ["is_imputed", "gap_length", "imputation_source_date", "is_long_gap"]

# 市場 → 交易日曆對應
MARKET_CALENDAR = {
    "adr":       "NYSE",
    "tw":        "XTAI",   # 台灣證交所
    "sentiment": None,     # 情緒資料不做日曆對齊
}

# Phase 2：清洗門檻
IQR_K        = 3.0  # IQR 倍數
ZSCORE_WIN   = 60   # 滾動 Z-score 視窗
ZSCORE_THR   = 5.0  # 報酬率異常門檻（標準差倍數）
ADF_ALPHA    = 0.05 # ADF 顯著水準


# ════════════════════════════════════════════════════════════
# 彙整報告（合併兩階段）
# ════════════════════════════════════════════════════════════

@dataclass
class PipelineReport:
    """
    單一標的的完整前處理報告，涵蓋填補與清洗兩個階段。

    對應 data_imputation_policy.md §9 驗收標準的欄位命名。
    """
    ticker:     str
    total_rows: int

    # ── Phase 0：交易日曆對齊 ──────────────────────────────────
    market_calendar:    str = ""     # 使用的交易日曆名稱（NYSE / XTAI / 無）
    n_calendar_days:    int = 0      # 應有的交易日總數
    n_raw_days:         int = 0      # 原始檔實際有的交易日
    n_tech_missing:     int = 0      # 技術性缺漏（應有但缺）

    # ── Phase 1：分級補值 ─────────────────────────────────────
    n_safe_ffill: Dict[str, int] = field(default_factory=dict)  # 缺口 ≤ 2 日
    n_soft_ffill: Dict[str, int] = field(default_factory=dict)  # 缺口 3-5 日（有 is_imputed）
    n_zero:       Dict[str, int] = field(default_factory=dict)  # Volume 填 0 筆數
    n_remain:     Dict[str, int] = field(default_factory=dict)  # 填補後殘餘 NaN
    long_gaps:    int            = 0                             # 超長缺口區段數
    n_long_gap_rows: int         = 0                             # 標記為 is_long_gap 的列數
    imputation_rate: float       = 0.0                           # 補值率（政策 §4 Level 0 檢查）

    # ── Phase 2：清洗 ──────────────────────────────────────
    outlier_price:    int   = 0
    outlier_ret:      int   = 0
    ohlc_violations:  int   = 0
    adf_close_p:      float = 1.0
    adf_logret_p:     float = 1.0
    logret_stationary: bool = True
    logret_std:       float = 0.0

    def print_summary(self):
        ok = "✅" if self.logret_stationary else "⚠️"
        sf_c  = self.n_safe_ffill.get("Close", 0)
        ft_c  = self.n_soft_ffill.get("Close", 0)
        remain_c = self.n_remain.get("Close", 0)
        # 政策 §4 Level 0：補值率過高警示
        warn = "❗" if self.imputation_rate > 0.05 else ""
        print(
            f"  {ok} [{self.ticker}]  rows={self.total_rows}  "
            f"cal={self.market_calendar}  "
            f"tech_miss={self.n_tech_missing}  "
            f"safe_ffill={sf_c}  soft_ffill={ft_c}  "
            f"long_gap_rows={self.n_long_gap_rows}  "
            f"imp_rate={self.imputation_rate:.3f}{warn}  "
            f"outlier_ret={self.outlier_ret}  "
            f"ADF_p={self.adf_logret_p:.4f}"
        )

    def to_dict(self) -> dict:
        return {
            "ticker":           self.ticker,
            "total_rows":       self.total_rows,
            # Phase 0
            "market_calendar":  self.market_calendar,
            "n_calendar_days":  self.n_calendar_days,
            "n_raw_days":       self.n_raw_days,
            "n_tech_missing":   self.n_tech_missing,
            # Phase 1
            "safe_ffill_Close": self.n_safe_ffill.get("Close", 0),
            "soft_ffill_Close": self.n_soft_ffill.get("Close", 0),
            "zero_Volume":      self.n_zero.get("Volume", 0),
            "remain_Close":     self.n_remain.get("Close", 0),
            "long_gaps":        self.long_gaps,
            "n_long_gap_rows":  self.n_long_gap_rows,
            "imputation_rate":  round(self.imputation_rate, 4),
            # Phase 2
            "outlier_price":    self.outlier_price,
            "outlier_ret":      self.outlier_ret,
            "ohlc_violations":  self.ohlc_violations,
            "adf_close_p":      round(self.adf_close_p, 4),
            "adf_logret_p":     round(self.adf_logret_p, 4),
            "logret_stationary": self.logret_stationary,
            "logret_std":       round(self.logret_std, 6),
        }


# ════════════════════════════════════════════════════════════
# 工具函數（兩階段共用）
# ════════════════════════════════════════════════════════════

def _find_gaps(series: pd.Series) -> List[Tuple[int, int, int]]:
    """找出所有連續 NaN 區段，回傳 [(start, end, length), ...]。"""
    gaps, in_gap, start = [], False, 0
    for i, is_na in enumerate(series.isna()):
        if is_na and not in_gap:
            in_gap, start = True, i
        elif not is_na and in_gap:
            in_gap = False
            gaps.append((start, i - 1, i - start))
    if in_gap:
        gaps.append((start, len(series) - 1, len(series) - start))
    return gaps


def _get_market_calendar(calendar_name: str,
                         start, end) -> pd.DatetimeIndex:
    """
    取得指定市場在 [start, end] 範圍的官方交易日。

    優先使用 pandas_market_calendars；若套件未安裝，退回 pd.bdate_range。

    效能注意事項
    -----------
    1. start / end 必須轉為 date（YYYY-MM-DD 字串）後再傳入 valid_days。
       直接傳 pd.Timestamp 會讓 CustomBusinessDay 走低效路徑，
       對 XTAI 等 holiday 表龐大的日曆會慢到無法接受（實測 1700 天耗時 30s+）。

    2. 例外只攔 ImportError 與 KeyError（找不到日曆），其他例外應拋出，
       避免靜默退回 bdate_range 而用錯誤資料繼續跑。
    """
    # 統一轉為 date 字串，這是效能關鍵
    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_str   = pd.Timestamp(end).strftime("%Y-%m-%d")

    try:
        import pandas_market_calendars as mcal
    except ImportError:
        # 套件未安裝：退回工作日（不精確，但可用）
        import warnings
        warnings.warn(
            "pandas_market_calendars 未安裝，退回 bdate_range（不含國定假日）。"
            "請執行：pip install pandas-market-calendars"
        )
        return pd.bdate_range(start=start_str, end=end_str)

    try:
        cal = mcal.get_calendar(calendar_name)
    except (KeyError, RuntimeError) as e:
        raise ValueError(
            f"找不到日曆 '{calendar_name}'。"
            f"請確認名稱（台灣證交所為 XTAI，非 TSX）。原始錯誤：{e}"
        )

    # 只攔型別 / 值錯誤，其他例外（含 KeyboardInterrupt）應正常拋出
    days = cal.valid_days(start_date=start_str, end_date=end_str)
    if days.tz is not None:
        days = days.tz_localize(None)
    return pd.DatetimeIndex(days)


def _apply_ffill_traceable(series: pd.Series,
                           gap_start: int,
                           gap_end: int,
                           index: pd.DatetimeIndex,
                           ) -> Tuple[pd.Series, int, pd.Timestamp]:
    """
    對 [gap_start, gap_end] 區段執行 forward fill，只用過去資料。

    回傳：(填補後序列, 填補筆數, 補值來源日期)

    關鍵：回傳「補值來源日期」供 imputation_source_date 欄位使用，
    確保政策 §8.1 的可追溯性。
    """
    s, cnt, prev_val, prev_date = series.copy(), 0, None, pd.NaT
    for i in range(gap_start - 1, -1, -1):
        if not pd.isna(s.iloc[i]):
            prev_val  = s.iloc[i]
            prev_date = index[i]
            break
    if prev_val is None:
        return s, 0, pd.NaT
    for i in range(gap_start, gap_end + 1):
        if pd.isna(s.iloc[i]):
            s.iloc[i] = prev_val
            cnt += 1
    return s, cnt, prev_date


# ════════════════════════════════════════════════════════════
# 主類別：DataPipeline
# ════════════════════════════════════════════════════════════

class DataPipeline:
    """
    OHLCV 資料前處理管線，整合缺漏值填補與資料清洗兩個階段。

    對應政策：data_imputation_policy.md v1.0

    Parameters
    ----------
    safe_ffill_limit : int    缺口 ≤ 此值使用標準 ffill（不加 is_imputed，預設 2）
    soft_ffill_limit : int    缺口 ≤ 此值使用 ffill 並標記 is_imputed（預設 5）
                              超過此值則完全不補，標記 is_long_gap
    iqr_k            : float  IQR 倍數門檻，超過視為價格異常（預設 3.0）
    zscore_win       : int    報酬率滾動 Z-score 視窗（預設 60）
    zscore_thr       : float  報酬率異常門檻（預設 5.0 個標準差）
    adf_alpha        : float  ADF 顯著水準（預設 0.05）
    """

    def __init__(self,
                 safe_ffill_limit: int   = SAFE_FFILL_LIMIT,
                 soft_ffill_limit: int   = SOFT_FFILL_LIMIT,
                 iqr_k:            float = IQR_K,
                 zscore_win:       int   = ZSCORE_WIN,
                 zscore_thr:       float = ZSCORE_THR,
                 adf_alpha:        float = ADF_ALPHA):
        self.safe_ffill_limit = safe_ffill_limit
        self.soft_ffill_limit = soft_ffill_limit
        self.iqr_k            = iqr_k
        self.zscore_win       = zscore_win
        self.zscore_thr       = zscore_thr
        self.adf_alpha        = adf_alpha

        # 日曆快取：(calendar_name, start_str, end_str) → DatetimeIndex
        # run_batch 處理同一個 market 下多支股票時可大幅省時
        self._calendar_cache: Dict[tuple, pd.DatetimeIndex] = {}

    def _cached_calendar(self, calendar_name: str,
                         start, end) -> pd.DatetimeIndex:
        """快取版的日曆查詢，避免同 market 下重複建構。"""
        key = (
            calendar_name,
            pd.Timestamp(start).strftime("%Y-%m-%d"),
            pd.Timestamp(end).strftime("%Y-%m-%d"),
        )
        if key not in self._calendar_cache:
            self._calendar_cache[key] = _get_market_calendar(
                calendar_name, start, end
            )
        return self._calendar_cache[key]

    # ── 公開介面 ─────────────────────────────────────────────

    def run(self,
            in_path:  str,
            out_path: str  = None,
            ticker:   str  = None,
            market:   str  = None,
            ) -> Tuple[pd.DataFrame, PipelineReport]:
        """
        讀取一個原始 OHLCV CSV，依序執行日曆對齊、分級補值與清洗。

        Parameters
        ----------
        in_path  : str  來源 CSV（data/raw/...）
        out_path : str  輸出 CSV；若為 None 則不寫檔（只回傳 df）
        ticker   : str  股票代碼（選填，未提供時從 in_path 檔名推斷）
        market   : str  市場代碼（adr / tw / sentiment）
                        未提供時從 in_path 的父目錄名推斷，用以決定交易日曆

        Returns
        -------
        df     : 清洗後的 DataFrame（含政策 §8.1 所列四個追溯欄位）
        report : PipelineReport
        """
        if ticker is None:
            ticker = Path(in_path).stem
        if market is None:
            market = Path(in_path).parent.name.lower()

        df = pd.read_csv(in_path, index_col="Date", parse_dates=True)
        df = df.sort_index()

        missing = [c for c in PRICE_COLS + VOLUME_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"[{ticker}] 缺少欄位：{missing}")

        report = PipelineReport(ticker=ticker, total_rows=len(df))

        # ── Phase 0：交易日曆對齊（區分結構性 vs 技術性缺漏） ─────
        df, report = self._phase0_calendar_align(df, report, market)

        # ── Phase 1：分級補值（僅針對技術性缺漏） ─────────────────
        df, report = self._phase1_impute(df, report)

        # ── Phase 2：清洗（含由 Close 重算 log_return） ────────────
        df, report = self._phase2_clean(df, report)

        # 更新最終 row 數與補值率
        report.total_rows = len(df)
        if len(df) > 0 and "is_imputed" in df.columns:
            report.imputation_rate = float(df["is_imputed"].mean())

        # ── 寫出 ───────────────────────────────────────────────
        if out_path:
            os.makedirs(Path(out_path).parent, exist_ok=True)
            df.to_csv(out_path)

        return df, report

    def run_batch(self,
                  in_dir:  str,
                  out_dir: str,
                  pattern: str = "*.csv",
                  ) -> pd.DataFrame:
        """
        批次處理 in_dir 下所有 CSV，結果寫至 out_dir。
        若 in_dir == out_dir，則原地覆寫（不建新目錄）。
        彙整報告寫至 out_dir/pipeline_report.csv。

        Parameters
        ----------
        in_dir  : 原始 CSV 目錄（data/raw/{market}/）
        out_dir : 輸出目錄（data/processed/{market}/）
        pattern : 篩選 glob，預設 *.csv
        """
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(in_dir, pattern)))
        files = [f for f in files if "report" not in Path(f).name]

        # 從 in_dir 推斷 market（例如 data/raw/adr → "adr"）
        market = Path(in_dir).name.lower()

        mode = "原地覆寫" if in_dir == out_dir else f"→ {out_dir}"
        print(f"\n[DataPipeline] {in_dir}  {mode}（market={market}，共 {len(files)} 個）")

        rows = []
        for in_path in files:
            ticker   = Path(in_path).stem
            out_path = os.path.join(out_dir, f"{ticker}.csv")
            try:
                _, report = self.run(in_path, out_path=out_path,
                                     ticker=ticker, market=market)
                rows.append(report.to_dict())
                report.print_summary()
            except Exception as e:
                print(f"  ✗ {ticker}：{e}")
                rows.append({"ticker": ticker, "error": str(e)})

        summary  = pd.DataFrame(rows)
        rpt_path = os.path.join(out_dir, "pipeline_report.csv")
        summary.to_csv(rpt_path, index=False, encoding="utf-8-sig")
        print(f"  報告 → {rpt_path}")
        return summary

    # ── Phase 0：交易日曆對齊（私有） ─────────────────────────

    def _phase0_calendar_align(self, df: pd.DataFrame,
                               report: PipelineReport,
                               market: str,
                               ) -> Tuple[pd.DataFrame, PipelineReport]:
        """
        依 market 對應的官方交易日曆 reindex，使「應有但缺」的日期顯現為 NaN。

        這一步是政策 §4 Level 0 的精神：先區分結構性 vs 技術性缺漏，
        再進入 Phase 1 決定是否補值。若 market 無對應日曆（如 sentiment），
        則跳過 reindex，僅初始化追溯欄位。
        """
        report.n_raw_days = len(df)

        calendar_name = MARKET_CALENDAR.get(market)
        report.market_calendar = calendar_name or "none"

        if calendar_name is None:
            # 無日曆對應：不做 reindex，直接初始化追溯欄位
            report.n_calendar_days = len(df)
            report.n_tech_missing  = int(df[PRICE_COLS].isna().any(axis=1).sum())
        else:
            # 取得官方交易日（使用快取，同 market 下不重複建構）
            cal_days = self._cached_calendar(
                calendar_name, df.index[0], df.index[-1]
            )
            df = df.reindex(cal_days)
            df.index.name = "Date"
            report.n_calendar_days = len(cal_days)
            report.n_tech_missing  = int(df["Close"].isna().sum())

        # 初始化追溯欄位（政策 §8.1）
        df["is_imputed"]              = False
        df["gap_length"]              = 0
        df["imputation_source_date"]  = pd.NaT
        df["is_long_gap"]             = False

        return df, report

    # ── Phase 1：分級補值（私有） ─────────────────────────────

    def _phase1_impute(self, df: pd.DataFrame,
                       report: PipelineReport,
                       ) -> Tuple[pd.DataFrame, PipelineReport]:
        """
        分級補值 OHLCV 缺漏值，符合 data_imputation_policy.md §4。

        規則：
          ① 缺口 ≤ safe_ffill_limit（預設 2 日）：Close ffill（不加 is_imputed）
          ② 缺口 ≤ soft_ffill_limit（預設 5 日）：Close ffill，設 is_imputed=True
          ③ 缺口 > soft_ffill_limit：不補值，設 is_long_gap=True，供下游剔除
          ④ Volume：一律填 0（不影響 log_return 計算）

        關鍵：OHLC 四欄以 Close 的缺口為基準統一處理，避免同一日 OHLC 不同步。
              log_return 不在此階段計算，而是在 Phase 2 Step 4 由清洗後的 Close 重算。
        """
        # 以 Close 的缺口作為整列的缺漏判準（OHLC 應同步）
        close_series = df["Close"].copy()
        gaps = _find_gaps(close_series)

        for col in PRICE_COLS:
            report.n_safe_ffill[col] = 0
            report.n_soft_ffill[col] = 0

        for gap_start, gap_end, gap_len in gaps:
            # 以 Close 決定策略，套用到全部 OHLC
            if gap_len <= self.safe_ffill_limit:
                mode = "safe"
            elif gap_len <= self.soft_ffill_limit:
                mode = "soft"
            else:
                mode = "long"

            # 記錄該缺口區段的所有列的 gap_length
            for i in range(gap_start, gap_end + 1):
                df.iloc[i, df.columns.get_loc("gap_length")] = gap_len

            if mode == "long":
                # 長缺口：不補值，標記 is_long_gap
                report.long_gaps += 1
                for i in range(gap_start, gap_end + 1):
                    df.iloc[i, df.columns.get_loc("is_long_gap")] = True
                continue

            # safe / soft：對 OHLC 四欄做 ffill，共用同一來源日
            source_date = pd.NaT
            for col in PRICE_COLS:
                s = df[col]
                s, cnt, src = _apply_ffill_traceable(
                    s, gap_start, gap_end, df.index
                )
                df[col] = s
                if mode == "safe":
                    report.n_safe_ffill[col] += cnt
                else:
                    report.n_soft_ffill[col] += cnt
                if col == "Close":
                    source_date = src

            # 寫入追溯欄位
            for i in range(gap_start, gap_end + 1):
                df.iloc[i, df.columns.get_loc("imputation_source_date")] = source_date
                if mode == "soft":
                    df.iloc[i, df.columns.get_loc("is_imputed")] = True

        # 統計 is_long_gap 列數
        report.n_long_gap_rows = int(df["is_long_gap"].sum())

        # Volume：一律填 0（缺漏時合理假設為無交易量）
        for col in VOLUME_COLS:
            cnt = int(df[col].isna().sum())
            report.n_zero[col] = cnt
            df[col] = df[col].fillna(0)

        # 記錄填補後殘餘 NaN（長缺口保留為 NaN 是正常的）
        for col in PRICE_COLS + VOLUME_COLS:
            report.n_remain[col] = int(df[col].isna().sum())

        return df, report

    # ── Phase 2：清洗（私有） ─────────────────────────────────

    def _phase2_clean(self, df: pd.DataFrame,
                      report: PipelineReport,
                      ) -> Tuple[pd.DataFrame, PipelineReport]:
        """依序執行五個清洗步驟。"""
        df, report = self._step1_outliers(df, report)
        df, report = self._step2_ohlc_consistency(df, report)
        report     = self._step3_stationarity(df, report)
        df         = self._step4_log_return(df)
        df, report = self._step5_normalize(df, report)
        return df, report

    def _step1_outliers(self, df, report):
        df = df.copy()

        # 1a：價格 IQR Clip
        cnt = 0
        for col in PRICE_COLS:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - self.iqr_k * iqr, q3 + self.iqr_k * iqr
            cnt   += int(((df[col] < lo) | (df[col] > hi)).sum())
            df[col] = df[col].clip(lower=lo, upper=hi)
        report.outlier_price = cnt

        # 1b：成交量 IQR Clip
        if "Volume" in df.columns:
            v = df["Volume"].replace(0, np.nan).dropna()
            if len(v) >= 4:
                q1, q3 = v.quantile(0.25), v.quantile(0.75)
                hi     = q3 + self.iqr_k * (q3 - q1)
                df["Volume"] = df["Volume"].clip(upper=hi)

        # 1c：報酬率滾動 Z-score — 極端跳空換成前後均值
        simple_ret = df["Close"].pct_change()
        rm  = simple_ret.rolling(self.zscore_win, min_periods=60).mean()
        rs  = simple_ret.rolling(self.zscore_win, min_periods=60).std()
        z   = (simple_ret - rm) / rs.replace(0, np.nan)
        ext = z.abs() > self.zscore_thr
        report.outlier_ret = int(ext.sum())
        for idx in df.index[ext]:
            pos = df.index.get_loc(idx)
            nb  = []
            if pos > 0 and not pd.isna(df["Close"].iloc[pos - 1]):
                nb.append(df["Close"].iloc[pos - 1])
            if pos < len(df) - 1 and not pd.isna(df["Close"].iloc[pos + 1]):
                nb.append(df["Close"].iloc[pos + 1])
            if nb:
                df.loc[idx, "Close"] = float(np.mean(nb))

        return df, report

    def _step2_ohlc_consistency(self, df, report):
        df, cnt = df.copy(), 0
        for i, row in df.iterrows():
            o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
            if any(pd.isna([o, h, l, c])):
                continue
            if h < max(o, c) or l > min(o, c) or h < l:
                df.loc[i, "High"] = max(o, h, l, c)
                df.loc[i, "Low"]  = min(o, h, l, c)
                cnt += 1
        report.ohlc_violations = cnt
        return df, report

    def _step3_stationarity(self, df, report):
        close = df["Close"].dropna()
        if len(close) < 20:
            return report
        try:
            r = adfuller(close, autolag="AIC")
            report.adf_close_p = float(r[1])
        except Exception:
            pass
        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) >= 20:
            try:
                r = adfuller(log_ret, autolag="AIC")
                report.adf_logret_p     = float(r[1])
                report.logret_stationary = r[1] < self.adf_alpha
            except Exception:
                pass
        return report

    def _step4_log_return(self, df):
        """
        由 Close 重新計算 log_return（政策原則 1：絕不對 log_return 直接 ffill）。

        is_long_gap 列的 Close 為 NaN，log_return 自然為 NaN；
        is_imputed 列的 Close 被 ffill 過，重算後的 log_return 會是 0
        （因為 C_t = C_{t-1}），這正是政策附錄 A 說明的合理行為：
        「市場休市或無交易」的合理報酬為 0，而非前一日的報酬。
        """
        df = df.copy()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["log_volume"] = np.log1p(df["Volume"])
        return df

    def _step5_normalize(self, df, report):
        df = df.copy()
        for col in ["log_return", "log_volume"]:
            if col not in df.columns:
                continue
            rm = df[col].rolling(self.zscore_win, min_periods=60).mean()
            rs = df[col].rolling(self.zscore_win, min_periods=60).std()
            df[f"{col}_z"] = (df[col] - rm) / rs.replace(0, np.nan)
        lr = df["log_return"].dropna()
        if len(lr) > 0:
            report.logret_std = float(lr.std())
        return df, report


# ════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════

def main():
    pipe = DataPipeline()

    for in_dir, out_dir in [
        ("data/raw/adr", "data/processed/adr"),
        ("data/raw/tw",  "data/processed/tw"),
        ("data/raw/sentiment",  "data/processed/sentiment"),
    ]:
        if not os.path.isdir(in_dir):
            print(f"跳過（目錄不存在）：{in_dir}")
            continue
        summary = pipe.run_batch(in_dir, out_dir)
        print(f"\n{'='*75}")
        cols = ["ticker", "n_calendar_days", "n_tech_missing",
                "safe_ffill_Close", "soft_ffill_Close", "n_long_gap_rows",
                "imputation_rate", "outlier_price", "outlier_ret",
                "ohlc_violations", "adf_logret_p", "logret_std"]
        print(summary[[c for c in cols if c in summary.columns]]
              .to_string(index=False))

    print("\n全部完成。")


if __name__ == "__main__":
    main()