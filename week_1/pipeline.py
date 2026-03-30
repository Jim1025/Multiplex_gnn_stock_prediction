"""
pipeline.py — OHLCV 資料前處理管線
=====================================
整合 impute.py（缺漏值填補）與 clean.py（資料清洗）為單一腳本。

完整流程：
    data/raw/{market}/{ticker}.csv
        │
        ├─ Phase 1：缺漏值填補（LayeredImputer）
        │     缺口 1-3 日  → Forward Fill
        │     缺口 4-5 日  → Rolling Mean
        │     缺口 > 5 日  → 保留 NaN
        │     Volume       → 填 0
        │
        └─ Phase 2：資料清洗（OHLCVCleaner）
              Step 1  異常值 IQR Clip + 報酬率 Z-score Clip
              Step 2  OHLC 內部一致性修正
              Step 3  ADF 平穩性檢驗
              Step 4  對數報酬率計算
              Step 5  Rolling Z-score 標準化
        │
        ▼
    data/processed/{market}/{ticker}.csv   ← 覆寫原始檔
    data/processed/{market}/pipeline_report.csv

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

# Phase 1：填補門檻
FFILL_LIMIT  = 3    # 缺口 ≤ 此值 → forward fill
ROLL_LIMIT   = 5    # 缺口 ≤ 此值 → rolling mean；超過 → 留 NaN
ROLL_WINDOW  = 5    # rolling mean 回顧視窗

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
    """
    ticker:     str
    total_rows: int

    # ── Phase 1：填補 ──────────────────────────────────────
    n_ffill:    Dict[str, int] = field(default_factory=dict)  # 各欄 ffill 筆數
    n_rolling:  Dict[str, int] = field(default_factory=dict)  # 各欄 rolling 筆數
    n_zero:     Dict[str, int] = field(default_factory=dict)  # Volume 填 0 筆數
    n_remain:   Dict[str, int] = field(default_factory=dict)  # 填補後殘餘 NaN
    long_gaps:  int            = 0                             # 超過 ROLL_LIMIT 的缺口數

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
        ffill_c  = self.n_ffill.get("Close", 0)
        roll_c   = self.n_rolling.get("Close", 0)
        remain_c = self.n_remain.get("Close", 0)
        print(
            f"  {ok} [{self.ticker}]  rows={self.total_rows}  "
            f"ffill={ffill_c}  roll={roll_c}  remain_NaN={remain_c}  "
            f"long_gaps={self.long_gaps}  "
            f"outlier_price={self.outlier_price}  outlier_ret={self.outlier_ret}  "
            f"ohlc_fix={self.ohlc_violations}  "
            f"ADF_p={self.adf_logret_p:.4f}  logret_std={self.logret_std:.4f}"
        )

    def to_dict(self) -> dict:
        return {
            "ticker":           self.ticker,
            "total_rows":       self.total_rows,
            # Phase 1
            "ffill_Close":      self.n_ffill.get("Close", 0),
            "roll_Close":       self.n_rolling.get("Close", 0),
            "zero_Volume":      self.n_zero.get("Volume", 0),
            "remain_Close":     self.n_remain.get("Close", 0),
            "long_gaps":        self.long_gaps,
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


def _apply_ffill(series: pd.Series,
                 gap_start: int, gap_end: int) -> Tuple[pd.Series, int]:
    """對 [gap_start, gap_end] 區段執行 forward fill，只用過去資料。"""
    s, cnt, prev = series.copy(), 0, None
    for i in range(gap_start - 1, -1, -1):
        if not pd.isna(s.iloc[i]):
            prev = s.iloc[i]
            break
    if prev is None:
        return s, 0
    for i in range(gap_start, gap_end + 1):
        if pd.isna(s.iloc[i]):
            s.iloc[i] = prev
            cnt += 1
    return s, cnt


def _apply_rolling_mean(series: pd.Series, gap_start: int,
                        gap_end: int, window: int) -> Tuple[pd.Series, int]:
    """用缺口前 window 個有效值的均值填補 [gap_start, gap_end]。"""
    s, cnt, past = series.copy(), 0, []
    for i in range(gap_start - 1, -1, -1):
        if not pd.isna(s.iloc[i]):
            past.append(s.iloc[i])
        if len(past) >= window:
            break
    if not past:
        return s, 0
    fill_val = float(np.mean(past))
    for i in range(gap_start, gap_end + 1):
        if pd.isna(s.iloc[i]):
            s.iloc[i] = fill_val
            cnt += 1
    return s, cnt


# ════════════════════════════════════════════════════════════
# 主類別：DataPipeline
# ════════════════════════════════════════════════════════════

class DataPipeline:
    """
    OHLCV 資料前處理管線，整合缺漏值填補與資料清洗兩個階段。

    Parameters
    ----------
    ffill_limit  : int    缺口 ≤ 此值（交易日）使用 forward fill（預設 3）
    roll_limit   : int    缺口 ≤ 此值使用 rolling mean；超過保留 NaN（預設 5）
    roll_window  : int    rolling mean 回顧視窗（預設 5）
    iqr_k        : float  IQR 倍數門檻，超過視為價格異常（預設 3.0）
    zscore_win   : int    報酬率滾動 Z-score 視窗（預設 60）
    zscore_thr   : float  報酬率異常門檻（預設 5.0 個標準差）
    adf_alpha    : float  ADF 顯著水準（預設 0.05）
    """

    def __init__(self,
                 ffill_limit: int   = FFILL_LIMIT,
                 roll_limit:  int   = ROLL_LIMIT,
                 roll_window: int   = ROLL_WINDOW,
                 iqr_k:       float = IQR_K,
                 zscore_win:  int   = ZSCORE_WIN,
                 zscore_thr:  float = ZSCORE_THR,
                 adf_alpha:   float = ADF_ALPHA):
        self.ffill_limit = ffill_limit
        self.roll_limit  = roll_limit
        self.roll_window = roll_window
        self.iqr_k       = iqr_k
        self.zscore_win  = zscore_win
        self.zscore_thr  = zscore_thr
        self.adf_alpha   = adf_alpha

    # ── 公開介面 ─────────────────────────────────────────────

    def run(self,
            in_path:  str,
            out_path: str  = None,
            ticker:   str  = None,
            ) -> Tuple[pd.DataFrame, PipelineReport]:
        """
        讀取一個原始 OHLCV CSV，依序執行填補與清洗，寫出結果。

        Parameters
        ----------
        in_path  : str  來源 CSV（data/raw/...）
        out_path : str  輸出 CSV；若為 None 則不寫檔（只回傳 df）
        ticker   : str  股票代碼（選填，未提供時從 in_path 檔名推斷）

        Returns
        -------
        df     : 清洗後的 DataFrame（含新增欄位）
        report : PipelineReport
        """
        if ticker is None:
            ticker = Path(in_path).stem

        df = pd.read_csv(in_path, index_col="Date", parse_dates=True)
        df = df.sort_index()

        missing = [c for c in PRICE_COLS + VOLUME_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"[{ticker}] 缺少欄位：{missing}")

        report = PipelineReport(ticker=ticker, total_rows=len(df))

        # ── Phase 1：填補 ──────────────────────────────────────
        df, report = self._phase1_impute(df, report)

        # ── Phase 2：清洗 ──────────────────────────────────────
        df, report = self._phase2_clean(df, report)

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

        mode = "原地覆寫" if in_dir == out_dir else f"→ {out_dir}"
        print(f"\n[DataPipeline] {in_dir}  {mode}（共 {len(files)} 個）")

        rows = []
        for in_path in files:
            ticker   = Path(in_path).stem
            out_path = os.path.join(out_dir, f"{ticker}.csv")
            try:
                _, report = self.run(in_path, out_path=out_path, ticker=ticker)
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

    # ── Phase 1：填補（私有） ─────────────────────────────────

    def _phase1_impute(self, df: pd.DataFrame,
                       report: PipelineReport,
                       ) -> Tuple[pd.DataFrame, PipelineReport]:
        """分層填補 OHLCV 缺漏值。"""

        # 價格欄：依缺口長度選策略
        for col in PRICE_COLS:
            s    = df[col].copy()
            gaps = _find_gaps(s)
            report.n_ffill[col]  = 0
            report.n_rolling[col] = 0

            for gap_start, gap_end, gap_len in gaps:
                if gap_len <= self.ffill_limit:
                    s, cnt = _apply_ffill(s, gap_start, gap_end)
                    report.n_ffill[col] += cnt
                elif gap_len <= self.roll_limit:
                    s, cnt = _apply_rolling_mean(
                        s, gap_start, gap_end, self.roll_window)
                    report.n_rolling[col] += cnt
                else:
                    report.long_gaps += 1   # 超過門檻，保留 NaN

            df[col] = s

        # Volume：一律填 0
        for col in VOLUME_COLS:
            cnt = int(df[col].isna().sum())
            report.n_zero[col] = cnt
            df[col] = df[col].fillna(0)

        # 記錄填補後殘餘 NaN
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
        cols = ["ticker", "ffill_Close", "roll_Close", "remain_Close",
                "long_gaps", "outlier_price", "outlier_ret",
                "ohlc_violations", "adf_logret_p", "logret_std"]
        print(summary[[c for c in cols if c in summary.columns]]
              .to_string(index=False))

    print("\n全部完成。")


if __name__ == "__main__":
    main()