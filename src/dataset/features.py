"""
features.py — 節點技術指標特徵工程（v2.0）
==============================================
承接 pipeline.py（已清洗完畢的 OHLCV + log_return + 追溯欄位），
計算 7 個技術指標，連同 pipeline 已算的 log_return 與 log_volume_z
共組成 9 維特徵向量。

對應政策：data_imputation_policy.md v1.0 §8.2
  「測試期樣本禁止包含任何 is_imputed=True 的列」

主要設計原則：
  ① 單向資料流：data/processed/ → features.py → data/features/，不覆寫上游
  ② 追溯欄位整合：is_long_gap=True 列的所有技術指標設為 NaN
  ③ Sentiment 不在此處理：個股技術指標對 VIX / 匯率 / 大盤指數無經濟意義
                          其特徵工程於 graph_builder.py 階段以全域外生變數方式處理

執行順序：pipeline.py → features.py → graph_builder.py

─────────────────────────────────────────────
L1 ADR 與 L2 台股共用 9 維特徵
─────────────────────────────────────────────
    log_return      對數日報酬率       （pipeline 已算）
    RSI_14          14日相對強弱指數
    MACD            MACD 線（EMA12 − EMA26）
    MACD_signal     訊號線（MACD 的 9 日 EMA）
    MACD_hist       MACD 柱（MACD − signal）
    BB_pos          布林帶位置（0=下軌, 1=上軌）
    MA5_dev         收盤價對 5 日均線偏離率
    MA20_dev        收盤價對 20 日均線偏離率
    log_volume_z    成交量滾動標準化   （pipeline 已算）

─────────────────────────────────────────────
執行方式：
    python features.py

作為模組：
    from features import FeatureBuilder
    fb = FeatureBuilder()
    df, report = fb.run("data/processed/adr/TSM.csv",
                        "data/features/adr/TSM.csv")
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple


# ════════════════════════════════════════════════════════════
# 常數
# ════════════════════════════════════════════════════════════

# L1 ADR 與 L2 台股共用的 9 個技術指標特徵（順序即 LSTM 輸入維度順序）
# 注意：MA{n}_dev 的 n 固定為 5 / 20（與 FeatureBuilder 內部參數綁定）
TECH_FEATURE_COLS = [
    "log_return",    # 1：對數日報酬率（pipeline 已算）
    "RSI_14",        # 2：相對強弱指數
    "MACD",          # 3：MACD 線
    "MACD_signal",   # 4：訊號線
    "MACD_hist",     # 5：MACD 柱狀圖
    "BB_pos",        # 6：布林帶位置
    "MA5_dev",       # 7：5 日均線偏離率
    "MA20_dev",      # 8：20 日均線偏離率
    "log_volume_z",  # 9：成交量滾動標準化（pipeline 已算）
]

# L1 / L2 目前均使用相同的 9 維技術指標
L1_FEATURE_COLS = TECH_FEATURE_COLS
L2_FEATURE_COLS = TECH_FEATURE_COLS

# pipeline.py 應產出的追溯欄位（政策 §8.1）
TRACE_COLS = ["is_imputed", "gap_length", "imputation_source_date", "is_long_gap"]

# 輸入必備欄位（缺一不可）
REQUIRED_INPUT_COLS = ["Close", "log_return"]

# 暖機期天數：取以下三者的最大值，前 60 列前任一特徵可能 NaN
#   - MACD 慢線(26) + 訊號線(9) = 35
#   - 布林帶(20) + std 暖機 ≈ 20
#   - pipeline.py 的 log_volume_z 滾動視窗（預設 60）  ← 最大
# 此值用於統計暖機期 NaN 數，並在訓練時剔除前 60 列。
WARMUP_DAYS = 60


# ════════════════════════════════════════════════════════════
# 報告容器
# ════════════════════════════════════════════════════════════

@dataclass
class FeatureReport:
    """單一標的的特徵工程報告。"""
    ticker:           str
    total_rows:       int = 0
    feature_count:    int = 0     # 實際產出的特徵數（應為 9）

    # NaN 來源分類
    warmup_nan_rows:   int = 0    # 暖機期 NaN（前 35 列，正常）
    long_gap_nan_rows: int = 0    # 因 is_long_gap=True 強制 NaN
    other_nan_rows:    int = 0    # 其他原因（應為 0，>0 代表異常）

    # 補值統計（讀自 pipeline 追溯欄位）
    imputed_rows:     int = 0
    long_gap_rows:    int = 0

    # 警告訊息
    warnings:         List[str] = field(default_factory=list)

    def print_summary(self):
        flag = "✅" if not self.warnings else "⚠️"
        print(
            f"  {flag} [{self.ticker}]  rows={self.total_rows}  "
            f"features={self.feature_count}/9  "
            f"warmup_NaN={self.warmup_nan_rows}  "
            f"long_gap_NaN={self.long_gap_nan_rows}  "
            f"imputed_rows={self.imputed_rows}  "
            f"long_gap_rows={self.long_gap_rows}"
        )
        for w in self.warnings:
            print(f"      ⚠ {w}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["warnings"] = "; ".join(d["warnings"])
        return d


# ════════════════════════════════════════════════════════════
# 主類別
# ════════════════════════════════════════════════════════════

class FeatureBuilder:
    """
    節點技術指標特徵工程器。

    讀取 pipeline.py 輸出的 CSV，計算 7 個技術指標（RSI / MACD×3 /
    BB_pos / MA5_dev / MA20_dev），連同 pipeline 已算的 log_return
    與 log_volume_z 共 9 維，輸出至獨立的 features 目錄。

    L1 ADR 和 L2 台股使用完全相同的特徵規格與計算邏輯。
    Sentiment 資料不在此處理。

    Parameters
    ----------
    rsi_period  : RSI 計算週期（預設 14）
    macd_fast   : MACD 快線 EMA span（預設 12）
    macd_slow   : MACD 慢線 EMA span（預設 26）
    macd_signal : MACD 訊號線 EMA span（預設 9）
    bb_window   : 布林帶滾動視窗（預設 20）

    Notes
    -----
    短期 / 長期均線固定為 5 日與 20 日（欄位名 MA5_dev / MA20_dev），
    若需更動，請同步修改 TECH_FEATURE_COLS。
    """

    def __init__(self,
                 rsi_period:  int = 14,
                 macd_fast:   int = 12,
                 macd_slow:   int = 26,
                 macd_signal: int = 9,
                 bb_window:   int = 20):
        self.rsi_period  = rsi_period
        self.macd_fast   = macd_fast
        self.macd_slow   = macd_slow
        self.macd_signal = macd_signal
        self.bb_window   = bb_window
        # 短期 / 長期均線固定（與 TECH_FEATURE_COLS 中的欄位名綁定）
        self.ma_short    = 5
        self.ma_long     = 20

    # ── 公開介面 ─────────────────────────────────────────────

    def run(self,
            in_path:  str,
            out_path: str,
            ticker:   str = None,
            ) -> Tuple[pd.DataFrame, FeatureReport]:
        """
        讀取一個 processed CSV → 計算技術指標 → 寫出至 features 目錄。

        Parameters
        ----------
        in_path  : data/processed/{market}/{ticker}.csv
        out_path : data/features/{market}/{ticker}.csv
        ticker   : 標的代碼（選填，未提供時從 in_path 檔名推斷）

        Returns
        -------
        df     : 含 9 維特徵的 DataFrame
        report : FeatureReport
        """
        if ticker is None:
            ticker = Path(in_path).stem

        df = pd.read_csv(in_path, index_col="Date", parse_dates=True)
        df = df.sort_index()

        report = FeatureReport(ticker=ticker, total_rows=len(df))

        # ── 輸入驗證 ─────────────────────────────────────────
        self._validate_input(df, report)

        # ── 計算技術指標 ──────────────────────────────────────
        df = self._calc_technical(df)

        # ── 整合追溯欄位（政策 §8.2）─────────────────────────
        df = self._propagate_traceability(df, report)

        # ── 統計報告 ─────────────────────────────────────────
        self._compute_report(df, report)

        # ── 寫出 ─────────────────────────────────────────────
        os.makedirs(Path(out_path).parent, exist_ok=True)
        df.to_csv(out_path)

        return df, report

    def run_batch(self,
                  in_dir:  str,
                  out_dir: str,
                  pattern: str = "*.csv") -> List[FeatureReport]:
        """
        批次處理目錄下所有 CSV，輸出至獨立目錄（不覆寫上游）。

        Parameters
        ----------
        in_dir  : data/processed/adr/  或 data/processed/tw/
        out_dir : data/features/adr/   或 data/features/tw/
        """
        files = sorted(glob.glob(os.path.join(in_dir, pattern)))
        files = [f for f in files if "report" not in Path(f).name]

        print(f"\n[FeatureBuilder] {in_dir} → {out_dir}（共 {len(files)} 個）")

        reports: List[FeatureReport] = []
        for in_path in files:
            ticker   = Path(in_path).stem
            out_path = os.path.join(out_dir, f"{ticker}.csv")
            try:
                _, report = self.run(in_path, out_path, ticker=ticker)
                reports.append(report)
                report.print_summary()
            except Exception as e:
                print(f"  ✗ [{ticker}]：{type(e).__name__}: {e}")

        # 寫批次報告
        if reports:
            report_path = os.path.join(out_dir, "feature_report.csv")
            pd.DataFrame([r.to_dict() for r in reports]).to_csv(
                report_path, index=False, encoding="utf-8-sig"
            )
            print(f"  報告 → {report_path}")

        return reports

    # ── 內部步驟（私有）──────────────────────────────────────

    def _validate_input(self, df: pd.DataFrame, report: FeatureReport):
        """檢查輸入 CSV 是否符合 pipeline.py 的產出規格。"""
        # 必要欄位
        missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"輸入 CSV 缺少必要欄位：{missing}。"
                f"請確認已執行 pipeline.py，目前欄位：{list(df.columns)}"
            )

        # 追溯欄位（缺漏時警告但不中斷，以兼容舊版 pipeline 產出）
        missing_trace = [c for c in TRACE_COLS if c not in df.columns]
        if missing_trace:
            report.warnings.append(
                f"缺少追溯欄位 {missing_trace}，已退回為「無補值」假設處理。"
                f"建議重跑最新版 pipeline.py。"
            )

        # 資料量過少警告
        if len(df) < WARMUP_DAYS * 2:
            report.warnings.append(
                f"資料量 {len(df)} 列偏少（建議 ≥ {WARMUP_DAYS * 2}），"
                f"暖機期後可用樣本可能不足"
            )

    def _calc_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算 7 個技術指標（不重複計算 pipeline 已算的 2 個欄位）。

        pipeline 已算：log_return（#1）、log_volume_z（#9）
        此處新增：RSI_14（#2）、MACD（#3）、MACD_signal（#4）、
                  MACD_hist（#5）、BB_pos（#6）、MA5_dev（#7）、MA20_dev（#8）
        """
        close = df["Close"]
        df    = df.copy()

        # ── #2 RSI（14 日）
        # 範圍 [0, 100]，需在訓練前正規化（除以 100 或 z-score）
        # 與 log_return 高度相關（同為 gain/loss 衍生），冗餘性需 ablation 確認
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(
                    com=self.rsi_period - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(
                    com=self.rsi_period - 1, adjust=False).mean()
        df["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        # ── #3 #4 #5 MACD（12, 26, 9）
        # MACD 線    = EMA12 − EMA26（中期動能）
        # 訊號線     = MACD 的 EMA9（觸發線）
        # 柱狀圖     = MACD − 訊號線（短期動能加速度）
        ema_fast          = close.ewm(span=self.macd_fast,   adjust=False).mean()
        ema_slow          = close.ewm(span=self.macd_slow,   adjust=False).mean()
        df["MACD"]        = ema_fast - ema_slow
        df["MACD_signal"] = df["MACD"].ewm(
                                span=self.macd_signal, adjust=False).mean()
        df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

        # ── #6 布林帶位置（BB_pos）
        # 公式：(Close − 下軌) / (上軌 − 下軌)
        # 0 = 下軌；1 = 上軌；> 1 / < 0 = 突破布林帶（極端訊號）
        ma_bb  = close.rolling(self.bb_window).mean()
        std_bb = close.rolling(self.bb_window).std()
        upper  = ma_bb + 2 * std_bb
        lower  = ma_bb - 2 * std_bb
        df["BB_pos"] = (close - lower) / (upper - lower).replace(0, np.nan)

        # ── #7 #8 均線偏離率（固定 5 / 20 日）
        # 公式：Close / MA − 1
        # 正值 → 股價在均線之上；負值 → 在均線之下
        df["MA5_dev"]  = close / close.rolling(self.ma_short).mean() - 1
        df["MA20_dev"] = close / close.rolling(self.ma_long).mean()  - 1

        return df

    def _propagate_traceability(self,
                                df:     pd.DataFrame,
                                report: FeatureReport) -> pd.DataFrame:
        """
        將 pipeline 的追溯欄位影響擴散到技術指標：
          ① is_long_gap=True：該列 Close 為 NaN，技術指標自然為 NaN，
                              此處做防禦性確認，避免 ewm 等操作意外產生非 NaN
          ② is_imputed=True：Close 為 ffill 值，log_return=0，
                              技術指標可計算但訊號失真。
                              不強制設 NaN（保留訓練彈性），僅統計後供下游剔除。

        對應政策 §8.2：測試期樣本禁止包含 is_imputed=True 的列
        （此規則由 graph_builder / dataset 層執行，features.py 僅統計與標記）
        """
        # 兼容缺少追溯欄位的舊版 pipeline 產出
        if "is_imputed" in df.columns:
            report.imputed_rows = int(df["is_imputed"].fillna(False).sum())
        if "is_long_gap" in df.columns:
            report.long_gap_rows = int(df["is_long_gap"].fillna(False).sum())

        # 防禦性：強制將 long_gap 列的所有技術特徵設為 NaN
        # （Close 本就是 NaN 會自動傳遞，但 ewm 在某些邊界可能殘留前值，故顯式覆蓋）
        if "is_long_gap" in df.columns:
            mask = df["is_long_gap"].fillna(False).astype(bool)
            tech_only = [c for c in TECH_FEATURE_COLS if c in df.columns]
            df.loc[mask, tech_only] = np.nan

        return df

    def _compute_report(self, df: pd.DataFrame, report: FeatureReport):
        """統計各類 NaN 的來源，生成最終報告。"""
        feat_cols = [c for c in TECH_FEATURE_COLS if c in df.columns]
        report.feature_count = len(feat_cols)

        if report.feature_count < 9:
            missing = [c for c in TECH_FEATURE_COLS if c not in df.columns]
            report.warnings.append(f"特徵欄位不齊：缺 {missing}")

        # 暖機期 NaN：前 WARMUP_DAYS 列任一特徵為 NaN
        warmup_idx = df.index[:WARMUP_DAYS]
        report.warmup_nan_rows = int(
            df.loc[warmup_idx, feat_cols].isna().any(axis=1).sum()
        )

        # long_gap NaN：is_long_gap=True 的列
        if "is_long_gap" in df.columns:
            mask = df["is_long_gap"].fillna(False).astype(bool)
            report.long_gap_nan_rows = int(mask.sum())

        # 其他 NaN（暖機期之後、且非 long_gap 的列卻仍有 NaN，這代表異常）
        post_warmup = df.iloc[WARMUP_DAYS:]
        if "is_long_gap" in post_warmup.columns:
            non_gap_post = post_warmup[
                ~post_warmup["is_long_gap"].fillna(False).astype(bool)
            ]
        else:
            non_gap_post = post_warmup
        other_nan = int(non_gap_post[feat_cols].isna().any(axis=1).sum())
        report.other_nan_rows = other_nan

        if other_nan > 0:
            report.warnings.append(
                f"暖機期之後仍有 {other_nan} 列出現 NaN（非 long_gap 造成），"
                f"請檢查 Close 序列是否有未標記的缺漏"
            )


# ════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════

def main():
    fb = FeatureBuilder()

    # ─── 重要：sentiment 不在此處理 ───────────────────────────
    # 個股技術指標（RSI / MACD / BB / MA）對 VIX、匯率、大盤指數
    # 無經濟意義。Sentiment 的特徵工程留待 graph_builder.py 階段，
    # 以「全域外生變數」方式處理（每個節點共享同一份 sentiment 序列）。
    # 如需獨立的 sentiment 預處理，請建立 sentiment_pipeline.py。
    market_pairs = [
        ("data/processed/adr", "data/features/adr"),
        ("data/processed/tw",  "data/features/tw"),
    ]

    all_reports: List[FeatureReport] = []
    for in_dir, out_dir in market_pairs:
        if not os.path.isdir(in_dir):
            print(f"跳過（目錄不存在）：{in_dir}")
            continue
        reports = fb.run_batch(in_dir, out_dir)
        all_reports.extend(reports)

    # ── 輸出特徵欄位驗證（抽樣 ADR 與 TW 各一支）──────────────
    print("\n" + "=" * 70)
    print("特徵欄位驗證（抽樣）")
    print("=" * 70)
    for csv_path, label in [
        ("data/features/adr/ASUUY.csv",   "ASUUY（華碩 ADR）"),
        ("data/features/tw/2357.csv",   "2357（華碩）"),
    ]:
        if not os.path.exists(csv_path):
            continue
        df   = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        cols = [c for c in TECH_FEATURE_COLS if c in df.columns]
        print(f"\n--- {label}  特徵數={len(cols)}/9 ---")
        nan_summary = df[cols].isna().sum().rename("NaN數")
        print(nan_summary.to_frame().T.to_string())

    # ── 整體警告匯總 ─────────────────────────────────────────
    warned = [r for r in all_reports if r.warnings]
    if warned:
        print("\n" + "=" * 70)
        print(f"⚠️ 共 {len(warned)} 個標的有警告")
        print("=" * 70)
        for r in warned:
            print(f"  [{r.ticker}]")
            for w in r.warnings:
                print(f"    · {w}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()