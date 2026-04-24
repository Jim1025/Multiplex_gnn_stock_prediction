"""
features.py — 節點技術指標特徵工程
=====================================
承接 pipeline.py（清洗完畢的 OHLCV + log_return），
計算 7 個技術指標，連同 pipeline 已算的 log_return 與
log_volume_z 共組成 9 維特徵向量。

市場情緒特徵（VIX / SP500_ret / USDTWD_chg / ADR_premium /
TAIEX_ret / 融資融券 / 法人）暫時移除，待資料就緒後補回。

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
    df = fb.run("data/processed/adr/TSM.csv")
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path


# ════════════════════════════════════════════════════════════
# 常數
# ════════════════════════════════════════════════════════════

# L1 ADR 與 L2 台股共用的 9 個技術指標特徵（順序即 LSTM 輸入維度順序）
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


# ════════════════════════════════════════════════════════════
# 主類別
# ════════════════════════════════════════════════════════════

class FeatureBuilder:
    """
    節點技術指標特徵工程器。

    讀取 pipeline.py 輸出的 CSV，計算 7 個技術指標（RSI /
    MACD×3 / BB_pos / MA5_dev / MA20_dev），連同 pipeline 已
    算的 log_return 與 log_volume_z 共 9 維，覆寫回同一個 CSV。

    L1 ADR 和 L2 台股使用完全相同的特徵規格與計算邏輯。

    Parameters
    ----------
    rsi_period  : RSI 計算週期（預設 14）
    macd_fast   : MACD 快線 EMA span（預設 12）
    macd_slow   : MACD 慢線 EMA span（預設 26）
    macd_signal : MACD 訊號線 EMA span（預設 9）
    bb_window   : 布林帶滾動視窗（預設 20）
    ma_short    : 短期均線天數（預設 5）
    ma_long     : 長期均線天數（預設 20）
    """

    def __init__(self,
                 rsi_period:  int = 14,
                 macd_fast:   int = 12,
                 macd_slow:   int = 26,
                 macd_signal: int = 9,
                 bb_window:   int = 20,
                 ma_short:    int = 5,
                 ma_long:     int = 20):
        self.rsi_period  = rsi_period
        self.macd_fast   = macd_fast
        self.macd_slow   = macd_slow
        self.macd_signal = macd_signal
        self.bb_window   = bb_window
        self.ma_short    = ma_short
        self.ma_long     = ma_long

    # ── 公開介面 ─────────────────────────────────────────────

    def run(self, csv_path: str) -> pd.DataFrame:
        """
        讀取一個 processed CSV → 計算技術指標 → 覆寫回原檔。

        Parameters
        ----------
        csv_path : data/processed/{market}/{ticker}.csv
        """
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        df = df.sort_index()
        df = self._calc_technical(df)
        df.to_csv(csv_path)
        return df

    def run_batch(self, processed_dir: str) -> None:
        """
        批次處理目錄下所有 CSV，原地覆寫（跳過 *report*.csv）。

        Parameters
        ----------
        processed_dir : data/processed/adr/ 或 data/processed/tw/
        """
        files = sorted(glob.glob(os.path.join(processed_dir, "*.csv")))
        files = [f for f in files if "report" not in Path(f).name]

        print(f"\n特徵工程：{processed_dir}（共 {len(files)} 個）")

        for csv_path in files:
            ticker = Path(csv_path).stem
            try:
                df        = self.run(csv_path)
                feat_cols = [c for c in TECH_FEATURE_COLS if c in df.columns]
                nan_cnt   = df[feat_cols].isna().sum().sum()
                print(f"  ✅ [{ticker}]  特徵={len(feat_cols)}/9  "
                      f"NaN總計={nan_cnt}")
            except Exception as e:
                print(f"  ✗ [{ticker}]：{e}")

    # ── 技術指標計算（私有）──────────────────────────────────

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
        # 原理：gain / loss 的指數移動平均之比
        # RSI 接近 100 → 超買；接近 0 → 超賣；50 為中性
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(
                    com=self.rsi_period - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(
                    com=self.rsi_period - 1, adjust=False).mean()
        df["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        # ── #3 #4 #5 MACD（12, 26, 9）
        # MACD 線    = EMA12 − EMA26（動能方向）
        # 訊號線     = MACD 的 EMA9（觸發線）
        # 柱狀圖     = MACD − 訊號線（動能強弱）
        ema_fast          = close.ewm(span=self.macd_fast,   adjust=False).mean()
        ema_slow          = close.ewm(span=self.macd_slow,   adjust=False).mean()
        df["MACD"]        = ema_fast - ema_slow
        df["MACD_signal"] = df["MACD"].ewm(
                                span=self.macd_signal, adjust=False).mean()
        df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

        # ── #6 布林帶位置（BB_pos）
        # 公式：(Close − 下軌) / (上軌 − 下軌)
        # 0 = 在下軌；1 = 在上軌；>1 或 <0 = 突破布林帶
        ma20   = close.rolling(self.bb_window).mean()
        std20  = close.rolling(self.bb_window).std()
        upper  = ma20 + 2 * std20
        lower  = ma20 - 2 * std20
        df["BB_pos"] = (close - lower) / (upper - lower).replace(0, np.nan)

        # ── #7 #8 均線偏離率
        # 公式：Close / MA − 1
        # 正值 → 股價在均線之上；負值 → 在均線之下
        df["MA5_dev"]  = close / close.rolling(self.ma_short).mean() - 1
        df["MA20_dev"] = close / close.rolling(self.ma_long).mean()  - 1

        return df


# ════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════

def main():
    fb = FeatureBuilder()

    # L1 ADR 和 L2 台股使用同一套邏輯，直接批次處理
    for market_dir in ["data/processed/adr", "data/processed/tw", "data/processed/sentiment"]:
        if not os.path.isdir(market_dir):
            print(f"跳過（目錄不存在）：{market_dir}")
            continue
        fb.run_batch(market_dir)

    # 輸出特徵欄位驗證
    for csv_path, label in [
        ("data/processed/adr/TSM.csv", "TSM（ADR）"),
        ("data/processed/tw/2330.csv", "2330（台積電）"),
    ]:
        if not os.path.exists(csv_path):
            continue
        df   = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        cols = [c for c in TECH_FEATURE_COLS if c in df.columns]
        print(f"\n=== {label}  欄位={len(cols)}/9 ===")
        print(df[cols].isna().sum().rename("NaN數").to_frame().T.to_string())

    print("\n全部完成。")


if __name__ == "__main__":
    main()