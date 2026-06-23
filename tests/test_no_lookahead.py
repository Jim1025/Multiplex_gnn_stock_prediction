"""
test_no_lookahead.py — Look-ahead Bias 防護測試套件 v2.0
==========================================================
針對 7 組 ADR-台股配對的真實資料（features.py 處理後）逐一驗證
時序對齊正確性與資料品質。

測試邏輯：ADR(t) → TW(t+1)
    合法：用 ADR 在 t 日的收盤特徵，預測台股 t+1 日報酬
    違法：用任何 t+1 日（或更晚）的 ADR 資料當輸入（未來洩漏）

對應政策：data_imputation_policy.md v1.0 §5

執行方式：
    pytest tests/test_no_lookahead.py -v
    pytest tests/test_no_lookahead.py -v --tb=short   # 失敗時只顯示摘要
    python  tests/test_no_lookahead.py                # 不需 pytest 的快速診斷模式

═══════════════════════════════════════════════════════════════════════
測試清單（共 10 項）
═══════════════════════════════════════════════════════════════════════
第一組：時序對齊（pipeline + features 共同保護）
  T1  test_no_nan_after_alignment_excluding_long_gap
  T2  test_adr_t_equals_previous_day（三段式抽樣）
  T3  test_no_perfect_correlation
  T4  test_monotonic_date_index
  T5  test_minimum_common_trading_days（門檻 1400）
  T6  test_train_test_split_no_leakage

第二組：features 層追溯欄位防護（v2.0 新增）
  T7  test_no_consecutive_identical_log_returns（只偵測非零連續相同）
  T8  test_imputation_flag_exists
  T9  test_log_return_recomputation_consistency
  T10 test_no_future_imputation

第三組：graph_builder 圖層測試（v2.0 新增）
  T11 test_graph_window_no_lookahead（視窗不含 target_date）
  T12 test_a12_strictly_diagonal（A12 嚴格對角）
  T13 test_node_features_use_correct_date（節點特徵日期正確）
"""

import os
import sys
import glob
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List


# ════════════════════════════════════════════════════════════
# 設定：路徑與閾值
# ════════════════════════════════════════════════════════════

# 注意：測試讀取 data/features/，因為這是訓練的最終輸入
# pipeline 階段的中間產物由 pipeline 自己的測試保護
ADR_DIR  = "data/features/adr"
TW_DIR   = "data/features/tw"

# 閾值（與 alignment_audit.py 對齊）
MIN_COMMON_DAYS    = 1400   # T5：對齊後最少共同交易日數
CORR_UPPER_BOUND   = 0.999  # T3：adr_t 與 tw_t1 相關係數不可超過此值
LOG_RETURN_TOL     = 1e-10  # T9：log_return 重算誤差容忍
IDENTICAL_LR_RATIO = 0.01   # T7：連續 3 日相同 log_return 比例上限（排除 imputed）
SAMPLE_HEAD_TAIL   = 30     # T2：頭尾各取此筆數
SAMPLE_MIDDLE      = 50     # T2：中段隨機取此筆數

# 切分比例（T6 用，未來應改由 config/split.yaml 統一管理）
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


# ════════════════════════════════════════════════════════════
# PAIR_MAP 來源：優先從 src/dataset/config.py 引入
# ════════════════════════════════════════════════════════════

def _load_pair_map() -> dict:
    """
    從 src/dataset/config.py 載入 PAIR_MAP。

    支援兩種介面：
      ① get_pair_dict() 函式 → 回傳 {adr: tw} 純字典
      ② PAIR_MAP 字典      → 直接使用（值可能是字串或含 metadata 的 dict）
    """
    # 嘗試將專案根目錄加入 sys.path
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from src.dataset import config as cfg
    except ImportError as e:
        pytest.skip(f"無法載入 src.dataset.config：{e}。"
                    f"請確認 src/dataset/config.py 存在且包含 PAIR_MAP。")

    # 介面 1：get_pair_dict()
    if hasattr(cfg, "get_pair_dict") and callable(cfg.get_pair_dict):
        return cfg.get_pair_dict()

    # 介面 2：PAIR_MAP 屬性
    if hasattr(cfg, "PAIR_MAP"):
        pm = cfg.PAIR_MAP
        if not pm:
            pytest.fail("config.PAIR_MAP 為空字典")
        # 自動處理「值為 dict 含 metadata」的情況
        first_val = next(iter(pm.values()))
        if isinstance(first_val, dict):
            return {k: v["tw"] for k, v in pm.items()}
        return dict(pm)

    pytest.fail("config.py 未提供 get_pair_dict() 或 PAIR_MAP")


PAIR_MAP = _load_pair_map()


# ════════════════════════════════════════════════════════════
# 工具函數
# ════════════════════════════════════════════════════════════

def _adr_path(ticker: str) -> Path:
    return Path(ADR_DIR) / f"{ticker}.csv"


def _tw_path(code: str) -> Path:
    return Path(TW_DIR) / f"{code}.csv"


def load_pair(adr_ticker: str,
              tw_code:    str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """讀取一組 ADR-TW 配對的 features CSV。"""
    adr_df = pd.read_csv(_adr_path(adr_ticker),
                         index_col="Date", parse_dates=True).sort_index()
    tw_df  = pd.read_csv(_tw_path(tw_code),
                         index_col="Date", parse_dates=True).sort_index()
    return adr_df, tw_df


def align_adr_to_tw(adr_df: pd.DataFrame,
                    tw_df:  pd.DataFrame,
                    feature_col: str = "log_return") -> pd.DataFrame:
    """
    時序對齊邏輯：ADR(t) → TW(t+1)

    `adr_df[feature_col].shift(1)` 後與 `tw_df[feature_col]` inner join。
    結果 merged.loc[d, "adr_t"] = adr_df 中 d 之前最近一個有效日的值。
    """
    adr_shifted = adr_df[feature_col].shift(1).rename("adr_t")
    tw_col      = tw_df[feature_col].rename("tw_t1")
    return pd.concat([adr_shifted, tw_col], axis=1, join="inner").dropna()


def get_available_pairs() -> List[Tuple[str, str]]:
    """動態檢查實際可用的配對（兩端 CSV 都存在才會列入）。"""
    pairs = []
    for adr_ticker, tw_code in PAIR_MAP.items():
        if _adr_path(adr_ticker).exists() and _tw_path(tw_code).exists():
            pairs.append((adr_ticker, tw_code))
    return pairs


def _exclude_imputed(adr_df: pd.DataFrame) -> pd.DataFrame:
    """過濾掉 is_imputed 與 is_long_gap 為 True 的列。"""
    df = adr_df.copy()
    if "is_imputed" in df.columns:
        df = df[~df["is_imputed"].fillna(False).astype(bool)]
    if "is_long_gap" in df.columns:
        df = df[~df["is_long_gap"].fillna(False).astype(bool)]
    return df


# ════════════════════════════════════════════════════════════
# 第一組：時序對齊測試（T1–T6，原版改寫）
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_no_nan_after_alignment_excluding_long_gap(adr_ticker, tw_code):
    """
    T1：對齊後不應有 NaN（排除合法的 long_gap 列）

    原版只檢查 NaN 總數為 0，但新版 features.py 會合法產生 long_gap NaN，
    這些 NaN 是政策規範的正常結果，需排除後再檢查。
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)

    # 先剔除 long_gap 列再對齊
    adr_clean = adr_df.copy()
    tw_clean  = tw_df.copy()
    if "is_long_gap" in adr_clean.columns:
        adr_clean = adr_clean[~adr_clean["is_long_gap"].fillna(False).astype(bool)]
    if "is_long_gap" in tw_clean.columns:
        tw_clean = tw_clean[~tw_clean["is_long_gap"].fillna(False).astype(bool)]

    merged = align_adr_to_tw(adr_clean, tw_clean)

    nan_count = merged[["adr_t", "tw_t1"]].isna().sum().sum()
    assert nan_count == 0, (
        f"[{adr_ticker}→{tw_code}] 對齊後（排除 long_gap）仍有 {nan_count} 個 NaN。"
        f"請檢查 features.py 是否正確處理追溯欄位。"
    )


@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_adr_t_equals_previous_day(adr_ticker, tw_code):
    """
    T2：核心 Look-ahead Bias 測試（三段式抽樣）

    對 merged 中的每個日期 d，驗證：
        merged["adr_t"][d]  ==  adr_df["log_return"][d-1]

    抽樣策略：
      - 頭部 30 筆（涵蓋對齊起始邏輯）
      - 中段隨機 50 筆（涵蓋一般情況）
      - 尾部 30 筆（涵蓋資料尾端，最接近測試集）
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)
    merged = align_adr_to_tw(adr_df, tw_df)

    adr_log_ret = adr_df["log_return"].dropna()
    n_total = len(merged)

    if n_total < SAMPLE_HEAD_TAIL * 2 + SAMPLE_MIDDLE:
        pytest.skip(f"資料量 {n_total} 不足以做三段式抽樣")

    # 三段式抽樣
    rng = np.random.default_rng(seed=hash(f"{adr_ticker}_{tw_code}") % 2**32)
    head_idx = list(range(SAMPLE_HEAD_TAIL))
    tail_idx = list(range(n_total - SAMPLE_HEAD_TAIL, n_total))
    middle_pool = list(range(SAMPLE_HEAD_TAIL, n_total - SAMPLE_HEAD_TAIL))
    middle_idx = sorted(rng.choice(middle_pool,
                                   size=min(SAMPLE_MIDDLE, len(middle_pool)),
                                   replace=False).tolist())
    sample_idx = head_idx + middle_idx + tail_idx

    fail_msgs = []
    checked = 0
    for i in sample_idx:
        d = merged.index[i]
        if d not in adr_log_ret.index:
            continue   # 該日不在 ADR 索引中，跳過
        pos = adr_log_ret.index.get_loc(d)
        if pos == 0:
            continue   # 第一筆無「前一日」可比

        expected = float(adr_log_ret.iloc[pos - 1])
        actual   = float(merged.loc[d, "adr_t"])
        diff     = abs(actual - expected)

        if diff >= 1e-9:
            fail_msgs.append(
                f"  [{i}] {d.date()}: adr_t={actual:.6f} "
                f"≠ adr[{adr_log_ret.index[pos-1].date()}]={expected:.6f} "
                f"(差 {diff:.2e})"
            )
        checked += 1

    assert checked > 0, f"[{adr_ticker}→{tw_code}] 抽樣後無有效驗證樣本"
    assert not fail_msgs, (
        f"[{adr_ticker}→{tw_code}] Look-ahead Bias 偵測！\n"
        f"  共驗證 {checked} 筆，失敗 {len(fail_msgs)} 筆：\n"
        + "\n".join(fail_msgs[:5])
        + (f"\n  ... 另外還有 {len(fail_msgs)-5} 筆" if len(fail_msgs) > 5 else "")
    )


@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_no_perfect_correlation(adr_ticker, tw_code):
    """
    T3：adr_t 與 tw_t1 相關係數不可趨近 1

    若兩者幾乎完全相關，代表用了同一欄資料或時間對齊錯誤
    （例如 ADR 被錯誤對齊到台股同一天）。
    閾值 0.999 用於擋極端 bug，並非衡量真實相關性。
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)
    merged = align_adr_to_tw(adr_df, tw_df)

    if len(merged) < 30:
        pytest.skip(f"資料量 {len(merged)} 不足以計算相關係數")

    corr = merged["adr_t"].corr(merged["tw_t1"])
    assert abs(corr) < CORR_UPPER_BOUND, (
        f"[{adr_ticker}→{tw_code}] adr_t 與 tw_t1 相關係數 {corr:.4f} "
        f"超過 {CORR_UPPER_BOUND}，疑似資料洩漏（用了同一欄或對齊錯誤）"
    )


@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_monotonic_date_index(adr_ticker, tw_code):
    """
    T4：日期索引必須嚴格遞增、無重複、無亂序

    時序亂序本身就是 Look-ahead Bias 的成因之一
    （例如某天的資料被插入到更早的位置）。
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)

    for label, df in [("ADR", adr_df), ("TW", tw_df)]:
        idx = df.index
        assert idx.is_monotonic_increasing, \
            f"[{adr_ticker}→{tw_code}] {label} 日期索引非嚴格遞增"
        assert idx.is_unique, \
            f"[{adr_ticker}→{tw_code}] {label} 日期索引有重複值"


@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_minimum_common_trading_days(adr_ticker, tw_code):
    """
    T5：對齊後共同交易日數須 ≥ MIN_COMMON_DAYS（1400）

    與 alignment_audit.py 的閾值一致。若不足代表：
      - 兩市場行事曆差異過大
      - 資料缺漏嚴重
      - 下載時間範圍不一致
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)
    merged = align_adr_to_tw(adr_df, tw_df)

    assert len(merged) >= MIN_COMMON_DAYS, (
        f"[{adr_ticker}→{tw_code}] 對齊後僅 {len(merged)} 筆 "
        f"< {MIN_COMMON_DAYS} 筆，訓練樣本不足"
    )


@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_train_test_split_no_leakage(adr_ticker, tw_code):
    """
    T6：模擬 train/val/test 切分後驗證跨 split 無洩漏

    確認訓練集最後一筆所使用的 ADR 原始日期，
    必須嚴格早於測試集起始日。
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)
    merged = align_adr_to_tw(adr_df, tw_df)

    n = len(merged)
    if n < 100:
        pytest.skip(f"資料量 {n} 太少，跳過切分測試")

    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_end_date = merged.index[n_train - 1]
    test_start_date = merged.index[n_train + n_val]

    # 訓練集最後一筆的 adr_t 對應的「ADR 原始日期」
    adr_log_ret = adr_df["log_return"].dropna()
    if train_end_date not in adr_log_ret.index:
        pytest.skip("訓練集尾端日期不在 ADR 索引中")
    train_end_pos = adr_log_ret.index.get_loc(train_end_date)
    if train_end_pos == 0:
        pytest.skip("訓練集尾端是 ADR 序列首筆，無法回推來源")
    train_adr_source_date = adr_log_ret.index[train_end_pos - 1]

    assert train_adr_source_date < test_start_date, (
        f"[{adr_ticker}→{tw_code}] 跨 split 洩漏！\n"
        f"  訓練集最後一筆 (TW {train_end_date.date()}) 用到的 ADR 來源日期 "
        f"{train_adr_source_date.date()} ≥ 測試集起始日 {test_start_date.date()}"
    )


# ════════════════════════════════════════════════════════════
# 第二組：features 層追溯欄位防護（T7–T10，v2.0 新增）
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_no_consecutive_identical_log_returns(adr_ticker, tw_code):
    """
    T7：偵測「非零」連續 3 日相同的 log_return（污染指標）

    重要的金融意義區分：
      連續多日 log_return = 0（合法）：
         代表「平盤收盤」，即連續多日 Close 完全相同。
         在低波動股票（電信、傳產）與小型股常見，是真實市場行為。
         診斷顯示 2412/2317 等標的此現象 100% 為平盤收盤。

      [FAIL] 連續多日 log_return = 非零相同值（ffill 污染）：
         若 pipeline 對 log_return 直接 ffill 而非由 Close 重算，
         會產生「同一個非零數字延續多日」，違反政策原則 1。
         此情況在自然市場中機率近乎為零。

    排除：
      - is_imputed=True 的列（補值列的 log_return=0 是政策允許的）
      - is_long_gap=True 的列
      - log_return 為 0 的事件（合法的平盤收盤）
    """
    for label, ticker in [("ADR", adr_ticker), ("TW", tw_code)]:
        path = _adr_path(ticker) if label == "ADR" else _tw_path(ticker)
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()

        # 排除 imputed 與 long_gap 列
        clean = df.copy()
        if "is_imputed" in clean.columns:
            clean = clean[~clean["is_imputed"].fillna(False).astype(bool)]
        if "is_long_gap" in clean.columns:
            clean = clean[~clean["is_long_gap"].fillna(False).astype(bool)]

        lr = clean["log_return"].dropna()
        if len(lr) < 100:
            continue

        # 連續 3 筆完全相同
        same_with_prev1 = (lr.values[1:]   == lr.values[:-1])
        same_with_prev2 = (lr.values[2:]   == lr.values[1:-1])
        triple_match = same_with_prev1[1:] & same_with_prev2

        # ★ 關鍵修正：只計算「非零」的連續相同（排除合法平盤收盤）
        triple_values = lr.values[2:][triple_match]
        nonzero_triple = (triple_values != 0).sum()

        # 同時統計零值事件數，作為診斷資訊（不觸發失敗）
        zero_triple = (triple_values == 0).sum()

        ratio = nonzero_triple / max(len(lr) - 2, 1)
        assert ratio < IDENTICAL_LR_RATIO, (
            f"[{adr_ticker}→{tw_code}] {label} ({ticker}) 連續 3 日「非零」"
            f"相同 log_return 比例 {ratio:.2%} > {IDENTICAL_LR_RATIO:.0%}，"
            f"疑似 log_return 直接 ffill 污染。"
            f"\n  非零事件數：{nonzero_triple}（疑似污染）"
            f"\n  零值事件數：{zero_triple}（合法平盤收盤）"
        )


@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_imputation_flag_exists(adr_ticker, tw_code):
    """
    T8：確認 pipeline 產出的四個追溯欄位均存在

    對應政策 §8.1 的可追溯性要求。
    """
    REQUIRED_TRACE = ["is_imputed", "gap_length",
                      "imputation_source_date", "is_long_gap"]

    for label, ticker in [("ADR", adr_ticker), ("TW", tw_code)]:
        path = _adr_path(ticker) if label == "ADR" else _tw_path(ticker)
        df = pd.read_csv(path, index_col="Date", parse_dates=True)

        missing = [c for c in REQUIRED_TRACE if c not in df.columns]
        assert not missing, (
            f"[{adr_ticker}→{tw_code}] {label} ({ticker}) 缺少追溯欄位 {missing}。"
            f"請重跑最新版 pipeline.py。"
        )


@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_log_return_recomputation_consistency(adr_ticker, tw_code):
    """
    T9：log_return 必須由 Close 重新計算的結果一致

    驗證政策原則 1：log_return 不可獨立 ffill，
    必須來自 Close 的對數差分。
    """
    for label, ticker in [("ADR", adr_ticker), ("TW", tw_code)]:
        path = _adr_path(ticker) if label == "ADR" else _tw_path(ticker)
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()

        # 排除 long_gap 列（其 Close 為 NaN，無法重算）
        if "is_long_gap" in df.columns:
            mask = ~df["is_long_gap"].fillna(False).astype(bool)
            df_check = df[mask]
        else:
            df_check = df

        # 獨立重算 log_return = ln(C_t / C_{t-1})
        recomputed = np.log(df_check["Close"] / df_check["Close"].shift(1))

        # 比對：兩者皆為 NaN 視為一致
        actual    = df_check["log_return"]
        both_nan  = recomputed.isna() & actual.isna()
        both_real = (~recomputed.isna()) & (~actual.isna())

        diff = (recomputed[both_real] - actual[both_real]).abs()
        max_diff = float(diff.max()) if len(diff) > 0 else 0.0

        assert max_diff < LOG_RETURN_TOL, (
            f"[{adr_ticker}→{tw_code}] {label} ({ticker}) log_return 與重算結果不一致："
            f"最大差距 {max_diff:.2e} > {LOG_RETURN_TOL:.0e}。"
            f"疑似 log_return 被獨立 ffill（違反政策原則 1）。"
        )


@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_no_future_imputation(adr_ticker, tw_code):
    """
    T10：補值來源日必須早於或等於該列日期（不可用未來資料補過去）
    """
    for label, ticker in [("ADR", adr_ticker), ("TW", tw_code)]:
        path = _adr_path(ticker) if label == "ADR" else _tw_path(ticker)
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()

        if "imputation_source_date" not in df.columns:
            pytest.skip(f"{label} 無 imputation_source_date 欄位（舊版 pipeline）")

        if "is_imputed" not in df.columns:
            continue

        imputed_rows = df[df["is_imputed"].fillna(False).astype(bool)]
        if len(imputed_rows) == 0:
            continue   # 沒有補值列，自動通過

        # 解析 imputation_source_date 為 datetime
        src_dates = pd.to_datetime(imputed_rows["imputation_source_date"],
                                   errors="coerce")

        # 對每一筆 imputed 列：source_date 必須 ≤ 該列日期
        violations = []
        for d, src in src_dates.items():
            if pd.isna(src):
                continue
            if src > d:
                violations.append((d, src))

        assert not violations, (
            f"[{adr_ticker}→{tw_code}] {label} ({ticker}) 偵測到未來補值！\n"
            + "\n".join(f"  {d.date()} ← {src.date()}（晚於本日）"
                        for d, src in violations[:5])
        )


# ════════════════════════════════════════════════════════════
# 第三組：graph_builder 圖層測試（T11–T13）
# ════════════════════════════════════════════════════════════

# 圖快照目錄（與 graph_builder.py 預設一致）
GRAPH_SNAPSHOT_DIR = "data/graphs/snapshots"

# 抽樣設定：不需要每張都驗，取頭、中、尾各幾張
GRAPH_SAMPLE_HEAD = 3
GRAPH_SAMPLE_MID  = 5
GRAPH_SAMPLE_TAIL = 3


def _get_snapshot_sample() -> list:
    """
    從圖快照目錄中取頭、中、尾三段樣本。
    回傳 .pt 檔路徑清單。若目錄不存在回傳空清單。
    """
    snap_dir = Path(GRAPH_SNAPSHOT_DIR)
    if not snap_dir.exists():
        return []
    files = sorted(snap_dir.glob("graph_*.pt"))
    if len(files) == 0:
        return []

    n = len(files)
    head = files[:GRAPH_SAMPLE_HEAD]
    tail = files[-GRAPH_SAMPLE_TAIL:]
    if n > GRAPH_SAMPLE_HEAD + GRAPH_SAMPLE_TAIL + GRAPH_SAMPLE_MID:
        rng = np.random.default_rng(seed=42)
        mid_pool = files[GRAPH_SAMPLE_HEAD : n - GRAPH_SAMPLE_TAIL]
        mid = list(rng.choice(mid_pool, size=GRAPH_SAMPLE_MID, replace=False))
    else:
        mid = files[GRAPH_SAMPLE_HEAD : n - GRAPH_SAMPLE_TAIL]

    # 去重並排序
    seen = set()
    result = []
    for f in head + sorted(mid) + tail:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def _load_snapshot(path):
    """安全載入一張圖快照。"""
    import torch
    return torch.load(path, weights_only=False)


def test_graph_window_no_lookahead():
    """
    T11：圖快照的回看視窗不可包含 target_date 及之後的資料

    對抽樣的快照逐一驗證：
      ① window_end < target_date（嚴格早於，核心防護）
      ② window_end 與 target_date 之間日曆日不超過 10 天（合理距離）
      ③ window 期間在 features CSV 中實際有 ≈ lookback_days 筆交易日
         （用 XTAI 交易日驗證，而非 pd.bdate_range，後者不含台灣假期）

    注意：graph_builder 的 corr_window=60 指的是「XTAI 交易日 60 筆」，
    而非「日曆工作日 60 天」。因此 window_start ~ window_end 的日曆跨度
    通常 > 60（因為中間有台灣國定假日被跳過）。
    """
    samples = _get_snapshot_sample()
    if not samples:
        pytest.skip(f"找不到圖快照於 {GRAPH_SNAPSHOT_DIR}，跳過圖層測試")

    # 載入一支 TW 的 features CSV 作為「XTAI 交易日基準」
    pair_map = PAIR_MAP
    first_tw_code = next(iter(pair_map.values()))
    tw_csv_path = _tw_path(first_tw_code)
    if not tw_csv_path.exists():
        pytest.skip(f"TW features CSV 不存在：{tw_csv_path}")
    tw_df = pd.read_csv(tw_csv_path, index_col="Date",
                        parse_dates=True).sort_index()
    tw_dates_set = set(tw_df.index)

    fail_msgs = []
    for path in samples:
        data = _load_snapshot(path)

        target  = pd.Timestamp(data.target_date)
        w_start = pd.Timestamp(data.window_start)
        w_end   = pd.Timestamp(data.window_end)
        lookback = data.lookback_days

        # 檢查 1：window_end 嚴格早於 target_date（核心防護）
        if w_end >= target:
            fail_msgs.append(
                f"  [{path.stem}] window_end={w_end.date()} "
                f">= target_date={target.date()}（Look-ahead Bias！）"
            )

        # 檢查 2：window_end 與 target_date 之間距離合理
        gap_days = (target - w_end).days
        if gap_days > 10:
            fail_msgs.append(
                f"  [{path.stem}] target - window_end = {gap_days} 天，"
                f"差距異常大（可能有缺漏）"
            )

        # 檢查 3：window 內的 XTAI 交易日數 ≈ lookback_days
        # 用 features CSV 的日期索引計算，而非 pd.bdate_range
        tw_in_window = [d for d in tw_dates_set
                        if w_start <= d <= w_end]
        actual_trading_days = len(tw_in_window)

        # 容忍 ±3 天（ADR 與 TW 交易日差異 + 邊界效應）
        if abs(actual_trading_days - lookback) > 3:
            fail_msgs.append(
                f"  [{path.stem}] 視窗內 XTAI 交易日 {actual_trading_days} "
                f"≠ lookback_days={lookback}（差距 > 3）"
            )

    assert not fail_msgs, (
        f"T11 Look-ahead Bias 圖層偵測失敗！\n"
        f"  共驗證 {len(samples)} 張快照，{len(fail_msgs)} 項違反：\n"
        + "\n".join(fail_msgs)
    )


def test_a12_strictly_diagonal():
    """
    T12：A12 跨層連接矩陣必須嚴格對角，無跨公司洩漏

    對抽樣的快照逐一驗證：
      ① A12 edge_index 的 src[i] == dst[i]（嚴格一對一）
      ② A12 覆蓋所有節點（0 ~ n_nodes-1）
      ③ 無重複邊
    """
    import torch

    samples = _get_snapshot_sample()
    if not samples:
        pytest.skip(f"找不到圖快照於 {GRAPH_SNAPSHOT_DIR}，跳過圖層測試")

    fail_msgs = []
    for path in samples:
        data = _load_snapshot(path)

        # 取得 A12 邊
        a12_type = ("adr", "cross", "tw")
        if a12_type not in data.edge_types:
            fail_msgs.append(f"  [{path.stem}] 缺少 A12 邊類型 {a12_type}")
            continue

        a12 = data[a12_type].edge_index
        n_nodes = data["adr"].x.shape[0]

        # 檢查 1：邊數 = 節點數
        if a12.shape[1] != n_nodes:
            fail_msgs.append(
                f"  [{path.stem}] A12 邊數 {a12.shape[1]} ≠ 節點數 {n_nodes}"
            )
            continue

        # 檢查 2：嚴格對角（src == dst for every edge）
        for k in range(a12.shape[1]):
            src, dst = int(a12[0, k]), int(a12[1, k])
            if src != dst:
                fail_msgs.append(
                    f"  [{path.stem}] A12 第 {k} 條邊 src={src} ≠ dst={dst}"
                    f"（跨公司洩漏！）"
                )

        # 檢查 3：覆蓋所有節點
        src_set = set(a12[0].tolist())
        expected = set(range(n_nodes))
        if src_set != expected:
            missing = expected - src_set
            fail_msgs.append(
                f"  [{path.stem}] A12 未覆蓋節點 {missing}"
            )

        # 檢查 4：無重複邊
        edge_pairs = [(int(a12[0, k]), int(a12[1, k]))
                      for k in range(a12.shape[1])]
        if len(set(edge_pairs)) != len(edge_pairs):
            fail_msgs.append(f"  [{path.stem}] A12 有重複邊")

    assert not fail_msgs, (
        f"T12 A12 跨層洩漏偵測失敗！\n"
        f"  共驗證 {len(samples)} 張快照，{len(fail_msgs)} 項違反：\n"
        + "\n".join(fail_msgs)
    )


def test_node_features_use_correct_date():
    """
    T13：圖快照中節點特徵取自 window_end 那天的資料

    驗證方式：
      ① 讀回 features CSV 中 window_end 那天的值
      ② 與快照中 data["adr"].x / data["tw"].x 逐一比對
      ③ 容忍 NaN 填 0 的差異（long_gap 列的處理）
    """
    import torch

    samples = _get_snapshot_sample()
    if not samples:
        pytest.skip(f"找不到圖快照於 {GRAPH_SNAPSHOT_DIR}，跳過圖層測試")

    # 載入 PAIR_MAP
    pair_map = PAIR_MAP   # 已在模組層級載入

    adr_tickers = list(pair_map.keys())
    tw_codes    = list(pair_map.values())

    # 載入所有 features CSV（快取）
    adr_dfs = {}
    tw_dfs  = {}
    for ticker in adr_tickers:
        p = _adr_path(ticker)
        if p.exists():
            adr_dfs[ticker] = pd.read_csv(p, index_col="Date",
                                           parse_dates=True).sort_index()
    for code in tw_codes:
        p = _tw_path(code)
        if p.exists():
            tw_dfs[code] = pd.read_csv(p, index_col="Date",
                                        parse_dates=True).sort_index()

    if not adr_dfs or not tw_dfs:
        pytest.skip("features CSV 不足以做交叉驗證")

    # 9 維特徵欄位名
    feat_cols = [
        "log_return", "RSI_14",
        "MACD", "MACD_signal", "MACD_hist",
        "BB_pos", "MA5_dev", "MA20_dev",
        "log_volume_z",
    ]

    fail_msgs = []
    for path in samples:
        data = _load_snapshot(path)
        w_end = pd.Timestamp(data.window_end)

        # 驗證 ADR 端
        for i, ticker in enumerate(adr_tickers):
            if ticker not in adr_dfs:
                continue
            df = adr_dfs[ticker]

            # graph_builder 取 window_end 或之前最近的有效日
            valid_dates = df.index[df.index <= w_end]
            if len(valid_dates) == 0:
                continue
            actual_date = valid_dates[-1]

            # 從 CSV 讀出的值
            try:
                csv_vals = df.loc[actual_date, feat_cols].values.astype(float)
            except KeyError:
                continue

            # 從快照讀出的值
            snap_vals = data["adr"].x[i].numpy()

            # 逐維比對（容忍 NaN→0 的填充差異）
            for j, col in enumerate(feat_cols):
                csv_v  = csv_vals[j]
                snap_v = snap_vals[j]

                if np.isnan(csv_v):
                    # CSV 是 NaN → 快照應填 0（graph_builder 的 NAN_FILL_VALUE）
                    if snap_v != 0.0:
                        fail_msgs.append(
                            f"  [{path.stem}] ADR {ticker}[{col}] "
                            f"CSV=NaN 但快照={snap_v:.4f}（應為 0.0）"
                        )
                else:
                    diff = abs(csv_v - snap_v)
                    if diff > 1e-5:
                        fail_msgs.append(
                            f"  [{path.stem}] ADR {ticker}[{col}] "
                            f"CSV={csv_v:.6f} ≠ 快照={snap_v:.6f}（差 {diff:.2e}）"
                        )

        # 驗證 TW 端（同邏輯）
        for i, code in enumerate(tw_codes):
            if code not in tw_dfs:
                continue
            df = tw_dfs[code]
            valid_dates = df.index[df.index <= w_end]
            if len(valid_dates) == 0:
                continue
            actual_date = valid_dates[-1]

            try:
                csv_vals = df.loc[actual_date, feat_cols].values.astype(float)
            except KeyError:
                continue

            snap_vals = data["tw"].x[i].numpy()

            for j, col in enumerate(feat_cols):
                csv_v  = csv_vals[j]
                snap_v = snap_vals[j]

                if np.isnan(csv_v):
                    if snap_v != 0.0:
                        fail_msgs.append(
                            f"  [{path.stem}] TW {code}[{col}] "
                            f"CSV=NaN 但快照={snap_v:.4f}（應為 0.0）"
                        )
                else:
                    diff = abs(csv_v - snap_v)
                    if diff > 1e-5:
                        fail_msgs.append(
                            f"  [{path.stem}] TW {code}[{col}] "
                            f"CSV={csv_v:.6f} ≠ 快照={snap_v:.6f}（差 {diff:.2e}）"
                        )

    assert not fail_msgs, (
        f"T13 節點特徵日期一致性檢查失敗！\n"
        f"  共驗證 {len(samples)} 張快照，{len(fail_msgs)} 項不一致：\n"
        + "\n".join(fail_msgs[:10])
        + (f"\n  ... 另有 {len(fail_msgs)-10} 項"
           if len(fail_msgs) > 10 else "")
    )


# ════════════════════════════════════════════════════════════
# T14：MultiplexDataset 組裝的 T 步歷史序列無 Look-ahead
# Corresponds to IMPLEMENTATION_SPEC §10 Step 6
# ════════════════════════════════════════════════════════════

def test_dataset_sequence_no_lookahead():
    """
    T14: MultiplexDataset 在組裝 LSTM 輸入序列時，
    必須嚴格保證最後一列日期 < target_date（即 ≤ target_date - 1 交易日）。

    抽樣策略：頭中尾各取若干張快照。
    """
    pytest.importorskip("yaml")
    from src.dataset.multiplex_dataset import (
        MultiplexDataset, ADR_TICKERS, TW_CODES, TECH_COLS,
    )

    project_root = Path(__file__).resolve().parents[1]
    ds = MultiplexDataset(
        snapshot_dir=str(project_root / "data" / "graphs" / "snapshots"),
        features_dir=str(project_root / "data" / "features"),
        T=20,
        split="all",
        config_path=str(project_root / "configs" / "base.yaml"),
    )

    n = len(ds)
    # 頭 3 + 中段 5 + 尾 3（共 11 張）
    rng = np.random.default_rng(42)
    head_idx = list(range(3))
    tail_idx = list(range(n - 3, n))
    mid_idx  = rng.choice(range(3, n - 3), size=5, replace=False).tolist()
    sample_idxs = sorted(set(head_idx + tail_idx + mid_idx))

    # 預載 CSV 拿到每市場的「< target_date 最後交易日」
    adr_dfs = {t: pd.read_csv(_adr_path(t), index_col=0, parse_dates=True) for t in ADR_TICKERS}
    tw_dfs  = {c: pd.read_csv(_tw_path(c),  index_col=0, parse_dates=True) for c in TW_CODES}

    fail_msgs: list[str] = []
    for idx in sample_idxs:
        item = ds[idx]
        target_date = pd.Timestamp(item["target_date"])

        # x_seq_L1 [T, n, F]：對每個 ticker，最後一步應等於 < target_date 的最後一筆 CSV 值
        x1 = item["x_seq_L1"].numpy()
        x2 = item["x_seq_L2"].numpy()

        # ADR 端
        for j, ticker in enumerate(ADR_TICKERS):
            df = adr_dfs[ticker]
            past = df.loc[df.index < target_date]
            if past.empty:
                continue
            last_csv_date = past.index[-1]
            # 嚴格守護：最後一筆 < target_date
            if last_csv_date >= target_date:
                fail_msgs.append(
                    f"[idx={idx}, {ticker}] last_csv_date={last_csv_date.date()} "
                    f">= target_date={target_date.date()}"
                )
                continue
            # x_seq 最後一步應 == CSV 該日的 9 維特徵（NaN→0）
            csv_vals = past.loc[last_csv_date, TECH_COLS].values.astype(float)
            csv_vals = np.nan_to_num(csv_vals, nan=0.0, posinf=0.0, neginf=0.0)
            seq_last = x1[-1, j, :]
            diff = np.abs(csv_vals - seq_last).max()
            if diff > 1e-5:
                fail_msgs.append(
                    f"[idx={idx}, ADR {ticker}] x_seq[-1] ≠ CSV last "
                    f"(diff={diff:.2e}, last_csv_date={last_csv_date.date()})"
                )

        # TW 端
        for j, code in enumerate(TW_CODES):
            df = tw_dfs[code]
            past = df.loc[df.index < target_date]
            if past.empty:
                continue
            last_csv_date = past.index[-1]
            if last_csv_date >= target_date:
                fail_msgs.append(
                    f"[idx={idx}, {code}] last_csv_date={last_csv_date.date()} "
                    f">= target_date={target_date.date()}"
                )
                continue
            csv_vals = past.loc[last_csv_date, TECH_COLS].values.astype(float)
            csv_vals = np.nan_to_num(csv_vals, nan=0.0, posinf=0.0, neginf=0.0)
            seq_last = x2[-1, j, :]
            diff = np.abs(csv_vals - seq_last).max()
            if diff > 1e-5:
                fail_msgs.append(
                    f"[idx={idx}, TW {code}] x_seq[-1] ≠ CSV last "
                    f"(diff={diff:.2e}, last_csv_date={last_csv_date.date()})"
                )

    assert not fail_msgs, (
        f"T14 Dataset 序列 Look-ahead 守護失敗！\n"
        f"  共驗證 {len(sample_idxs)} 張快照，{len(fail_msgs)} 項不一致：\n"
        + "\n".join(fail_msgs[:10])
        + (f"\n  ... 另有 {len(fail_msgs)-10} 項" if len(fail_msgs) > 10 else "")
    )


# ════════════════════════════════════════════════════════════
# 獨立執行模式（不需 pytest）
# ════════════════════════════════════════════════════════════

def _run_one_test(test_func, adr_ticker, tw_code):
    """嘗試執行單一測試，回傳 (status, message)。"""
    try:
        test_func(adr_ticker, tw_code)
        return ("PASS", "")
    except pytest.skip.Exception as e:
        return ("SKIP", str(e))
    except AssertionError as e:
        msg = str(e).split("\n")[0][:80]
        return ("FAIL", msg)
    except Exception as e:
        return ("ERROR", f"{type(e).__name__}: {e}")


def _diagnose():
    """獨立診斷模式：對每組配對跑全部測試並輸出表格。"""
    pairs = get_available_pairs()

    if not pairs:
        print(f"[FAIL] 找不到任何配對檔案於 {ADR_DIR} / {TW_DIR}")
        return 1

    print(f"\n[test_no_lookahead 診斷] 共 {len(pairs)} 組配對，13 項測試\n")

    # 對應每個 test 的函式
    # T1–T10：逐配對跑（需要 adr_ticker, tw_code 參數）
    # T11–T13：全域跑（不需要配對參數）
    pair_tests = [
        ("T1  no_nan_excl_long_gap",
         test_no_nan_after_alignment_excluding_long_gap),
        ("T2  adr_t_eq_prev_day",
         test_adr_t_equals_previous_day),
        ("T3  no_perfect_corr",
         test_no_perfect_correlation),
        ("T4  monotonic_idx",
         test_monotonic_date_index),
        ("T5  min_common_days",
         test_minimum_common_trading_days),
        ("T6  split_no_leakage",
         test_train_test_split_no_leakage),
        ("T7  no_consec_identical",
         test_no_consecutive_identical_log_returns),
        ("T8  trace_cols_exist",
         test_imputation_flag_exists),
        ("T9  logret_consistency",
         test_log_return_recomputation_consistency),
        ("T10 no_future_imp",
         test_no_future_imputation),
    ]

    global_tests = [
        ("T11 graph_no_lookahead",
         test_graph_window_no_lookahead),
        ("T12 a12_diagonal",
         test_a12_strictly_diagonal),
        ("T13 node_feat_date",
         test_node_features_use_correct_date),
        ("T14 dataset_seq_no_lookahead",
         test_dataset_sequence_no_lookahead),
    ]

    # 表頭
    pair_labels = [f"{a}→{t}" for a, t in pairs]
    col_w = 12
    header = f"{'Test':<28}" + "".join(f"{lbl[:col_w-1]:<{col_w}}" for lbl in pair_labels)
    print(header)
    print("-" * len(header))

    summary = {p: {"pass": 0, "fail": 0, "skip": 0, "error": 0} for p in pair_labels}
    fail_details = []

    for test_name, func in pair_tests:
        row = f"{test_name:<28}"
        for (a, t), lbl in zip(pairs, pair_labels):
            status, msg = _run_one_test(func, a, t)
            symbol = {"PASS": "OK", "FAIL": "FAIL", "SKIP": "—", "ERROR": "!"}[status]
            row += f"{symbol:<{col_w}}"
            summary[lbl][status.lower()] += 1
            if status in ("FAIL", "ERROR"):
                fail_details.append((test_name, lbl, status, msg))
        print(row)

    print("-" * len(header))

    # 統計列
    stat_pass  = "PASS  ".ljust(28) + "".join(
        f"{summary[lbl]['pass']:<{col_w}}"  for lbl in pair_labels)
    stat_fail  = "FAIL  ".ljust(28) + "".join(
        f"{summary[lbl]['fail']:<{col_w}}"  for lbl in pair_labels)
    stat_skip  = "SKIP  ".ljust(28) + "".join(
        f"{summary[lbl]['skip']:<{col_w}}"  for lbl in pair_labels)
    print(stat_pass)
    if any(summary[lbl]["fail"] for lbl in pair_labels):
        print(stat_fail)
    if any(summary[lbl]["skip"] for lbl in pair_labels):
        print(stat_skip)

    # ── T11–T13 全域測試（不逐配對跑）─────────────────────
    print(f"\n{'Test':<28}{'Status':<12}")
    print("-" * 40)

    global_fail = 0
    for test_name, func in global_tests:
        try:
            func()
            status, msg = "PASS", ""
        except pytest.skip.Exception as e:
            status, msg = "SKIP", str(e)
        except AssertionError as e:
            status, msg = "FAIL", str(e).split("\n")[0][:80]
            global_fail += 1
        except Exception as e:
            status, msg = "ERROR", f"{type(e).__name__}: {e}"
            global_fail += 1

        symbol = {"PASS": "OK", "FAIL": "FAIL", "SKIP": "—", "ERROR": "!"}[status]
        print(f"{test_name:<28}{symbol}")
        if status in ("FAIL", "ERROR"):
            fail_details.append((test_name, "global", status, msg))

    print("-" * 40)

    # 失敗細節
    if fail_details:
        print("\n失敗細節：")
        for tname, lbl, status, msg in fail_details:
            print(f"  [{status}] {tname} @ {lbl}：{msg}")

    # 整體結論
    total_fail = sum(summary[lbl]["fail"] + summary[lbl]["error"]
                     for lbl in pair_labels) + global_fail
    if total_fail == 0:
        print(f"\n全部通過")
        return 0
    else:
        print(f"\n[FAIL] 共 {total_fail} 項失敗")
        return 1


if __name__ == "__main__":
    raise SystemExit(_diagnose())