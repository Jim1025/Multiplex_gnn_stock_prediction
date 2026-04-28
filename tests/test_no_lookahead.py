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
  T7  test_no_consecutive_identical_log_returns
  T8  test_imputation_flag_exists
  T9  test_log_return_recomputation_consistency
  T10 test_no_future_imputation

第三組：預留位置（graph_builder.py 動工後補）
  T11 test_graph_window_no_lookahead       [TODO]
  T12 test_a12_strictly_diagonal           [TODO]
  T13 test_node_features_use_correct_date  [TODO]
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
    T7：偵測連續 3 日完全相同的 log_return（污染指標）

    若連續 3 日 log_return 完全相同，極可能是 ffill 污染 log_return 本身
    （違反政策原則 1）。但需排除 is_imputed=True 的列，因為政策允許
    補值列的 log_return = 0（且連續補值必然產生連續 0）。
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

        # 連續 3 筆完全相同 → 可疑
        same_with_prev1 = (lr.values[1:]   == lr.values[:-1])
        same_with_prev2 = (lr.values[2:]   == lr.values[1:-1])
        # 兩個布林陣列長度差 1，需對齊
        triple_match = same_with_prev1[1:] & same_with_prev2
        
        triple_values = lr.values[2:][triple_match]
        nonzero_triple = (triple_values != 0).sum()  # ★ 只計算非零事件
        ratio = nonzero_triple / max(len(lr) - 2, 1)
        
        assert ratio < IDENTICAL_LR_RATIO, (
            f"[{adr_ticker}→{tw_code}] {label} ({ticker}) 連續 3 日相同 "
            f"log_return 比例 {ratio:.2%} > {IDENTICAL_LR_RATIO:.0%}，"
            f"疑似 log_return 直接 ffill 污染"
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
# 第三組：graph_builder 預留位置（T11–T13）
# ════════════════════════════════════════════════════════════

# 以下測試將於 graph_builder.py 動工後實作。預留 stub 讓 pytest 顯示
# 為 SKIPPED，提醒開發者這些測試尚未涵蓋。

@pytest.mark.skip(reason="待 graph_builder.py 完成後實作")
def test_graph_window_no_lookahead():
    """T11：圖快照的回看視窗不可包含 target_date 及之後的資料"""
    pass


@pytest.mark.skip(reason="待 graph_builder.py 完成後實作")
def test_a12_strictly_diagonal():
    """T12：A12 跨層連接矩陣必須嚴格對角，無跨公司洩漏"""
    pass


@pytest.mark.skip(reason="待 graph_builder.py 完成後實作")
def test_node_features_use_correct_date():
    """T13：圖快照中 L1 節點用 t 日特徵，L2 節點用 t-1 日特徵"""
    pass


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
        print(f"❌ 找不到任何配對檔案於 {ADR_DIR} / {TW_DIR}")
        return 1

    print(f"\n[test_no_lookahead 診斷] 共 {len(pairs)} 組配對，10 項測試\n")

    # 對應每個 test 的函式
    tests = [
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

    # 表頭
    pair_labels = [f"{a}→{t}" for a, t in pairs]
    col_w = 12
    header = f"{'Test':<28}" + "".join(f"{lbl[:col_w-1]:<{col_w}}" for lbl in pair_labels)
    print(header)
    print("-" * len(header))

    summary = {p: {"pass": 0, "fail": 0, "skip": 0, "error": 0} for p in pair_labels}
    fail_details = []

    for test_name, func in tests:
        row = f"{test_name:<28}"
        for (a, t), lbl in zip(pairs, pair_labels):
            status, msg = _run_one_test(func, a, t)
            symbol = {"PASS": "✓", "FAIL": "✗", "SKIP": "—", "ERROR": "!"}[status]
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

    # 失敗細節
    if fail_details:
        print("\n失敗細節：")
        for tname, lbl, status, msg in fail_details:
            print(f"  [{status}] {tname} @ {lbl}：{msg}")

    # 整體結論
    total_fail = sum(summary[lbl]["fail"] + summary[lbl]["error"]
                     for lbl in pair_labels)
    if total_fail == 0:
        print(f"\n✅ 全部通過")
        return 0
    else:
        print(f"\n❌ 共 {total_fail} 項失敗")
        return 1


if __name__ == "__main__":
    raise SystemExit(_diagnose())