"""
test_no_lookahead.py — Look-ahead Bias 批次防護測試
=====================================================
針對所有 ADR-台股配對的真實資料逐一驗證時區對齊正確性。

測試邏輯：ADR(t) → TW(t+1)
    合法：用 ADR 在 t 日的收盤特徵，預測台股 t+1 日報酬
    違法：用任何 t+1 日的 ADR 資料當輸入（未來洩漏）

執行方式：
    pytest tests/test_no_lookahead.py -v
    pytest tests/test_no_lookahead.py -v --tb=short   # 失敗時只顯示摘要
"""

import pytest
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path


# ════════════════════════════════════════════════════════════
# 設定：ADR-台股配對表 + 資料路徑
# ════════════════════════════════════════════════════════════

ADR_DIR  = "src/data/processed/adr"
TW_DIR   = "src/data/processed/tw"

# ADR ticker → 台股代碼（1-to-1 對應）
PAIR_MAP = {
    "ASX":   "3711",    # 日月光     # 半導體
    "AUOTY": "2409",    # 友達光電   # 光電產業
    "CHT":   "2412",    # 中華電信   # 電信業
    "TSM":   "2330",    # 台積電     # 半導體
    "UMC":   "2303",    # 聯電       # 晶圓代工
    "IMOS":  "8150",    # 南茂       # 半導體
    "HNHPF": "2317",    # 鴻海       # 其他電子業
}

MIN_COMMON_DAYS = 200   # 對齊後最少共同交易日數
CORR_UPPER_BOUND = 0.999  # adr_t 與 tw_t1 相關係數不可超過此值


# ════════════════════════════════════════════════════════════
# 工具函數
# ════════════════════════════════════════════════════════════

def load_pair(adr_ticker: str, tw_code: str):
    """讀取一對 ADR / 台股 CSV，回傳 (adr_df, tw_df)。"""
    adr_path = os.path.join(ADR_DIR, f"{adr_ticker}.csv")
    tw_path  = os.path.join(TW_DIR,  f"{tw_code}.csv")

    if not os.path.exists(adr_path):
        pytest.skip(f"ADR 檔案不存在：{adr_path}")
    if not os.path.exists(tw_path):
        pytest.skip(f"台股檔案不存在：{tw_path}")

    adr_df = pd.read_csv(adr_path, index_col="Date", parse_dates=True).sort_index()
    tw_df  = pd.read_csv(tw_path,  index_col="Date", parse_dates=True).sort_index()
    return adr_df, tw_df


def align_adr_to_tw(adr_df: pd.DataFrame,
                    tw_df:  pd.DataFrame,
                    feature_col: str = "log_return") -> pd.DataFrame:
    """
    核心對齊函數：ADR(t) → TW(t+1)。

    adr_df 的特徵欄位 shift(1)，使得：
        merged.loc[d, "adr_t"]  = adr_df[feature_col][d-1]  ← 昨日 ADR
        merged.loc[d, "tw_t1"] = tw_df[feature_col][d]     ← 今日台股（預測目標）
    """
    adr_shifted = adr_df[feature_col].shift(1).rename("adr_t")
    tw_col      = tw_df[feature_col].rename("tw_t1")

    merged = pd.concat([adr_shifted, tw_col], axis=1, join="inner").dropna()
    return merged


def get_available_pairs():
    """
    掃描實際存在的 ADR-台股配對，回傳 pytest.param list。
    只回傳兩個 CSV 都存在的配對，讓測試可以有選擇性地執行。
    """
    pairs = []
    for adr_ticker, tw_code in PAIR_MAP.items():
        adr_path = os.path.join(ADR_DIR, f"{adr_ticker}.csv")
        tw_path  = os.path.join(TW_DIR,  f"{tw_code}.csv")
        if os.path.exists(adr_path) and os.path.exists(tw_path):
            pairs.append(pytest.param(adr_ticker, tw_code,
                                      id=f"{adr_ticker}→{tw_code}"))
    return pairs


# ════════════════════════════════════════════════════════════
# Test 1：對齊後無 NaN
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_no_nan_after_alignment(adr_ticker, tw_code):
    """
    對齊後的 DataFrame 不應有 NaN。

    shift(1) 會讓第一列變 NaN，dropna() 應將其移除。
    若仍有 NaN 代表原始資料在中段有缺漏值未處理。
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)
    merged = align_adr_to_tw(adr_df, tw_df)

    nan_count = merged.isna().sum().sum()
    assert nan_count == 0, (
        f"[{adr_ticker}→{tw_code}] 對齊後仍有 {nan_count} 個 NaN，"
        f"請檢查原始 CSV 中段是否有未填補的缺漏值。"
    )


# ════════════════════════════════════════════════════════════
# Test 2：核心測試 — adr_t[d] 必須等於 adr[d-1]
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_adr_t_equals_previous_day(adr_ticker, tw_code):
    """
    核心 Look-ahead Bias 測試。

    對合併表中每一個日期 d，驗證：
        merged["adr_t"][d]  ==  adr_df["log_return"][d-1]

    若不相等，代表 shift(1) 沒有正確執行，
    或對齊邏輯用了當日的 ADR 值（未來洩漏）。

    為節省時間，取前 50 筆逐一驗證，已足以涵蓋邊界情況。
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)
    merged = align_adr_to_tw(adr_df, tw_df)

    adr_log_ret = adr_df["log_return"].dropna()
    check_dates = merged.index[:50]   # 驗證前 50 筆

    for d in check_dates:
        # 取前一個在 adr_log_ret 中存在的日期
        pos_in_adr = adr_log_ret.index.get_loc(d) if d in adr_log_ret.index else None

        if pos_in_adr is None or pos_in_adr == 0:
            continue   # d 不在 ADR 索引中，或已是第一筆，跳過

        expected = float(adr_log_ret.iloc[pos_in_adr - 1])
        actual   = float(merged.loc[d, "adr_t"])

        assert abs(actual - expected) < 1e-9, (
            f"[{adr_ticker}→{tw_code}] Look-ahead Bias 偵測！\n"
            f"  日期 {d.date()}: adr_t={actual:.6f} "
            f"≠ adr[前一日]={expected:.6f}\n"
            f"  差距={abs(actual-expected):.2e}  → shift(1) 未正確執行"
        )


# ════════════════════════════════════════════════════════════
# Test 3：adr_t 與 tw_t1 相關係數不可趨近 1
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_no_perfect_correlation(adr_ticker, tw_code):
    """
    若 adr_t 和 tw_t1 的相關係數接近 1.0，
    代表輸入特徵與預測目標幾乎相同——這通常意味著用了同一欄資料，
    或未來資訊洩漏（例如 adr_t 被對齊到了 tw_t1 同一天）。
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)
    merged = align_adr_to_tw(adr_df, tw_df)

    if len(merged) < 30:
        pytest.skip(f"[{adr_ticker}→{tw_code}] 樣本數不足 30，跳過相關係數測試")

    corr = merged["adr_t"].corr(merged["tw_t1"])

    assert corr < CORR_UPPER_BOUND, (
        f"[{adr_ticker}→{tw_code}] 異常高相關係數：{corr:.4f} ≥ {CORR_UPPER_BOUND}\n"
        f"  可能原因：adr_t 與 tw_t1 使用了相同欄位，或時間對齊邏輯錯誤。"
    )


# ════════════════════════════════════════════════════════════
# Test 4：日期索引嚴格遞增
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_monotonic_date_index(adr_ticker, tw_code):
    """
    合併後的日期索引必須嚴格遞增（無重複、無亂序）。

    日期亂序是另一種 Look-ahead Bias 的成因：
    若訓練時資料包含未來日期，模型等同於偷看了答案。
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)
    merged = align_adr_to_tw(adr_df, tw_df)

    assert merged.index.is_monotonic_increasing, (
        f"[{adr_ticker}→{tw_code}] 日期索引非嚴格遞增，請檢查資料排序。\n"
        f"  重複日期數：{merged.index.duplicated().sum()}"
    )


# ════════════════════════════════════════════════════════════
# Test 5：共同交易日數量足夠
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_minimum_common_trading_days(adr_ticker, tw_code):
    """
    對齊後的共同交易日數量必須 ≥ MIN_COMMON_DAYS。

    若過少，代表：
      (a) 兩市場行事曆差異過大（例如 ADR 資料起始日晚於台股）
      (b) 其中一方資料有大量缺漏
      (c) 原始資料下載的時間範圍不一致
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)
    merged = align_adr_to_tw(adr_df, tw_df)

    assert len(merged) >= MIN_COMMON_DAYS, (
        f"[{adr_ticker}→{tw_code}] 共同交易日僅 {len(merged)} 筆 "
        f"< 最低要求 {MIN_COMMON_DAYS}。\n"
        f"  ADR 期間：{adr_df.index[0].date()} ~ {adr_df.index[-1].date()}  "
        f"({len(adr_df)} 筆)\n"
        f"  台股期間：{tw_df.index[0].date()} ~ {tw_df.index[-1].date()}  "
        f"({len(tw_df)} 筆)"
    )


# ════════════════════════════════════════════════════════════
# Test 6（加強版）：訓練集內不包含測試集日期的 ADR 特徵
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("adr_ticker,tw_code", get_available_pairs())
def test_train_test_split_no_leakage(adr_ticker, tw_code):
    """
    模擬 70/15/15 切分後，確認訓練集的 adr_t 不包含測試集期間的日期。

    這是防止「跨 split 洩漏」：
    若 Rolling Window 切分實作有誤，訓練樣本的 adr_t 可能包含
    測試期間的 ADR 資料，導致評估指標虛高。
    """
    adr_df, tw_df = load_pair(adr_ticker, tw_code)
    merged = align_adr_to_tw(adr_df, tw_df)

    if len(merged) < MIN_COMMON_DAYS:
        pytest.skip(f"[{adr_ticker}→{tw_code}] 樣本不足，跳過 split 洩漏測試")

    n       = len(merged)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_dates = merged.index[:n_train]
    test_dates  = merged.index[n_train + n_val:]

    # 訓練集的 adr_t 值對應的「原始 ADR 日期」= 前一個交易日
    # 所有這些日期都應早於測試集的起始日
    test_start = test_dates[0] if len(test_dates) > 0 else merged.index[-1]

    # 訓練集最後一筆的 adr_t 對應的原始日期 = train_dates[-1] - 1
    # 必須 < test_start
    last_train_adr_date = adr_df.index[
        adr_df.index.get_loc(train_dates[-1]) - 1
    ]

    assert last_train_adr_date < test_start, (
        f"[{adr_ticker}→{tw_code}] 訓練集的最後一筆 adr_t "
        f"對應的 ADR 日期（{last_train_adr_date.date()}）"
        f"不早於測試集起始日（{test_start.date()}）。\n"
        f"  可能存在跨 split 的時間洩漏。"
    )


# ════════════════════════════════════════════════════════════
# 獨立執行時的彙總報告
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    直接執行時輸出所有配對的簡易報告（不需要 pytest）。
    用途：快速人工確認，不取代 pytest 的正式驗證。
    """
    print("=" * 65)
    print("Look-ahead Bias 快速診斷報告")
    print("=" * 65)

    results = []

    for adr_ticker, tw_code in PAIR_MAP.items():
        adr_path = os.path.join(ADR_DIR, f"{adr_ticker}.csv")
        tw_path  = os.path.join(TW_DIR,  f"{tw_code}.csv")

        if not (os.path.exists(adr_path) and os.path.exists(tw_path)):
            results.append({
                "pair": f"{adr_ticker}→{tw_code}",
                "status": "SKIP", "reason": "CSV 不存在", "days": 0
            })
            continue

        adr_df = pd.read_csv(adr_path, index_col="Date", parse_dates=True).sort_index()
        tw_df  = pd.read_csv(tw_path,  index_col="Date", parse_dates=True).sort_index()
        merged = align_adr_to_tw(adr_df, tw_df)

        issues = []

        # 1. NaN 檢查
        nan_cnt = merged.isna().sum().sum()
        if nan_cnt > 0:
            issues.append(f"NaN={nan_cnt}")

        # 2. 核心對齊檢查（前 20 筆）
        bias_found = False
        adr_lr = adr_df["log_return"].dropna()
        for d in merged.index[:20]:
            if d not in adr_lr.index:
                continue
            pos = adr_lr.index.get_loc(d)
            if pos == 0:
                continue
            expected = float(adr_lr.iloc[pos - 1])
            actual   = float(merged.loc[d, "adr_t"])
            if abs(actual - expected) > 1e-9:
                bias_found = True
                issues.append(f"Look-ahead Bias@{d.date()}")
                break

        # 3. 樣本數
        if len(merged) < MIN_COMMON_DAYS:
            issues.append(f"樣本不足({len(merged)}<{MIN_COMMON_DAYS})")

        # 4. 相關係數
        if len(merged) >= 30:
            corr = merged["adr_t"].corr(merged["tw_t1"])
            if corr >= CORR_UPPER_BOUND:
                issues.append(f"異常相關({corr:.3f})")

        status = "PASS" if not issues else "FAIL"
        results.append({
            "pair":   f"{adr_ticker}→{tw_code}",
            "status": status,
            "days":   len(merged),
            "reason": "；".join(issues) if issues else "—",
        })

    # 印出結果表
    print(f"  {'配對':<18} {'狀態':^6} {'共同天數':>8}  說明")
    print(f"  {'-'*18} {'-'*6} {'-'*8}  {'-'*30}")
    pass_n = fail_n = skip_n = 0
    for r in results:
        icon = "✅" if r["status"] == "PASS" else ("⏭" if r["status"] == "SKIP" else "❌")
        print(f"  {r['pair']:<18} {icon} {r['status']:^4} {r['days']:>8}  {r['reason']}")
        if r["status"] == "PASS": pass_n += 1
        elif r["status"] == "SKIP": skip_n += 1
        else: fail_n += 1

    print(f"\n  結果：PASS={pass_n}  FAIL={fail_n}  SKIP={skip_n}")
    if fail_n == 0:
        print("  ✅ 所有可用配對均通過 Look-ahead Bias 檢查")
    else:
        print("  ❌ 有配對未通過，請修復後重新執行 pytest")
    print("=" * 65)