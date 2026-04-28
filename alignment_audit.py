"""
alignment_audit.py — ADR / TW 對齊診斷報告
==============================================
在進入 features.py 之前的「地基檢查」工具。

對 PAIR_MAP 中每組 ADR-TW 配對輸出：
  ① 各自原始交易日數
  ② 共同交易日（ADR 日期 ∩ TW 日期）
  ③ 只在 ADR 有 / 只在 TW 有的日期
  ④ ADR(t) → TW(t+1) 對齊後的有效樣本數
  ⑤ Look-ahead Bias sanity check（adr_t == 前一日 ADR）
  ⑥ 訓練可行性建議（樣本是否足夠、是否該剔除）

執行：
    python alignment_audit.py
    python alignment_audit.py --adr-dir data/processed/adr --tw-dir data/processed/tw

輸出：
    終端機表格摘要
    docs/data_quality/alignment_audit.csv（細項）
    docs/data_quality/alignment_audit.md（給 PM 文件用的摘要）
"""

import os
import argparse
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict


# ════════════════════════════════════════════════════════════
# 配置（與 test_no_lookahead.py 對齊；缺的可在此補上）
# ════════════════════════════════════════════════════════════

# ADR ticker → TW 股票代碼（pipeline 輸出的檔名不含 .TW 後綴）
PAIR_MAP = {
    "ASX":   "3711",    # 日月光    # 半導體
    "AUOTY": "2409",    # 友達光電  # 光電產業
    "CHT":   "2412",    # 中華電信  # 電信業
    "TSM":   "2330",    # 台積電    # 半導體  
    "UMC":   "2303",    # 聯電     # 晶圓代工
    "IMOS":  "8150",    # 南茂     # 半導體
    # "ASUUY": "2357",    # 華碩     # 電子業
    "HNHPF": "2317",    # 鴻海     # 其他電子業
}

# 訓練可行性閾值
MIN_COMMON_DAYS = 1500    # 共同交易日 < 此值警告
MIN_ALIGNED_SAMPLES = 1400  # 對齊後樣本數 < 此值警告
MAX_DATE_DRIFT_DAYS = 5   # ADR 與 TW 起訖日相差超過此天數提醒

OUT_DIR = Path("docs/data_quality")


# ════════════════════════════════════════════════════════════
# 結果容器
# ════════════════════════════════════════════════════════════

@dataclass
class PairAuditResult:
    adr_ticker: str
    tw_code:    str

    # 檔案存在性
    adr_exists: bool = False
    tw_exists:  bool = False

    # 原始天數
    adr_days:   int = 0
    tw_days:    int = 0

    # 起訖日
    adr_start:  Optional[pd.Timestamp] = None
    adr_end:    Optional[pd.Timestamp] = None
    tw_start:   Optional[pd.Timestamp] = None
    tw_end:     Optional[pd.Timestamp] = None

    # 集合差異
    common_days:        int = 0
    only_in_adr:        int = 0
    only_in_tw:         int = 0
    only_in_adr_dates:  List[str] = field(default_factory=list)
    only_in_tw_dates:   List[str] = field(default_factory=list)

    # 對齊後樣本
    aligned_samples:    int = 0
    aligned_start:      Optional[pd.Timestamp] = None
    aligned_end:        Optional[pd.Timestamp] = None

    # Look-ahead sanity check
    lookahead_check_pass: bool = False
    lookahead_max_diff:   float = 0.0

    # 補值統計（從 pipeline 輸出的追溯欄位）
    adr_imputed_rows:   int = 0
    tw_imputed_rows:    int = 0
    adr_long_gap_rows:  int = 0
    tw_long_gap_rows:   int = 0

    # 整體判定
    status:   str = "PENDING"   # PASS / WARN / FAIL / SKIP
    notes:    List[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════
# 工具函數
# ════════════════════════════════════════════════════════════

def _load_csv_safely(path: Path) -> Optional[pd.DataFrame]:
    """讀取 pipeline 處理後的 CSV，失敗回傳 None。"""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
        return df
    except Exception:
        return None


def _align_adr_to_tw(adr_df: pd.DataFrame,
                     tw_df:  pd.DataFrame,
                     feature_col: str = "log_return") -> pd.DataFrame:
    """
    與 test_no_lookahead.py 完全相同的對齊邏輯：ADR(t) → TW(t+1)。

    adr_df[feature_col].shift(1) 後與 tw_df[feature_col] inner join。
    """
    adr_shifted = adr_df[feature_col].shift(1).rename("adr_t")
    tw_col      = tw_df[feature_col].rename("tw_t1")
    return pd.concat([adr_shifted, tw_col], axis=1, join="inner").dropna()


def _check_lookahead(adr_df: pd.DataFrame,
                     merged: pd.DataFrame,
                     n_samples: int = 50) -> Tuple[bool, float]:
    """
    驗證 merged["adr_t"][d] == adr_df["log_return"][d-1]。

    為效率取前 50 筆，回傳 (是否全部通過, 最大差距)。
    """
    if "log_return" not in adr_df.columns:
        return False, float("inf")

    adr_lr = adr_df["log_return"].dropna()
    if len(adr_lr) < 2 or len(merged) == 0:
        return False, float("inf")

    max_diff = 0.0
    checked  = 0
    for d in merged.index[:n_samples]:
        if d not in adr_lr.index:
            continue
        pos = adr_lr.index.get_loc(d)
        if pos == 0:
            continue
        expected = float(adr_lr.iloc[pos - 1])
        actual   = float(merged.loc[d, "adr_t"])
        diff     = abs(actual - expected)
        max_diff = max(max_diff, diff)
        checked += 1

    if checked == 0:
        return False, float("inf")
    return max_diff < 1e-9, max_diff


def _format_date_list(dates: pd.DatetimeIndex, max_show: int = 5) -> List[str]:
    """把 DatetimeIndex 轉成字串清單，超過 max_show 只顯示頭尾。"""
    if len(dates) == 0:
        return []
    if len(dates) <= max_show:
        return [d.strftime("%Y-%m-%d") for d in dates]
    head = [d.strftime("%Y-%m-%d") for d in dates[:3]]
    tail = [d.strftime("%Y-%m-%d") for d in dates[-2:]]
    return head + [f"... ({len(dates) - 5} more) ..."] + tail


# ════════════════════════════════════════════════════════════
# 單一配對審計
# ════════════════════════════════════════════════════════════

def audit_pair(adr_ticker: str,
               tw_code:    str,
               adr_dir:    Path,
               tw_dir:     Path) -> PairAuditResult:
    """對單一 ADR-TW 配對執行完整審計。"""

    r = PairAuditResult(adr_ticker=adr_ticker, tw_code=tw_code)

    adr_df = _load_csv_safely(adr_dir / f"{adr_ticker}.csv")
    tw_df  = _load_csv_safely(tw_dir  / f"{tw_code}.csv")

    r.adr_exists = adr_df is not None
    r.tw_exists  = tw_df  is not None

    if not r.adr_exists:
        r.status = "SKIP"
        r.notes.append(f"ADR 檔案不存在：{adr_dir / f'{adr_ticker}.csv'}")
        return r
    if not r.tw_exists:
        r.status = "SKIP"
        r.notes.append(f"TW 檔案不存在：{tw_dir / f'{tw_code}.csv'}")
        return r

    # ── 原始天數 ─────────────────────────────────────────────
    r.adr_days  = len(adr_df)
    r.tw_days   = len(tw_df)
    r.adr_start = adr_df.index.min()
    r.adr_end   = adr_df.index.max()
    r.tw_start  = tw_df.index.min()
    r.tw_end    = tw_df.index.max()

    # ── 集合差異 ─────────────────────────────────────────────
    adr_set    = set(adr_df.index)
    tw_set     = set(tw_df.index)
    common     = adr_set & tw_set
    only_adr   = sorted(adr_set - tw_set)
    only_tw    = sorted(tw_set - adr_set)

    r.common_days        = len(common)
    r.only_in_adr        = len(only_adr)
    r.only_in_tw         = len(only_tw)
    r.only_in_adr_dates  = _format_date_list(pd.DatetimeIndex(only_adr))
    r.only_in_tw_dates   = _format_date_list(pd.DatetimeIndex(only_tw))

    # ── ADR(t) → TW(t+1) 對齊 ────────────────────────────────
    if "log_return" not in adr_df.columns or "log_return" not in tw_df.columns:
        r.status = "FAIL"
        r.notes.append("缺少 log_return 欄位，請先執行 pipeline.py")
        return r

    merged = _align_adr_to_tw(adr_df, tw_df)
    r.aligned_samples = len(merged)
    if len(merged) > 0:
        r.aligned_start = merged.index.min()
        r.aligned_end   = merged.index.max()

    # ── Look-ahead sanity check ──────────────────────────────
    ok, max_diff = _check_lookahead(adr_df, merged)
    r.lookahead_check_pass = ok
    r.lookahead_max_diff   = max_diff

    # ── 補值統計（若 pipeline 有輸出追溯欄位）─────────────────
    for src_df, prefix in [(adr_df, "adr"), (tw_df, "tw")]:
        if "is_imputed" in src_df.columns:
            setattr(r, f"{prefix}_imputed_rows",
                    int(src_df["is_imputed"].fillna(False).sum()))
        if "is_long_gap" in src_df.columns:
            setattr(r, f"{prefix}_long_gap_rows",
                    int(src_df["is_long_gap"].fillna(False).sum()))

    # ── 整體判定 ─────────────────────────────────────────────
    r.status = "PASS"

    if r.common_days < MIN_COMMON_DAYS:
        r.status = "WARN"
        r.notes.append(
            f"共同交易日 {r.common_days} < {MIN_COMMON_DAYS}，"
            f"訓練樣本可能偏少"
        )

    if r.aligned_samples < MIN_ALIGNED_SAMPLES:
        r.status = "WARN"
        r.notes.append(
            f"對齊後樣本 {r.aligned_samples} < {MIN_ALIGNED_SAMPLES}"
        )

    if not r.lookahead_check_pass:
        r.status = "FAIL"
        r.notes.append(
            f"Look-ahead 檢查失敗（最大差距 {r.lookahead_max_diff:.2e}）"
        )

    # 起訖日漂移檢查
    if r.adr_start and r.tw_start:
        drift = abs((r.adr_start - r.tw_start).days)
        if drift > MAX_DATE_DRIFT_DAYS:
            r.notes.append(
                f"起始日相差 {drift} 天（ADR={r.adr_start.date()}, "
                f"TW={r.tw_start.date()}），請確認下載範圍一致"
            )

    if r.adr_long_gap_rows > 0 or r.tw_long_gap_rows > 0:
        r.notes.append(
            f"含長缺口列（ADR={r.adr_long_gap_rows}, TW={r.tw_long_gap_rows}），"
            f"訓練時建議依政策 §8.2 剔除"
        )

    return r


# ════════════════════════════════════════════════════════════
# 報告輸出
# ════════════════════════════════════════════════════════════

def print_terminal_summary(results: List[PairAuditResult]) -> None:
    """終端機表格摘要。"""
    print("\n" + "=" * 95)
    print("ADR-TW 對齊審計報告")
    print("=" * 95)

    header = (
        f"{'狀態':<5}{'ADR':<8}{'TW':<7}"
        f"{'ADR天':>7}{'TW天':>7}{'共同':>7}"
        f"{'僅ADR':>7}{'僅TW':>7}{'對齊樣本':>10}"
        f"{'LookAhead':>11}"
    )
    print(header)
    print("-" * 95)

    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "—"}
    for r in results:
        la = "✓" if r.lookahead_check_pass else (
             "—" if r.status == "SKIP" else "✗")
        print(
            f"{icon.get(r.status, '?'):<5}"
            f"{r.adr_ticker:<8}{r.tw_code:<7}"
            f"{r.adr_days:>7}{r.tw_days:>7}{r.common_days:>7}"
            f"{r.only_in_adr:>7}{r.only_in_tw:>7}{r.aligned_samples:>10}"
            f"{la:>11}"
        )

    # 統計
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    print("-" * 95)
    print(
        f"  總計：{len(results)} 組  "
        f"PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}  SKIP={n_skip}"
    )

    # 印出 notes
    has_notes = any(r.notes for r in results)
    if has_notes:
        print("\n備註：")
        for r in results:
            if r.notes:
                print(f"  [{r.adr_ticker}→{r.tw_code}]")
                for n in r.notes:
                    print(f"    · {n}")


def write_csv_report(results: List[PairAuditResult], path: Path) -> None:
    """寫出細項 CSV 報告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        d = asdict(r)
        # 將 list / Timestamp 轉成可序列化格式
        d["only_in_adr_dates"] = "; ".join(d["only_in_adr_dates"])
        d["only_in_tw_dates"]  = "; ".join(d["only_in_tw_dates"])
        d["notes"]             = "; ".join(d["notes"])
        for k in ("adr_start", "adr_end", "tw_start", "tw_end",
                  "aligned_start", "aligned_end"):
            v = d[k]
            d[k] = v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else ""
        rows.append(d)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_markdown_report(results: List[PairAuditResult], path: Path) -> None:
    """寫出可貼進 PM 文件的 Markdown 摘要。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# ADR-TW 對齊審計報告\n")
    lines.append(f"> 自動生成於 `alignment_audit.py`\n")

    # 摘要表
    lines.append("## 一、總覽\n")
    lines.append(
        "| 狀態 | ADR | TW | ADR天數 | TW天數 | 共同 | 僅ADR | 僅TW "
        "| 對齊樣本 | LookAhead |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for r in results:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "—"}.get(r.status, "?")
        la = "✓" if r.lookahead_check_pass else ("—" if r.status == "SKIP" else "✗")
        lines.append(
            f"| {icon} | {r.adr_ticker} | {r.tw_code} | "
            f"{r.adr_days} | {r.tw_days} | {r.common_days} | "
            f"{r.only_in_adr} | {r.only_in_tw} | {r.aligned_samples} | {la} |"
        )

    # 統計
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    lines.append(f"\n**統計**：PASS={n_pass}　WARN={n_warn}　FAIL={n_fail}　SKIP={n_skip}\n")

    # 細項
    lines.append("## 二、配對細項\n")
    for r in results:
        lines.append(f"### {r.adr_ticker} → {r.tw_code}　[{r.status}]\n")
        if r.status == "SKIP":
            lines.append(f"- 略過：{'; '.join(r.notes)}\n")
            continue
        lines.append(
            f"- ADR：{r.adr_days} 筆　"
            f"({r.adr_start.date() if r.adr_start else '?'} ~ "
            f"{r.adr_end.date() if r.adr_end else '?'})"
        )
        lines.append(
            f"- TW：{r.tw_days} 筆　"
            f"({r.tw_start.date() if r.tw_start else '?'} ~ "
            f"{r.tw_end.date() if r.tw_end else '?'})"
        )
        lines.append(f"- 共同交易日：{r.common_days}")
        lines.append(
            f"- 對齊後樣本（ADR(t)→TW(t+1)）：{r.aligned_samples}"
        )
        lines.append(
            f"- Look-ahead sanity check："
            f"{'通過' if r.lookahead_check_pass else '失敗'}"
            f"（最大差距 {r.lookahead_max_diff:.2e}）"
        )
        if r.only_in_adr_dates:
            lines.append(f"- 僅在 ADR 出現的日期樣本：{', '.join(r.only_in_adr_dates)}")
        if r.only_in_tw_dates:
            lines.append(f"- 僅在 TW 出現的日期樣本：{', '.join(r.only_in_tw_dates)}")
        if r.adr_imputed_rows or r.tw_imputed_rows:
            lines.append(
                f"- 補值列：ADR={r.adr_imputed_rows}, TW={r.tw_imputed_rows}"
            )
        if r.notes:
            lines.append(f"- 備註：")
            for n in r.notes:
                lines.append(f"  - {n}")
        lines.append("")

    # 後續建議
    lines.append("## 三、後續行動建議\n")
    if n_fail > 0:
        lines.append(f"- ❌ 有 {n_fail} 組配對 FAIL，**進入 features.py 前必須修正**。")
    if n_warn > 0:
        lines.append(f"- ⚠️ 有 {n_warn} 組配對樣本偏少，建議檢視是否從 PAIR_MAP 暫時移除。")
    if n_skip > 0:
        lines.append(f"- — 有 {n_skip} 組配對檔案缺漏，請補齊資料下載。")
    if n_fail == 0 and n_warn == 0 and n_skip == 0:
        lines.append("- ✅ 全部通過，可進入 features.py 開發。")

    path.write_text("\n".join(lines), encoding="utf-8")


# ════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ADR-TW 對齊審計")
    parser.add_argument("--adr-dir", default="data/processed/adr",
                        help="ADR 處理後 CSV 目錄")
    parser.add_argument("--tw-dir",  default="data/processed/tw",
                        help="TW 處理後 CSV 目錄")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="報告輸出目錄")
    args = parser.parse_args()

    adr_dir = Path(args.adr_dir)
    tw_dir  = Path(args.tw_dir)
    out_dir = Path(args.out_dir)

    print(f"\n[alignment_audit] ADR={adr_dir}  TW={tw_dir}")
    print(f"[alignment_audit] 配對數量：{len(PAIR_MAP)}")

    results: List[PairAuditResult] = []
    for adr_ticker, tw_code in PAIR_MAP.items():
        r = audit_pair(adr_ticker, tw_code, adr_dir, tw_dir)
        results.append(r)

    # 輸出報告
    print_terminal_summary(results)

    csv_path = out_dir / "alignment_audit.csv"
    md_path  = out_dir / "alignment_audit.md"
    write_csv_report(results, csv_path)
    write_markdown_report(results, md_path)
    print(f"\n細項 CSV → {csv_path}")
    print(f"摘要 MD  → {md_path}")

    # 回傳非零 exit code 若有 FAIL
    n_fail = sum(1 for r in results if r.status == "FAIL")
    if n_fail > 0:
        print(f"\n❌ 有 {n_fail} 組配對審計失敗。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())