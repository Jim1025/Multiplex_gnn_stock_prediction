"""
m6_stage0_summary.py — Stage 0 ablation 對照表 + Stage 1 路線建議

讀 runs/INDEX.csv 中 4 個 M6 ablation 標籤的 test_IC，依 M6 plan 的判讀矩陣
建議進入 Stage 1A（universe 擴展）或 Stage 1B（macro features）。

四個 tag:
    opt_p17_baseline_lstm     LSTM-only（無圖、無 ADR）
    opt_p18_baseline_tw_gnn   TW-only 單層 GNN（無 ADR）
    opt_p19_ablation_no_a12   MAGNET 但 A12 跨層訊號零化
    opt_p2_variance_penalty   完整 MAGNET (reference)

使用:
    python scripts/m6_stage0_summary.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "runs" / "INDEX.csv"

ABLATION_TAGS = {
    "lstm_only":  "opt_p17_baseline_lstm",
    "tw_gnn":     "opt_p18_baseline_tw_gnn",
    "magnet_no_a12": "opt_p19_ablation_no_a12",
    "magnet_full":   "opt_p2_variance_penalty",
}


def _load_index() -> dict[str, dict]:
    """讀最新 INDEX，依 tag 取最後一筆（同 tag 重跑時取最新）。"""
    if not INDEX.exists():
        raise SystemExit(f"找不到 {INDEX}")
    by_tag: dict[str, dict] = {}
    for row in csv.DictReader(open(INDEX)):
        by_tag[row["tag"]] = row
    return by_tag


def _f(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _row_str(label: str, tag: str, row: dict | None) -> str:
    if row is None:
        return f"  {label:18s}  [{tag}]  -- 缺少（尚未跑或被刪除）"
    return (
        f"  {label:18s}  [{tag}]  "
        f"test_IC={_f(row,'test_IC'):.4f}  "
        f"RankIC={_f(row,'test_RankIC'):.4f}  "
        f"ICIR={_f(row,'test_ICIR'):.4f}  "
        f"best_val_IC={_f(row,'best_val_IC'):.4f}  "
        f"epochs={row.get('n_epochs','?')}"
    )


def _suggest(rows: dict[str, dict | None]) -> str:
    full = _f(rows.get("magnet_full") or {}, "test_IC") if rows.get("magnet_full") else None
    tw   = _f(rows.get("tw_gnn")     or {}, "test_IC") if rows.get("tw_gnn")     else None
    noA  = _f(rows.get("magnet_no_a12") or {}, "test_IC") if rows.get("magnet_no_a12") else None
    lstm = _f(rows.get("lstm_only") or {}, "test_IC") if rows.get("lstm_only") else None

    notes: list[str] = []
    suggest: list[str] = []

    if full is not None and tw is not None:
        delta = full - tw
        notes.append(f"Δ(magnet - tw_gnn) = {delta:+.4f}")
        if delta >= 0.02:
            suggest.append("Stage 1A: ADR 層有實質貢獻 → 擴展 universe（取消 ASUUY 註解 + 加 5-7 對）")
        elif abs(delta) < 0.005:
            suggest.append("Stage 1B: ADR 層幾無貢獻 → 改加 macro/sentiment，放棄擴 universe")
        else:
            suggest.append("Stage 1A 偏小幅度（0.005 ≤ |Δ| < 0.02）：可擴 universe，但留意 ADR 訊號弱")

    if full is not None and noA is not None:
        delta = full - noA
        notes.append(f"Δ(magnet - magnet_no_a12) = {delta:+.4f}")
        if abs(delta) < 0.005:
            suggest.append("A12 跨層連線無用 → 架構可簡化（Stage 1 同時移除 fusion）")

    if lstm is not None and tw is not None and lstm > tw:
        notes.append(f"警示：lstm_only ({lstm:+.4f}) > tw_gnn ({tw:+.4f}) → 圖反而有害")
        suggest.append("先回頭檢視 graph_builder 的相關係數閾值 / edge dropout")

    if not suggest:
        suggest.append("等四個 tag 都齊 → 才能依判讀矩陣下決策")

    return "\n".join(["  - " + s for s in notes + suggest])


def main() -> None:
    by_tag = _load_index()
    rows = {k: by_tag.get(t) for k, t in ABLATION_TAGS.items()}

    print("=" * 78)
    print("M6 Stage 0 — Ablation table")
    print("=" * 78)
    print(_row_str("LSTM-only",        ABLATION_TAGS["lstm_only"],      rows["lstm_only"]))
    print(_row_str("TW-only GNN",      ABLATION_TAGS["tw_gnn"],         rows["tw_gnn"]))
    print(_row_str("MAGNET (no A12)",  ABLATION_TAGS["magnet_no_a12"],  rows["magnet_no_a12"]))
    print(_row_str("MAGNET (full)",    ABLATION_TAGS["magnet_full"],    rows["magnet_full"]))
    print()
    print("=" * 78)
    print("判讀 + Stage 1 路線建議")
    print("=" * 78)
    print(_suggest(rows))


if __name__ == "__main__":
    main()
