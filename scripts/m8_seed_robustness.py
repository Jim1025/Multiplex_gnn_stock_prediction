"""
m8_seed_robustness.py — M8 multi-seed robustness 表產出器

背景：
    M8 Step 1-2 發現 (a) 弱連結 beta 全程趴在 0（機制 null result），
    (b) 三個函數上近乎相同的模型（beta≈0）test_IC 卻是 0.072 / 0.056 / 0.002
    —— best-val checkpoint selection 在小樣本 + val IC 振盪下是高變異估計器。
    本腳本聚合 3 model × 3 seed 矩陣，報 mean ± std，量化該變異。

執行：
    python scripts/m8_seed_robustness.py
    python scripts/m8_seed_robustness.py --out docs/m8_seed_robustness.md
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "runs" / "INDEX.csv"

# family → {seed: tag}
MATRIX = {
    "MAGNET": {
        42:  "opt_p2_variance_penalty",
        7:   "opt_p37_magnet_seed7",
        123: "opt_p38_magnet_seed123",
    },
    "MAGNET + weak (free x42)": {
        42:  "opt_p35_magnet_weak_free",
        7:   "opt_p39_wfree_seed7",
        123: "opt_p40_wfree_seed123",
    },
    "MAGNET + weak (industry x12)": {
        42:  "opt_p36_magnet_weak_industry",
        7:   "opt_p41_wind_seed7",
        123: "opt_p42_wind_seed123",
    },
}

# M8 Part C: lr 5e-4 協議下的 A12 因果對比
MATRIX_A12 = {
    "MAGNET (full) @ lr 5e-4": {
        42:  "opt_p46_raw_lr5e4_s42",
        7:   "opt_p47_raw_lr5e4_s7",
        123: "opt_p48_raw_lr5e4_s123",
    },
    "MAGNET (no A12) @ lr 5e-4": {
        42:  "opt_p52_noa12_lr5e4_s42",
        7:   "opt_p53_noa12_lr5e4_s7",
        123: "opt_p54_noa12_lr5e4_s123",
    },
}

# M8 Part E（路線 A 實驗 4）: 受控 replicates——同 config 同 seed 重複跑
# p46 為原 run；p66 為 Figure 1 instrumented run（訓練數學不變）；p67-p69 為受控補跑
MATRIX_REPLICATES = {
    "MAGNET seed 42 @ lr 5e-4 (5 replicates)": {
        "r1 (p46)":  "opt_p46_raw_lr5e4_s42",
        "r2 (p66)":  "opt_p66_fig1_lr5e4",
        "r3 (p67)":  "opt_p67_rep_s42_a",
        "r4 (p68)":  "opt_p68_rep_s42_b",
        "r5 (p69)":  "opt_p69_rep_s42_c",
    },
}

# M8 Part D（路線 A 實驗 1）: top baselines × 3 seeds @ 各自 best-of-grid config
# 檢驗「發表風格的單 seed 數字能否跨 seed 複現」——誠實版 Table 3 的資料來源
MATRIX_BASELINES = {
    "Adv-ALSTM (default cfg)": {
        42:  "opt_p20_adv_alstm",
        7:   "opt_p55_advalstm_s7",
        123: "opt_p56_advalstm_s123",
    },
    "MAN-SF no-text (default cfg)": {
        42:  "opt_p22_man_sf",
        7:   "opt_p57_mansf_s7",
        123: "opt_p58_mansf_s123",
    },
    "DeltaLag (lr 5e-4)": {
        42:  "opt_p29_dl_lr5e4_pat15",
        7:   "opt_p59_dl5e4_s7",
        123: "opt_p60_dl5e4_s123",
    },
    "HGT (lr 1e-3)": {
        42:  "opt_p23_hgt",
        7:   "opt_p61_hgt_s7",
        123: "opt_p62_hgt_s123",
    },
    "MEIG-core (lr 5e-4)": {
        42:  "opt_p33_meig_lr5e4_pat15",
        7:   "opt_p63_meig5e4_s7",
        123: "opt_p64_meig5e4_s123",
    },
    "MAGNET (lr 1e-3, original protocol)": {
        42:  "opt_p2_variance_penalty",
        7:   "opt_p37_magnet_seed7",
        123: "opt_p38_magnet_seed123",
    },
    "MAGNET (lr 5e-4)": {
        42:  "opt_p46_raw_lr5e4_s42",
        7:   "opt_p47_raw_lr5e4_s7",
        123: "opt_p48_raw_lr5e4_s123",
    },
}

# M8 route 1+2 穩定化 factorial（全部 architecture=magnet）：
#   選點規則 {raw best-val, 3-epoch trailing MA} × lr {1e-3, 5e-4} × 3 seeds
MATRIX_STABILIZE = {
    "raw + lr 1e-3 (current protocol)": {
        42:  "opt_p2_variance_penalty",
        7:   "opt_p37_magnet_seed7",
        123: "opt_p38_magnet_seed123",
    },
    "MA-3 + lr 1e-3": {
        42:  "opt_p43_ma_lr1e3_s42",
        7:   "opt_p44_ma_lr1e3_s7",
        123: "opt_p45_ma_lr1e3_s123",
    },
    "raw + lr 5e-4": {
        42:  "opt_p46_raw_lr5e4_s42",
        7:   "opt_p47_raw_lr5e4_s7",
        123: "opt_p48_raw_lr5e4_s123",
    },
    "MA-3 + lr 5e-4": {
        42:  "opt_p49_ma_lr5e4_s42",
        7:   "opt_p50_ma_lr5e4_s7",
        123: "opt_p51_ma_lr5e4_s123",
    },
}

METRICS = ["best_val_IC", "test_IC", "test_RankIC", "test_ICIR"]


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_index() -> dict[str, dict]:
    by_tag: dict[str, dict] = {}
    for row in csv.DictReader(open(INDEX)):
        by_tag[row["tag"]] = row      # 同 tag 重跑取最後一筆
    return by_tag


def _detail_rows(matrix: dict, by_tag: dict) -> list[str]:
    lines = [
        "| Config | Seed | best\\_ep | val\\_IC | test\\_IC | test\\_RankIC | val→test gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, seeds in matrix.items():
        for seed, tag in seeds.items():
            r = by_tag.get(tag, {})
            v, t = _f(r.get("best_val_IC")), _f(r.get("test_IC"))
            lines.append(
                f"| {family} | {seed} | {r.get('best_epoch', '?')} | "
                f"{v:.4f} | {t:.4f} | {_f(r.get('test_RankIC')):.4f} | {v - t:+.4f} |"
            )
    return lines


def _aggregate_rows(matrix: dict, by_tag: dict, emph_key: str | None = None) -> list[str]:
    lines = [
        "| Config | val\\_IC | **test\\_IC** | test\\_RankIC | test\\_IC range |",
        "|---|---|---|---|---|",
    ]
    for family, seeds in matrix.items():
        vals = {m: [] for m in METRICS}
        for tag in seeds.values():
            r = by_tag.get(tag, {})
            for m in METRICS:
                x = _f(r.get(m))
                if x == x:
                    vals[m].append(x)

        def ms(m: str) -> str:
            xs = vals[m]
            if len(xs) < 2:
                return "n/a"
            return f"{statistics.mean(xs):.4f} ± {statistics.stdev(xs):.4f}"

        tics = vals["test_IC"]
        rng = f"[{min(tics):.4f}, {max(tics):.4f}]" if tics else "n/a"
        emph = "**" if family == emph_key else ""
        lines.append(
            f"| {emph}{family}{emph} | {ms('best_val_IC')} | **{ms('test_IC')}** | "
            f"{ms('test_RankIC')} | {rng} |"
        )
    return lines


def build_markdown(by_tag: dict) -> str:
    lines: list[str] = []
    lines.append("# M8 Multi-Seed Robustness")
    lines.append("")
    lines.append("Auto-generated by `scripts/m8_seed_robustness.py`. "
                 "Seeds: 42 / 7 / 123.")
    lines.append("")

    lines.append("## Part A — Weak-link ladder (3 models x 3 seeds, opt_p2 regime)")
    lines.append("")
    lines.append("### Per-run detail")
    lines.append("")
    lines.extend(_detail_rows(MATRIX, by_tag))
    lines.append("")
    lines.append("### Aggregate (mean ± std over seeds)")
    lines.append("")
    lines.extend(_aggregate_rows(MATRIX, by_tag, emph_key="MAGNET"))
    lines.append("")

    lines.append("## Part B — Stabilization factorial "
                 "(MAGNET, selection rule x lr, 2x2 x 3 seeds)")
    lines.append("")
    lines.append("Route 1 = 3-epoch trailing moving-average checkpoint selection "
                 "(`--smooth-window 3`); Route 2 = lr 5e-4.")
    lines.append("")
    lines.append("### Per-run detail")
    lines.append("")
    lines.extend(_detail_rows(MATRIX_STABILIZE, by_tag))
    lines.append("")
    lines.append("### Aggregate (mean ± std over seeds)")
    lines.append("")
    lines.extend(_aggregate_rows(MATRIX_STABILIZE, by_tag))
    lines.append("")

    lines.append("## Part C — A12 causal ablation @ lr 5e-4 "
                 "(raw selection, 3 seeds)")
    lines.append("")
    lines.append("### Per-run detail")
    lines.append("")
    lines.extend(_detail_rows(MATRIX_A12, by_tag))
    lines.append("")
    lines.append("### Aggregate (mean ± std over seeds)")
    lines.append("")
    lines.extend(_aggregate_rows(MATRIX_A12, by_tag))
    lines.append("")

    lines.append("## Part E — Controlled replicates: same config, same seed, "
                 "repeated runs (Route A Experiment 4)")
    lines.append("")
    lines.append("MPS numerics are not bitwise deterministic; each run is an "
                 "independent draw regardless of seed. 'Seed' column = replicate id.")
    lines.append("")
    lines.append("### Per-run detail")
    lines.append("")
    lines.extend(_detail_rows(MATRIX_REPLICATES, by_tag))
    lines.append("")
    lines.append("### Aggregate (mean ± std over replicates)")
    lines.append("")
    lines.extend(_aggregate_rows(MATRIX_REPLICATES, by_tag))
    lines.append("")

    lines.append("## Part D — Honest Table 3: top baselines × 3 seeds "
                 "@ each model's best-of-grid config")
    lines.append("")
    lines.append("Tests whether published-style single-seed numbers replicate "
                 "across seeds. Seed-42 entries are the original M7 runs.")
    lines.append("")
    lines.append("### Per-run detail")
    lines.append("")
    lines.extend(_detail_rows(MATRIX_BASELINES, by_tag))
    lines.append("")
    lines.append("### Aggregate (mean ± std over seeds)")
    lines.append("")
    lines.extend(_aggregate_rows(MATRIX_BASELINES, by_tag))
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- All runs share the opt_p2 regime (lr=1e-3, wd=1e-3, patience=15, "
                 "CosineAnnealingWarmRestarts, early stop on val IC).")
    lines.append("- Weak-link variants: beta L1 lambda=1e-3; learned beta remained "
                 "~0 in all runs (max |beta| < 1e-3), so families are functionally "
                 "near-identical — the spread within and across families measures "
                 "trajectory + checkpoint-selection variance, not mechanism effects.")
    lines.append("- Selection instability context: per-epoch val IC oscillates "
                 "between roughly -0.07 and +0.08 in adjacent epochs while val loss "
                 "stays flat; best-val selection samples a peak of that oscillation.")
    lines.append("- Part C significance (paired daily IC, full vs no_A12, same seeds "
                 "and test days, n=735 day-pairs): mean Δ = +0.0077, t = 0.47, "
                 "p = 0.64. Seed-level paired deltas: +0.0211 / +0.0023 / -0.0005. "
                 "The A12 effect is statistically indistinguishable from zero under "
                 "the stable regime at this sample size; detecting a true Δ of "
                 "0.008 would require roughly 45+ seed-pairs.")
    lines.append("- MPS nondeterminism: identical seed + config reruns can diverge "
                 "under load (observed opt_p45 vs opt_p38 trajectory mismatch after "
                 "a mid-run machine sleep); seed alone does not pin the trajectory.")
    lines.append("- Part E revision of Part B: the 'lr 5e-4 stabilizes' conclusion "
                 "does not survive controlled replication. Five same-seed same-config "
                 "runs at lr 5e-4 give test_IC sigma ≈ 0.026 — about 4x the "
                 "across-seed sigma (0.007) measured from three single runs in "
                 "Part B, meaning that tight spread was itself a sampling artifact. "
                 "The RUN, not the seed, is the unit of variance; honest reporting "
                 "requires replicate-based mean ± SE per configuration. "
                 "Re-evaluating identical weights also shifts val IC by ~0.006 "
                 "(evaluation-level nondeterminism), enough to flip which epoch "
                 "best-val selection chooses.")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="docs/m8_seed_robustness.md")
    args = p.parse_args()

    md = build_markdown(load_index())
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
