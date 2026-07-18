"""
m8_pairwise_tests.py — 路線 A 實驗 2：全配對顯著性檢定

對 Part D 的 7 個 model family（各 3 seeds）做兩兩比較：
    同 seed 同測試日的 daily IC 配對差，pool 三顆 seed（n ≈ 735 day-pairs），
    one-sample t-test 對 0，Holm 校正 21 個比較。

產出 docs/m8_pairwise_tests.md：
    - 配對差矩陣（mean daily ΔIC）
    - p 值矩陣與存活配對清單
"""

from __future__ import annotations

import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "Adv-ALSTM":     ["opt_p20_adv_alstm", "opt_p55_advalstm_s7", "opt_p56_advalstm_s123"],
    "MAN-SF":        ["opt_p22_man_sf", "opt_p57_mansf_s7", "opt_p58_mansf_s123"],
    "DeltaLag":      ["opt_p29_dl_lr5e4_pat15", "opt_p59_dl5e4_s7", "opt_p60_dl5e4_s123"],
    "HGT":           ["opt_p23_hgt", "opt_p61_hgt_s7", "opt_p62_hgt_s123"],
    "MEIG-core":     ["opt_p33_meig_lr5e4_pat15", "opt_p63_meig5e4_s7", "opt_p64_meig5e4_s123"],
    "MAGNET@1e-3":   ["opt_p2_variance_penalty", "opt_p37_magnet_seed7", "opt_p38_magnet_seed123"],
    "MAGNET@5e-4":   ["opt_p46_raw_lr5e4_s42", "opt_p47_raw_lr5e4_s7", "opt_p48_raw_lr5e4_s123"],
}


def daily_ic_series(tag: str) -> pd.Series:
    path = sorted(glob.glob(str(ROOT / "runs" / f"*{tag}" / "predictions" / "test_predictions.csv")))[-1]
    df = pd.read_csv(path)
    out = {}
    for d, g in df.groupby("target_date"):
        if g["y_hat"].std() > 0 and g["y"].std() > 0:
            out[d] = float(np.corrcoef(g["y_hat"], g["y"])[0, 1])
    return pd.Series(out)


def family_series(tags: list[str]) -> list[pd.Series]:
    return [daily_ic_series(t) for t in tags]


def main() -> None:
    series = {fam: family_series(tags) for fam, tags in FAMILIES.items()}
    fams = list(FAMILIES)

    rows = []
    for a, b in combinations(fams, 2):
        diffs = []
        for sa, sb in zip(series[a], series[b]):     # 同 seed 配對
            idx = sa.index.intersection(sb.index)    # 同測試日配對
            diffs.append(sa[idx] - sb[idx])
        pooled = pd.concat(diffs)
        t, p = ttest_1samp(pooled, 0.0)
        rows.append({"A": a, "B": b, "mean_dIC": pooled.mean(),
                     "t": t, "p": p, "n": len(pooled)})

    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    # Holm 校正
    m = len(df)
    df["p_holm"] = [min(1.0, p * (m - i)) for i, p in enumerate(df["p"])]
    df["sig@0.05"] = df["p_holm"] < 0.05

    lines = [
        "# M8 Pairwise Significance Tests (Route A Experiment 2)",
        "",
        "Paired daily test-day IC differences, pooled over 3 matched seeds "
        "(n ≈ 735 day-pairs per comparison). One-sample t vs 0, "
        "Holm correction over 21 comparisons.",
        "",
        "| # | A − B | mean ΔIC | t | p (raw) | p (Holm) | sig@0.05 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for i, r in df.iterrows():
        lines.append(
            f"| {i+1} | {r['A']} − {r['B']} | {r['mean_dIC']:+.4f} | "
            f"{r['t']:+.2f} | {r['p']:.4f} | {r['p_holm']:.4f} | "
            f"{'YES' if r['sig@0.05'] else 'no'} |"
        )
    lines.append("")
    n_sig = int(df["sig@0.05"].sum())
    lines.append(f"**Surviving comparisons after Holm correction: {n_sig} / {m}.**")
    lines.append("")

    out = ROOT / "docs" / "m8_pairwise_tests.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
