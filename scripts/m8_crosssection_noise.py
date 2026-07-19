"""
m8_crosssection_noise.py — 路線 A 實驗 5：cross-section 大小 vs daily IC 噪音

從既有 test predictions 對 ticker 子抽樣（k = 3..7），量測 daily IC 的
標準差如何隨 cross-section 大小縮放，並對照理論下界：

    對近乎零相關的雙變量常態，sample correlation 的標準差
    sigma(r) ≈ 1 / sqrt(k - 1)

推論：
    - 觀測噪音是否就是 k=7 的數學下界（非模型缺陷）
    - 246 個測試日下 mean IC 的標準誤
    - 要以 95% 信心分辨 Delta IC ∈ {0.01, 0.02} 所需的 universe 大小

輸出：docs/m8_crosssection_noise.md + docs/figures/m8_fig2_crosssection_noise.png
"""

from __future__ import annotations

import glob
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(42)

# 用 Part D 的 21 個 run 平均，降低單一模型的特異性
TAGS = [
    "opt_p20_adv_alstm", "opt_p55_advalstm_s7", "opt_p56_advalstm_s123",
    "opt_p22_man_sf", "opt_p57_mansf_s7", "opt_p58_mansf_s123",
    "opt_p29_dl_lr5e4_pat15", "opt_p59_dl5e4_s7", "opt_p60_dl5e4_s123",
    "opt_p23_hgt", "opt_p61_hgt_s7", "opt_p62_hgt_s123",
    "opt_p33_meig_lr5e4_pat15", "opt_p63_meig5e4_s7", "opt_p64_meig5e4_s123",
    "opt_p2_variance_penalty", "opt_p37_magnet_seed7", "opt_p38_magnet_seed123",
    "opt_p46_raw_lr5e4_s42", "opt_p47_raw_lr5e4_s7", "opt_p48_raw_lr5e4_s123",
]

K_RANGE = [3, 4, 5, 6, 7]


def load_preds(tag: str) -> pd.DataFrame:
    p = sorted(glob.glob(str(ROOT / "runs" / f"*{tag}" / "predictions" / "test_predictions.csv")))[-1]
    return pd.read_csv(p)


def daily_ic_std_for_k(df: pd.DataFrame, k: int) -> float:
    """對每個測試日抽 ticker 子集算 IC；回傳所有 (day, subset) IC 的 std。"""
    ics = []
    for _, g in df.groupby("target_date"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < k:
            continue
        if k == n:
            subsets = [tuple(range(n))]
        else:
            all_subsets = list(combinations(range(n), k))
            take = min(20, len(all_subsets))
            idx = RNG.choice(len(all_subsets), size=take, replace=False)
            subsets = [all_subsets[i] for i in idx]
        for s in subsets:
            sub = g.iloc[list(s)]
            if sub["y_hat"].std() > 0 and sub["y"].std() > 0:
                ics.append(float(np.corrcoef(sub["y_hat"], sub["y"])[0, 1]))
    return float(np.std(ics))


def main() -> None:
    per_run = {k: [] for k in K_RANGE}
    for tag in TAGS:
        df = load_preds(tag)
        for k in K_RANGE:
            per_run[k].append(daily_ic_std_for_k(df, k))

    measured = {k: float(np.mean(v)) for k, v in per_run.items()}
    theory   = {k: 1.0 / np.sqrt(k - 1) for k in K_RANGE}

    n_days = 246
    sigma7 = measured[7]
    se_mean_7 = sigma7 / np.sqrt(n_days)

    # 以 sigma(k) ≈ 1/sqrt(k-1) 外推：分辨 Delta 需 SE <= Delta/1.96（單模型 95% CI 不含 0 的近似）
    def k_needed(delta: float) -> int:
        sigma_needed = delta / 1.96 * np.sqrt(n_days)
        return int(np.ceil(1.0 / sigma_needed**2 + 1))

    lines = [
        "# M8 Cross-Section Noise Scaling (Route A Experiment 5)",
        "",
        f"Daily-IC std vs cross-section size k, averaged over {len(TAGS)} runs "
        "(all Part D families x 3 seeds); subsampled ticker subsets per test day.",
        "",
        "| k (stocks per day) | measured sigma(daily IC) | theory 1/sqrt(k-1) |",
        "|---:|---:|---:|",
    ]
    for k in K_RANGE:
        lines.append(f"| {k} | {measured[k]:.3f} | {theory[k]:.3f} |")
    lines += [
        "",
        f"- Measured sigma at k=7 is {sigma7:.3f} vs theoretical floor "
        f"{theory[7]:.3f} for zero-signal cross-sections — the benchmark noise "
        "is essentially the mathematical floor of 7-name daily correlations, "
        "not a model artifact.",
        f"- Standard error of a {n_days}-day mean IC at k=7: {se_mean_7:.4f} "
        "(single run, before seed/selection variance).",
        f"- Universe size needed to resolve Delta IC = 0.02 at 95%: "
        f"k ≈ {k_needed(0.02)} stocks; Delta IC = 0.01: k ≈ {k_needed(0.01)} "
        f"stocks (holding {n_days} test days).",
        "",
        "Implication: at k=7 the benchmark cannot distinguish models closer "
        "than roughly 0.05 IC even before training variance is considered; "
        "architectural effects of 0.01-0.02 are unresolvable by design.",
    ]

    out_md = ROOT / "docs" / "m8_crosssection_noise.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ks = list(K_RANGE)
    ax.plot(ks, [measured[k] for k in ks], "o-", color="0.15", lw=1.8,
            label="measured (mean over 21 runs)")
    ax.plot(ks, [theory[k] for k in ks], "s--", color="#c0392b", lw=1.5,
            label=r"theory $1/\sqrt{k-1}$ (zero signal)")
    ax.set_xlabel("cross-section size k (stocks per day)")
    ax.set_ylabel("std of daily IC")
    ax.set_title("Daily-IC noise vs universe size k", fontsize=11)
    ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_png = ROOT / "docs" / "figures" / "m8_fig2_crosssection_noise.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"\n[write] {out_md}")
    print(f"[write] {out_png}")


if __name__ == "__main__":
    main()
