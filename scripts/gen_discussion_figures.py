"""
gen_discussion_figures.py — 和教授討論評估章節用的示意圖（英文，2:1）

四張圖，對應討論出的議題組合：
    eval_plan_info_source.svg      資訊來源（只用台股 vs 台股 + 美股）
    eval_plan_info_lag.svg         資訊時效（美股輸入落後 1–4 天）
    eval_plan_market_state.svg     市場狀態（美股前一晚等權報酬絕對值的五分位）
    eval_plan_ablation.svg         消融（原 2x2 圖只留第一折 IC）

真實數字取自 docs/results_table.md 與 proposal §83（第一折 IC，種子平均）。
標為 SIM 的是假設數據，只用來示意圖的形式，不是結果；圖上以空心或斜線標出。

    .venv/bin/python scripts/gen_discussion_figures.py [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "svg.fonttype": "none"})

# 每個方法固定一種顏色與一種標記（折線圖用不同圖形區分）
STYLE = {
    "MAGNET": dict(color="#b5541c", marker="o", ls="-"),
    "MAGNET-single market": dict(color="#6b7280", marker="s", ls="--"),
    "LASSO [24]": dict(color="#1f5f99", marker="^", ls="-."),
    "MEIG [1]": dict(color="#2f7d4f", marker="D", ls=":"),
}
SIM_NOTE = "Hollow markers / hatched bars = simulated data (illustration only)"


def _finish(fig, ax, path, title, note=SIM_NOTE):
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    if note:
        fig.text(0.01, 0.01, note, fontsize=10, color="#6b7280")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def info_source(out):
    # (值, 是否為模擬)；None = 不存在（MEIG 原生跨市場，不做只用台股的版本）
    data = {
        "MAGNET": [(0.0250, False), (0.1127, False)],
        "LASSO [24]": [(0.0200, True), (0.1042, False)],
        "MEIG [1]": [None, (0.0061, False)],
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.36
    for i, (m, vals) in enumerate(data.items()):
        for k, v in enumerate(vals):
            x = i + (k - 0.5) * w
            if v is None:
                ax.text(x, 0.004, "n/a\n(native\ncross-market)", ha="center", fontsize=9, color="#6b7280")
                continue
            val, sim = v
            ax.bar(x, val, w * 0.95, color="white" if k == 0 else STYLE[m]["color"],
                   edgecolor=STYLE[m]["color"], hatch="//" if sim else None, lw=1.5)
            ax.text(x, val + 0.002, f"{val:.3f}" + (" (sim)" if sim else ""), ha="center", fontsize=10)
    ax.set_xticks(range(len(data)), list(data))
    ax.set_ylabel("Test IC (fold 1)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="white", ec="k"),
                       plt.Rectangle((0, 0), 1, 1, fc="#9ca3af", ec="k")],
              labels=["TW only", "TW + US"], frameon=False, loc="upper right")
    _finish(fig, ax, out / "eval_plan_info_source.svg", "Effect of information source")


def info_lag(out):
    lags = [1, 2, 3, 4]
    # 落後 1 天 = 真實值；2–4 天為模擬。單市場不用美股，與落後天數無關，所以是真實值的水平線
    data = {
        "MAGNET": [0.1127, 0.060, 0.042, 0.035],
        "MAGNET-single market": [0.0250] * 4,
        "LASSO [24]": [0.1042, 0.050, 0.030, 0.025],
        "MEIG [1]": [0.0061, 0.004, 0.003, 0.003],
    }
    real = {"MAGNET": [0], "MAGNET-single market": [0, 1, 2, 3], "LASSO [24]": [0], "MEIG [1]": [0]}
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, ys in data.items():
        s = STYLE[m]
        ax.plot(lags, ys, color=s["color"], ls=s["ls"], lw=2, marker=s["marker"], ms=9, mfc="white", mew=2, label=m)
        for j, y in enumerate(ys):
            filled = j in real[m]
            ax.plot(lags[j], y, marker=s["marker"], ms=9, color=s["color"],
                    mfc=s["color"] if filled else "white", mew=2)
    ax.set_xticks(lags, ["t−1\n(default)", "t−2", "t−3", "t−4"])
    ax.set_xlabel("Lag of US input (trading days)")
    ax.set_ylabel("Test IC (fold 1)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _finish(fig, ax, out / "eval_plan_info_lag.svg", "Effect of information timeliness")


def market_state(out):
    qs = ["Q1\n(calm)", "Q2", "Q3", "Q4", "Q5\n(large move)"]
    # 全部為模擬
    data = {
        "MAGNET": [0.085, 0.060, 0.120, 0.150, 0.200],
        "MAGNET-single market": [0.040, 0.035, 0.040, 0.045, 0.045],
        "LASSO [24]": [0.045, 0.050, 0.115, 0.140, 0.195],
        "MEIG [1]": [0.002, 0.000, 0.005, 0.004, 0.006],
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(qs))
    for m, ys in data.items():
        s = STYLE[m]
        ax.plot(x, ys, color=s["color"], ls=s["ls"], lw=2, marker=s["marker"], ms=9,
                mfc="white", mew=2, label=m)
    ax.set_xticks(x, qs)
    ax.set_xlabel("Size of last night's US move: quintile of |equal-weighted US return|")
    ax.set_ylabel("Mean daily IC")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _finish(fig, ax, out / "eval_plan_market_state.svg", "Effect of market state", note="All values simulated (illustration only)")


def ablation(out):
    # 真實數字：proposal §83.3、§83.4、scripts/sa_us_graph_eval.py（第一折 IC，種子平均 ± 種子 SD）
    arms = [("MAGNET", 0.1127, 0.0039, None),
            ("w/o relational\nTW graph", 0.1090, 0.0051, "//"),
            ("w/o US\nintra-layer graph", 0.1027, 0.0034, ".."),
            ("MAGNET-\nsingle market", 0.0250, 0.0021, "--")]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (lab, v, sd, h) in enumerate(arms):
        ax.bar(i, v, 0.6, yerr=sd, capsize=5, color="#b5541c" if h is None else "white",
               edgecolor="#b5541c", hatch=h, lw=1.5)
        ax.text(i, v + sd + 0.003, f"{v:.3f}", ha="center", fontsize=11)
    ax.set_xticks(range(len(arms)), [a[0] for a in arms])
    ax.set_ylabel("Test IC (fold 1, mean ± seed SD, 10 seeds)")
    _finish(fig, ax, out / "eval_plan_ablation.svg", "Ablation: where the gain comes from",
            note="Real results. Fold 2 and RankIC are reported in the text.")


def cum_ic(out):
    """第一折累積逐日 IC（真實）：主結果、[24] LASSO、MEIG [1]。資料讀取方式與主表（results_table.py）相同。"""
    import sys
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import results_table as R  # noqa: E402
    specs = [("MAGNET", R.neural_stats("tw50_rgcnSA")),
             ("LASSO [24]", R.nonneural_stats(R.BIP.format("LASSO"))),
             ("MEIG [1]", R.neural_stats("tw50_bl_meig"))]
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, st in specs:
        v = np.nan_to_num(np.asarray(st["daily_ic"], float))
        dates = np.asarray([np.datetime64(str(d)[:10]) for d in st["dates"]])
        c = np.cumsum(v)
        s = STYLE[m]
        ax.plot(dates, c, color=s["color"], ls="-", lw=2, marker=s["marker"], ms=6, mfc=s["color"],
                mew=2, markevery=40, label=f"{m} (mean IC {np.nanmean(st['daily_ic']):.3f})")
        ax.text(dates[-1], c[-1], f"  {c[-1]:.1f}", va="center", color=s["color"], fontsize=11)
    ax.set_xlabel("Test-period trading date (fold 1, 2025)")
    ax.set_ylabel("Cumulative daily IC")
    ax.legend(frameon=False, loc="upper left")
    _finish(fig, ax, out / "eval_plan_cum_ic.svg", "Temporal stability: cumulative daily IC",
            note="Real results. MAGNET: 10-seed mean; MEIG: 3-seed mean.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/discussion")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (info_source, info_lag, market_state, ablation, cum_ic):
        f(out)
    print(f"寫入 {out}")


if __name__ == "__main__":
    main()
