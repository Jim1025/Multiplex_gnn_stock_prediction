"""
gen_discussion_figures.py — 和教授討論評估章節用的示意圖（英文，2:1）

四張圖，對應討論出的議題組合：
    eval_plan_info_source.svg      資訊來源（只用台股 vs 台股 + 美股）
    eval_plan_info_lag.svg         資訊時效（美股輸入落後 1–4 天）
    eval_plan_market_state.svg     市場狀態（美股前一晚等權報酬絕對值的五分位）
    eval_plan_ablation.svg         消融（原 2x2 圖只留第一折 IC）

全部是真實數據（第一折 IC，種子平均），沒有假設值（2026-11-16 起，proposal_1209 §1）：
    資訊來源、資訊時效   docs/figures/data/eval_bipartite_variants.json（scripts/bipartite_variants_eval.py --write）
    資訊時效的 MAGNET、MEIG  docs/figures/data/eval_input_lag.json（scripts/input_lag_eval.py --write，proposal_1209 §2）
    市場狀態             docs/figures/data/eval_market_state.json（scripts/market_state_eval.py --write）
    消融、MEIG           docs/results_table.md 與 proposal_819 §83

    .venv/bin/python scripts/gen_discussion_figures.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
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
DATA = Path(__file__).resolve().parents[1] / "docs" / "figures" / "data"
MEIG_IC = 0.0061          # 主表，第一折，3 顆平均


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def _finish(fig, ax, path, title, note=None):
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
    # 兩種來源：只用台股、台股 + 美股。None = 不存在（MEIG 原生跨市場，不做只用台股的版本）
    # [24] 的「台股 + 美股」是原文 Appendix E 的設定（來源 = 美股 + 台股）；主表的 [24] 是只用美股（0.1042），
    # 之前把它標成 TW + US 是錯的（proposal_1209 §1.3）。
    v = _load("eval_bipartite_variants.json")
    lasso, mag = v["models"]["LASSO"]["ic"], v["magnet"]
    data = {
        "MAGNET": [mag["tw50_smktSA"]["ic"], mag["tw50_rgcnSA"]["ic"]],
        "LASSO [24]": [lasso["tw"], lasso["both"]],
        "MEIG [1]": [None, MEIG_IC],
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.36
    for i, (m, vals) in enumerate(data.items()):
        for k, val in enumerate(vals):
            x = i + (k - 0.5) * w
            if val is None:
                ax.text(x, 0.004, "n/a\n(native\ncross-market)", ha="center", fontsize=9, color="#6b7280")
                continue
            ax.bar(x, val, w * 0.95, color="white" if k == 0 else STYLE[m]["color"],
                   edgecolor=STYLE[m]["color"], lw=1.5)
            ax.text(x, val + 0.002, f"{val:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(data)), list(data))
    ax.set_ylabel("Test IC (fold 1)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="white", ec="k"),
                       plt.Rectangle((0, 0), 1, 1, fc="#9ca3af", ec="k")],
              labels=["TW only", "TW + US"], frameon=False, loc="upper right")
    _finish(fig, ax, out / "eval_plan_info_source.svg", "Effect of information source",
            note=f"Real results. LASSO [24] with TW + US predictors follows Appendix E of [24]; "
                 f"its main-table setting (US predictors only) gives {lasso['us']:.3f}.")


def info_lag(out):
    lags = [1, 2, 3, 4]
    # 全部是真實值（proposal_1209 §1.4、§2）。MAGNET 與 MEIG 的落後 2–4 天是重新訓練的延遲版
    # （只換美股層輸入，圖、台股輸入與模型都不變；MAGNET 10 顆、MEIG 3 顆種子平均）。
    # 單市場不用美股、不受落後天數影響，畫成參考線而非方法
    v = _load("eval_bipartite_variants.json")
    lasso, mag = v["models"]["LASSO"]["ic"], v["magnet"]
    nn = _load("eval_input_lag.json")["methods"]
    data = {
        "MAGNET": [nn["MAGNET"][str(l)]["ic"] for l in lags],
        "LASSO [24]": [lasso["us"], lasso["lag2"], lasso["lag3"], lasso["lag4"]],
        "MEIG [1]": [nn["MEIG [1]"][str(l)]["ic"] for l in lags],
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    ref = mag["tw50_smktSA"]["ic"]
    ax.axhline(ref, color="#9ca3af", ls="--", lw=1.5, zorder=1,
               label=f"No-US reference (MAGNET-single market, {ref:.3f})")
    for m, ys in data.items():
        s = STYLE[m]
        ax.plot(lags, ys, color=s["color"], ls="-", lw=2, marker=s["marker"], ms=6, mfc=s["color"], mew=2,
                label=m)
    # 數值標籤的位置逐點指定，避開折線與參考線：(dx, dy, ha, va)。MEIG 四個點都在 0 附近，不標數值
    place = {
        "MAGNET": [(0.06, 0, "left", "center"), (0, 0.006, "center", "bottom"),
                   (0, 0.006, "center", "bottom"), (0.05, -0.010, "left", "center")],
        "LASSO [24]": [(-0.06, 0, "right", "center"), (0, -0.005, "center", "top"),
                       (0, -0.005, "center", "top"), (0, 0.005, "center", "bottom")],
    }
    for m, pl in place.items():
        for j, y in enumerate(data[m]):
            dx, dy, ha, va = pl[j]
            ax.text(lags[j] + dx, y + dy, f"{y:.3f}".replace("-", "−"), ha=ha, va=va, fontsize=10,
                    color=STYLE[m]["color"])
    ax.set_xticks(lags, ["t−1\n(default)", "t−2", "t−3", "t−4"])
    ax.set_xlim(0.7, 4.35)
    ax.set_ylim(bottom=-0.03)
    ax.set_xlabel("Lag of US input (trading days)")
    ax.set_ylabel("Test IC (fold 1)")
    ax.legend(frameon=False, loc="upper right")
    _finish(fig, ax, out / "eval_plan_info_lag.svg", "Effect of information timeliness",
            note="Real results (MAGNET: 10 seeds; MEIG: 3 seeds). Only the US input is lagged; "
                 "graphs still use data up to t−1 (as in Sec. 5.4 of [24]).")


def market_state(out):
    qs = ["Q1\n(calm)", "Q2", "Q3", "Q4", "Q5\n(large move)"]
    # 全部是真實值：各組的平均逐日 IC。不畫誤差線（四條線疊在一起太亂），標準誤的大小寫在圖下的註記
    methods = _load("eval_market_state.json")["methods"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(qs))
    se_max = 0.0
    for m, s in STYLE.items():
        ys = [c["mean"] for c in methods[m]["ic"]]
        se_max = max(se_max, *(c["se"] for c in methods[m]["ic"]))
        ax.plot(x, ys, color=s["color"], ls="-", lw=2, marker=s["marker"], ms=6,
                mfc=s["color"], mew=2, label=m)
    ax.set_xticks(x, qs)
    ax.set_xlabel("Size of last night's US move: quintile of |equal-weighted US return|")
    ax.set_ylabel("Mean daily IC (fold 1)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _finish(fig, ax, out / "eval_plan_market_state.svg", "Effect of market state",
            note=f"Real results. Each point averages about 49 days; its standard error is 0.01–{se_max:.2f}, "
                 f"so gaps smaller than about 0.05 are within noise.")


def ablation(out):
    # 真實數字：proposal §83.3、§83.4、scripts/sa_us_graph_eval.py（第一折 IC，種子平均 ± 種子 SD）
    arms = [("MAGNET", 0.1127, 0.0039, None),
            ("w/o relational\nTW layer", 0.1090, 0.0051, "//"),
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
