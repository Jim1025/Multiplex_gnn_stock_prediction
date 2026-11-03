"""
gen_cumic_figure.py — 逐日累積 IC 曲線（英文版，2:1），performance evaluation 的穩健性圖。

    eval_cum_ic_en.svg    主結果（SA）、[24] LASSO、主表 IC 最高的文獻神經模型，兩折並列

每條線是「種子平均的逐日 IC」的累加（[24] 是確定性模型，只有一條）。逐日序列與主表同一套算法：
第一折用 results_table.neural_stats / nonneural_stats，第二折用 best_vs_literature.series（評估窗到 2024-12-25）。
序列的平均必須等於主表的 IC，否則不輸出。線除了顏色也用實線、虛線、點線區分，黑白印製仍分得出來。

    .venv/bin/python scripts/gen_cumic_figure.py [--metric ic|rankic]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import best_vs_literature as BVL  # noqa: E402
import gen_eval_figures as G  # noqa: E402
import results_table as R  # noqa: E402

from gen_eval_figures import GREY, INK, MAIN, TIE, H, W, text, svg_wrap, check_texts  # noqa: E402

MAIN_ARM = {1: "tw50_rgcnSA", 2: "f2_rgcnSA"}
NEURAL_ARM = {lab: arm for lab, arm, cat, _ in R.NEURAL if cat == "文獻"}


def f2_files(arm: str) -> list[str]:
    return sorted(glob.glob(str(ROOT / f"runs/**/*{arm}_s*/predictions/test_predictions.csv"), recursive=True))


def load_series(metric: str, tab) -> dict:
    """{fold: [(圖例, 日期, 逐日值, 樣式), ...]}；順序 = 主結果、LASSO、最佳文獻神經模型。"""
    i, key = (0, "ic") if metric == "ic" else (1, "rk")
    out = {}
    for f in (1, 2):
        best = max(G.NEURAL, key=lambda n: tab[f][n][key])
        specs = [("MAGNET", G.MAIN_ROW, "main"),
                 ("LASSO [24] (best [24] learner)", "[24] 二部圖 LASSO", "lasso"),
                 (f"{best} (best of other single market baseline)", best, "neural")]
        rows = []
        for lab, row, kind in specs:
            if f == 1:
                if kind == "lasso":
                    s = R.nonneural_stats(R.BIP.format("LASSO"))
                else:
                    s = R.neural_stats(MAIN_ARM[1] if kind == "main" else NEURAL_ARM[row])
                dates, v = list(s["dates"]), np.asarray(s["daily_ic" if i == 0 else "daily_ric"], float)
            else:
                if kind == "lasso":
                    files = [R._latest(f"runs_f2/{R.BIP.format('LASSO')}") + "/predictions/test_predictions.csv"]
                    files = [str(ROOT / files[0])]
                elif kind == "main":
                    files = f2_files(MAIN_ARM[2])
                else:
                    files = f2_files("f2_bl_" + NEURAL_ARM[row][len("tw50_bl_"):])
                dates, V = BVL.series(files)
                v = V[i]
            dates = [str(d)[:10] for d in dates]
            got, want = float(np.nanmean(v)), tab[f][row][key]
            if abs(got - want) > 5e-5:
                sys.exit(f"第{f}折 {lab}：逐日平均 {got:+.4f} 與主表 {want:+.4f} 不符")
            rows.append((lab, dates, v, kind))
        out[f] = rows
    return out


STYLE = {"main": (MAIN, 4, ""), "lasso": (TIE, 3, "14 8"), "neural": (GREY, 3, "3 6")}


def fig_cum(series: dict, metric: str) -> str:
    mname = "IC" if metric == "ic" else "RankIC"
    allc = [np.nancumsum(v) for f in series for _, _, v, _ in series[f]]
    hi = float(np.ceil(max(c.max() for c in allc) / 10) * 10)
    lo = min(0.0, float(np.floor(min(c.min() for c in allc) / 5) * 5))
    top, base = 190, 700
    panels = {2: (150, 830), 1: (990, 1700)}       # 依時間由左到右：第二折 2024、第一折 2025

    def Y(v):
        return base - (base - top) * (v - lo) / (hi - lo)

    b = [text(60, 70, f"Cumulative daily {mname} over the test period", 34, weight="bold"),
         text(60, 106, f"Running sum of the seed-mean daily {mname}; slope = mean daily {mname}, so end value = mean × "
              f"{len(series[1][0][2])} trading days", 20, fill=GREY)]
    for f, (x0, x1) in panels.items():
        rows = series[f]
        d0 = dt.date.fromisoformat(rows[0][1][0])
        d1 = dt.date.fromisoformat(rows[0][1][-1])

        def X(s, x0=x0, x1=x1, d0=d0, d1=d1):
            return x0 + (x1 - x0) * (dt.date.fromisoformat(s) - d0).days / (d1 - d0).days

        b.append(text(x0, 160, f"Fold {f} ({d1.year})", 22, weight="bold"))
        for t in np.arange(np.ceil(lo / 10) * 10, hi + 1e-9, 10):
            y = Y(t)
            b.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                     f'stroke="{"#9aa5b1" if t == 0 else "#e4e7eb"}" stroke-width="1.5"/>')
            b.append(text(x0 - 12, y + 6, f"{t + 0:.0f}", 18, "end", GREY))
        q = d0.replace(day=1)
        while q <= d1:                                    # 每季一個刻度
            if q >= d0 and q.month in (1, 4, 7, 10):
                x = X(q.isoformat())
                b.append(f'<line x1="{x:.1f}" y1="{base}" x2="{x:.1f}" y2="{base + 8}" stroke="{GREY}" stroke-width="1.5"/>')
                b.append(text(x, base + 32, q.strftime("%Y-%m"), 17, "middle", GREY))
            q = (q.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        b.append(text((x0 + x1) / 2, base + 62, "Test-period trading date (year-month)", 17, "middle", GREY))
        yc = (top + base) / 2                             # 縱軸標題：旋轉 90 度，不經 text() 所以不進碰撞檢查
        b.append(f'<text x="{x0 - 52}" y="{yc:.1f}" font-size="17" text-anchor="middle" fill="{GREY}" '
                 f'transform="rotate(-90 {x0 - 52} {yc:.1f})">Cumulative {mname} (sum of daily {mname}, unitless)</text>')
        ends = []
        for lab, dates, v, kind in rows:
            col, sw, dash = STYLE[kind]
            c = np.nancumsum(v)
            pts = " ".join(f"{X(d):.1f},{Y(y):.1f}" for d, y in zip(dates, c))
            b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{sw}" '
                     f'stroke-linejoin="round"{f" stroke-dasharray=\"{dash}\"" if dash else ""}/>')
            ends.append([Y(c[-1]), c[-1], col, float(np.nanmean(v))])
        ends.sort()
        for k in range(1, len(ends)):                    # 終點標籤（兩行）至少相隔 42px
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 42)
        for y, val, col, m in ends:
            b.append(text(x1 + 10, y + 6, f"{val:.1f}", 18, fill=col, weight="bold"))
            b.append(text(x1 + 10, y + 24, f"mean {m:.3f}", 14, fill=col))
    ly = 812
    n1, n2 = (series[f][2][0].split(" (")[0] for f in (1, 2))
    labs = [series[1][0][0], "LASSO [24] (best [24] learner)", f"Best of other single market baseline (fold 2: {n2} / fold 1: {n1})"]
    for k, ((_, _, _, kind), lab) in enumerate(zip(series[1], labs)):
        col, sw, dash = STYLE[kind]
        lx = 150 + [0, 300, 700][k]
        b.append(f'<line x1="{lx}" y1="{ly - 6}" x2="{lx + 60}" y2="{ly - 6}" stroke="{col}" stroke-width="{sw}"'
                 f'{f" stroke-dasharray=\"{dash}\"" if dash else ""}/>')
        b.append(text(lx + 72, ly, lab, 19))
    b.append(text(60, H - 42, f"Best neural baseline = highest table {mname} among the six neural baselines in that fold; "
                  "untuned, 3 seeds. MAGNET 10 seeds; [24] deterministic.",
                  16, fill=GREY))
    return svg_wrap(b)


def main():
    ap = argparse.ArgumentParser(description="逐日累積 IC 曲線")
    ap.add_argument("--metric", choices=["ic", "rankic"], default="ic")
    ap.add_argument("--out-dir", default=str(G.OUT_DIR))
    a = ap.parse_args()
    s = load_series(a.metric, G.load_tables())
    for f in (1, 2):
        print(f"第{f}折：" + "；".join(f"{lab} 累積 {np.nancumsum(v)[-1]:.2f}（{len(v)} 天）" for lab, _, v, _ in s[f]))
    name = f"eval_cum_{'ic' if a.metric == 'ic' else 'rankic'}_en.svg"
    (Path(a.out_dir) / name).write_text(fig_cum(s, a.metric) + "\n")
    bad = check_texts(name)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
