"""
gen_sensitivity_figure.py — beta_init_std 敏感度圖（英文版，2:1），performance evaluation 的穩健性圖。

    eval_sensitivity_en.svg    橫軸 beta_init_std（0.1 / 0.3 / 1.0，對數刻度），左 IC、右 RankIC，兩折各一條線，誤差帶 = 種子 SD

數字由預測檔重算（與 sa_optional_eval.py 的 C 項同一套：tw_lag_channel 的載入與逐種子 IC），
0.3 就是主結果 SA，其值必須等於主表，否則不輸出。arm：rgcnSAbis01、rgcnSA、rgcnSAbis10（proposal §83.10 C）。
兩折用實線 / 虛線與圓 / 方形標記區分，黑白印製仍分得出來。

    .venv/bin/python scripts/gen_sensitivity_figure.py
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import gen_eval_figures as G  # noqa: E402
import tw_lag_channel as T  # noqa: E402

from gen_eval_figures import GREY, INK, MAIN, TIE, H, text, svg_wrap, check_texts  # noqa: E402

PREFIX = {"第一折": "tw50", "第二折": "f2"}
FOLD_NO = {"第一折": 1, "第二折": 2}
ARMS = [(0.1, "rgcnSAbis01"), (0.3, "rgcnSA"), (1.0, "rgcnSAbis10")]
DEFAULT = 0.3
N_SEEDS = 10


def load(tab) -> dict:
    """{fold: {"ic": [(x, mean, sd)], "rk": [...]}}"""
    out = {}
    for fold, f in T.FOLDS.items():
        D = T.load_fold(f["cfg"], f["f2"])
        fo = FOLD_NO[fold]
        res = {"ic": [], "rk": []}
        for x, tag in ARMS:
            P, seeds = T.preds(f"runs/**/*{PREFIX[fold]}_{tag}_s*/predictions/{{}}_predictions.csv",
                               "test", D["d_test"], D["codes"])
            if len(seeds) != N_SEEDS:
                sys.exit(f"{fold} {tag}：預期 {N_SEEDS} 顆種子，實際 {len(seeds)} 顆")
            r = [T.summ(p, D["y_test"]) for p in P]
            for key, k in (("ic", "IC"), ("rk", "RIC")):
                v = np.array([z[k] for z in r])
                res[key].append((x, float(v.mean()), float(v.std(ddof=1))))
                if x == DEFAULT and abs(v.mean() - tab[fo][G.MAIN_ROW][key]) > 5e-5:
                    sys.exit(f"{fold} 主結果 {key} {v.mean():+.4f} 與主表 {tab[fo][G.MAIN_ROW][key]:+.4f} 不符")
        out[fo] = res
    return out


STYLE = {1: (MAIN, "", "circle"), 2: (TIE, "12 7", "square")}


def fig(data: dict) -> str:
    panels = [("ic", "IC", 150, 820), ("rk", "RankIC", 1010, 1680)]
    top, base, lo, hi = 190, 680, 0.08, 0.18

    def Y(v):
        return base - (base - top) * (v - lo) / (hi - lo)

    b = [text(60, 70, "Sensitivity to the beta-layer initialisation scale", 34, weight="bold"),
         text(60, 106, "Seed mean with seed SD band (10 seeds each); one setting changed at a time, "
              "everything else as in the main result", 20, fill=GREY)]
    for key, mname, x0, x1 in panels:
        def X(v, x0=x0, x1=x1):
            a, c = math.log10(0.1), math.log10(1.0)
            pad = 60
            return x0 + pad + (x1 - x0 - 2 * pad) * (math.log10(v) - a) / (c - a)

        b.append(text(x0, 160, mname, 24, weight="bold"))
        for t in (0.08, 0.10, 0.12, 0.14, 0.16, 0.18):
            y = Y(t)
            b.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e4e7eb" stroke-width="1.5"/>')
            b.append(text(x0 - 12, y + 6, f"{t:.2f}", 18, "end", GREY))
        xd = X(DEFAULT)
        b.append(f'<line x1="{xd:.1f}" y1="{top - 10}" x2="{xd:.1f}" y2="{base}" stroke="{INK}" '
                 'stroke-width="1.5" stroke-dasharray="4 5"/>')
        b.append(text(xd + 8, top + 8, "default (main result)", 17, fill=INK))
        for x, _ in ARMS:
            b.append(text(X(x), base + 32, f"{x:g}", 19, "middle", GREY))
        b.append(text((x0 + x1) / 2, base + 64, "beta_init_std (log scale)", 18, "middle", GREY))
        for fo in (1, 2):
            col, dash, mk = STYLE[fo]
            pts = data[fo][key]
            up = " ".join(f"{X(x):.1f},{Y(m + s):.1f}" for x, m, s in pts)
            dn = " ".join(f"{X(x):.1f},{Y(m - s):.1f}" for x, m, s in reversed(pts))
            b.append(f'<polygon points="{up} {dn}" fill="{col}" fill-opacity="0.15" stroke="none"/>')
            line = " ".join(f"{X(x):.1f},{Y(m):.1f}" for x, m, _ in pts)
            b.append(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="3.5"'
                     f'{f" stroke-dasharray=\"{dash}\"" if dash else ""}/>')
            for x, m, _ in pts:
                if mk == "circle":
                    b.append(f'<circle cx="{X(x):.1f}" cy="{Y(m):.1f}" r="8" fill="{col}" stroke="#ffffff" stroke-width="2"/>')
                else:
                    b.append(f'<rect x="{X(x) - 7.5:.1f}" y="{Y(m) - 7.5:.1f}" width="15" height="15" fill="{col}" '
                             'stroke="#ffffff" stroke-width="2"/>')
                dy = -16 if fo == 2 else 30
                if x == DEFAULT:                          # 預設值的標籤移到虛線右側
                    b.append(text(X(x) + 14, Y(m) + dy, f"{m:.3f}", 16, "start", col, "bold"))
                else:
                    b.append(text(X(x), Y(m) + dy, f"{m:.3f}", 16, "middle", col, "bold"))
    ly = 812
    for k, fo in enumerate((1, 2)):
        col, dash, mk = STYLE[fo]
        lx = 150 + 300 * k
        b.append(f'<line x1="{lx}" y1="{ly - 6}" x2="{lx + 56}" y2="{ly - 6}" stroke="{col}" stroke-width="3.5"'
                 f'{f" stroke-dasharray=\"{dash}\"" if dash else ""}/>')
        b.append(f'<circle cx="{lx + 28}" cy="{ly - 6}" r="7" fill="{col}"/>' if mk == "circle" else
                 f'<rect x="{lx + 21}" y="{ly - 13}" width="14" height="14" fill="{col}"/>')
        b.append(text(lx + 68, ly, f"Fold {fo}", 19))
    b.append(text(60, H - 42, "Pre-registered: both 0.1 and 1.0 judged robust (same direction in both folds with "
                  "Welch p < 0.05 needed to call it sensitive).", 16, fill=GREY))
    b.append(text(60, H - 16, "0.1 is flat; 1.0 is lower in both folds (fold 1 significant) and fold 2 seeds "
                  "spread widely. Main result chosen post hoc.", 16, fill=GREY))
    return svg_wrap(b)


def main():
    ap = argparse.ArgumentParser(description="beta_init_std 敏感度圖")
    ap.add_argument("--out-dir", default=str(G.OUT_DIR))
    a = ap.parse_args()
    data = load(G.load_tables())
    for fo in (1, 2):
        for key in ("ic", "rk"):
            print(f"第{fo}折 {key}：" + "；".join(f"{x:g} {m:+.4f} ± {s:.4f}" for x, m, s in data[fo][key]))
    name = "eval_sensitivity_en.svg"
    (Path(a.out_dir) / name).write_text(fig(data) + "\n")
    sys.exit(1 if check_texts(name) else 0)


if __name__ == "__main__":
    main()
