"""
gen_eval_figures.py — Performance evaluation 的三張圖（英文版，2:1）。

    eval_timeline_en.svg      兩折 walk-forward 的 train / val / test 與 2026 探索性窗的時間軸
    eval_main_dots_en.svg     主結果對文獻方法：兩折並列的點圖（預設 IC，--metric rankic 換 RankIC）
    eval_ablation_en.svg      主結果、不帶關係、單市場消融：兩折 x 兩指標的長條圖

數字一律從 docs/results_table.md 的兩張主表讀，不寫死。[24] 的「tie / worse」由主表的逐日 p
依 16 個文獻方法做 Holm（每折每指標一族）算出，與 best_vs_literature.py 同一規則。
切分日期取自 data/graphs/snapshots_tw50 的檔名（依 configs/base.yaml 的 split 索引與第二折的 903 / 1150）。

    .venv/bin/python scripts/gen_eval_figures.py [--metric ic|rankic]

產生後檢查：文字兩兩不重疊、文字在畫布內。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"

W, H = 1800, 900
INK = "#1f2933"           # 與 gen_magnet_figure.py / gen_tw_relation_figure.py 相同
GREY = "#5b6672"
LINE = "#7d8a99"
MAIN = "#b45309"          # 主結果：架構圖 v7 的琥珀深色
TIE = "#2b6cb0"           # [24] 打平
WORSE = "#9aa5b1"         # 顯著較差
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

NEURAL = ["MAN-SF [11]", "MEIG [1]", "HGT [14]", "Adv-ALSTM [10]", "DeltaLag [13]", "HATS [12]"]
L24 = ["LASSO", "RIDGE", "OLS", "ens-med", "ens-avg", "RF", "AdaBoost", "LightGBM", "XGBoost", "SVM"]
MAIN_ROW = "**MAGNET（+ 同產業關係，主結果）**"
PRIOR_ROW = "MAGNET 不帶關係（原主結果）"
SINGLE_ROW = "單市場消融（主結果）"

TEXTS: list[tuple[float, float, float, str, str]] = []   # (x, y, size, anchor, text) 供碰撞檢查


# ---------------------------------------------------------------- 讀資料
def _num(cell: str):
    m = re.match(r"\s*([+-]?\d+\.\d+)(?:\s*±\s*(\d+\.\d+))?", cell)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else None) if m else (None, None)


def load_tables() -> dict[int, dict[str, dict]]:
    """回傳 {fold: {列名: {ic, ic_sd, rk, rk_sd, p_ic, p_rk}}}。"""
    txt = (ROOT / "docs" / "results_table.md").read_text()
    sec1 = txt.split("### 主表：IC 與 RankIC\n", 1)[1].split("\n## ", 1)[0]
    sec2 = txt.split("### 主表：IC 與 RankIC（第二折）\n", 1)[1].split("\n**", 1)[0]
    out = {}
    for fold, sec, off in ((1, sec1, 3), (2, sec2, 2)):
        rows = {}
        for line in sec.splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) < off + 4 or not c[off].startswith(("+", "-")):
                continue
            ic, ic_sd = _num(c[off])
            rk, rk_sd = _num(c[off + 1])
            p = [float(x) if re.match(r"\d", x) else None for x in c[off + 2:off + 4]]
            rows[c[0]] = dict(ic=ic, ic_sd=ic_sd, rk=rk, rk_sd=rk_sd, p_ic=p[0], p_rk=p[1])
        out[fold] = rows
    return out


def holm_sig(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    order = sorted(pvals, key=pvals.get)
    sig, m, stop = {}, len(order), False
    for k, name in enumerate(order):
        if not stop and pvals[name] <= alpha / (m - k):
            sig[name] = True
        else:
            stop = True
            sig[name] = False
    return sig


def split_dates() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())["data"]["split"]
    d = sorted(p.stem[6:] for p in (ROOT / "data" / "graphs" / "snapshots_tw50").glob("graph_*.pt"))
    tr1, va1 = cfg["train_end"], cfg["val_end"]
    tr2, va2 = 903, tr1            # 第二折：train 0-902 / val 903-1149 / test 1150-1396（results_table）
    return {
        1: (d[0], d[tr1 - 1], d[tr1], d[va1 - 1], d[va1], "2025-12-30"),
        2: (d[0], d[tr2 - 1], d[tr2], d[va2 - 1], d[va2], d[va1 - 1]),
    }


# ---------------------------------------------------------------- SVG 小工具
def text(x, y, s, size=20, anchor="start", fill=INK, weight="normal", style="normal"):
    TEXTS.append((x, y, size, anchor, s))
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
            f'font-weight="{weight}" font-style="{style}">{s}</text>')


def svg_wrap(body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'font-family="{FONT}">\n<rect width="{W}" height="{H}" fill="#ffffff"/>\n'
            + "\n".join(body) + "\n</svg>")


def _tw(size, s):
    return 0.53 * size * len(s)


def check_texts(name: str):
    boxes = []
    for x, y, size, anchor, s in TEXTS:
        w = _tw(size, s)
        x0 = x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
        boxes.append((x0, y - 0.8 * size, x0 + w, y + 0.22 * size, s))
    bad = 0
    for i, a in enumerate(boxes):
        if a[0] < 0 or a[2] > W or a[1] < 0 or a[3] > H:
            print(f"  [{name}] 超出畫布：{a[4]}")
            bad += 1
        for b in boxes[i + 1:]:
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                print(f"  [{name}] 重疊：{a[4]!r} / {b[4]!r}")
                bad += 1
    print(f"檢查 {name}：{len(TEXTS)} 個文字，{bad} 處問題")
    TEXTS.clear()
    return bad


# ---------------------------------------------------------------- 圖 1：時間軸
def fig_timeline() -> str:
    sd = split_dates()
    import datetime as dt

    t0, t1 = dt.date(2019, 1, 1), dt.date(2026, 12, 31)
    X0, X1 = 260, 1720

    def X(s):
        d = dt.date.fromisoformat(s)
        return X0 + (X1 - X0) * (d - t0).days / (t1 - t0).days

    b = [text(70, 80, "Walk-forward evaluation windows", 34, weight="bold"),
         text(70, 118, "Two non-overlapping test periods; the 2026 window was used after model selection (exploratory)",
              20, fill=GREY)]
    for yr in range(2019, 2027):
        x = X(f"{yr}-01-01")
        b.append(f'<line x1="{x:.1f}" y1="180" x2="{x:.1f}" y2="700" stroke="#e4e7eb" stroke-width="1.5"/>')
        b.append(text(x, 730, str(yr), 20, "middle", GREY))
    colors = {"Train": "#cbd2d9", "Validation": "#7d8a99", "Test": MAIN}
    rows = [("Fold 1 (main)", 1, 260), ("Fold 2", 2, 420)]
    for label, f, y in rows:
        a, b_, c, d_, e, g = sd[f]
        b.append(text(230, y + 8, label, 24, "end", weight="bold"))
        for (s, t, kind) in ((a, b_, "Train"), (c, d_, "Validation"), (e, g, "Test")):
            x1, x2 = X(s), X(t)
            b.append(f'<rect x="{x1:.1f}" y="{y - 26}" width="{x2 - x1:.1f}" height="52" rx="4" fill="{colors[kind]}"/>')
        b.append(text((X(a) + X(b_)) / 2, y + 8, "Train", 20, "middle", INK))
        b.append(text((X(c) + X(d_)) / 2, y + 8, "Val", 20, "middle", "#ffffff", "bold"))
        b.append(text((X(e) + X(g)) / 2, y + 8, "Test", 20, "middle", "#ffffff", "bold"))
        b.append(text(X(e), y + 56, f"test {e} to {g}", 18, "start" if f == 2 else "end", GREY)
                 if f == 2 else text(X(g), y + 56, f"test {e} to {g}", 18, "end", GREY))
    y = 580
    b.append(text(230, y + 8, "2026 window", 24, "end", weight="bold"))
    x1, x2 = X("2026-01-02"), X("2026-10-16")
    b.append(f'<rect x="{x1:.1f}" y="{y - 26}" width="{x2 - x1:.1f}" height="52" rx="4" fill="none" '
             f'stroke="{MAIN}" stroke-width="2.5" stroke-dasharray="9,7"/>')
    b.append(text(x1 - 14, y + 8, "frozen fold-1 models, exploratory", 18, "end", GREY))
    b.append(text(x2, y + 56, "2026-01-02 to 2026-10-16", 18, "end", GREY))
    lx = 260
    for kind in ("Train", "Validation", "Test"):
        b.append(f'<rect x="{lx}" y="782" width="34" height="22" rx="3" fill="{colors[kind]}"/>')
        b.append(text(lx + 46, 800, kind, 20))
        lx += 200
    b.append(f'<rect x="{lx}" y="782" width="34" height="22" rx="3" fill="none" stroke="{MAIN}" '
             f'stroke-width="2.5" stroke-dasharray="6,4"/>')
    b.append(text(lx + 46, 800, "Exploratory (not a clean holdout)", 20))
    b.append(text(70, 860, "No shuffling; all models of a fold are trained and selected on the same train / validation days.",
                  17, fill=GREY))
    return svg_wrap(b)


# ---------------------------------------------------------------- 圖 2：主結果點圖
def fig_dots(tab, metric: str) -> str:
    k, ksd, kp = ("ic", "ic_sd", "p_ic") if metric == "ic" else ("rk", "rk_sd", "p_rk")
    mname = "IC" if metric == "ic" else "RankIC"
    lit = [f"[24] 二部圖 {x}" for x in L24] + NEURAL
    sig = {f: holm_sig({n: tab[f][n][kp] for n in lit}) for f in (1, 2)}
    l24 = sorted(L24, key=lambda x: -tab[1][f"[24] 二部圖 {x}"][k])
    neu = sorted(NEURAL, key=lambda x: -tab[1][x][k])
    order = [("MAGNET (main) †", MAIN_ROW, "main")] + \
            [(f"[24] {x}", f"[24] 二部圖 {x}", "24") for x in l24] + [(x, x, "nn") for x in neu]
    y0, step, gap = 190, 33, 22
    ys, y = [], y0
    prev = None
    for _, _, g in order:
        if prev and g != prev:
            y += gap
        ys.append(y)
        y += step
        prev = g
    lo, hi = -0.04, 0.18
    panels = {1: (440, 1060), 2: (1140, 1760)}

    def X(f, v):
        a, b_ = panels[f]
        return a + (b_ - a) * (v - lo) / (hi - lo)

    b = [text(60, 70, f"MAGNET vs literature methods ({mname})", 34, weight="bold"),
         text(60, 106, "Both test periods; dots are seed means, bars are seed SD", 20, fill=GREY)]
    for f, title in ((1, "Fold 1: 2024-12-26 to 2025-12-30"), (2, "Fold 2: 2023-12-20 to 2024-12-25")):
        a, b_ = panels[f]
        b.append(text((a + b_) / 2, 160, title, 22, "middle", weight="bold"))
        for t in (0.0, 0.05, 0.10, 0.15):
            x = X(f, t)
            b.append(f'<line x1="{x:.1f}" y1="{y0 - 18}" x2="{x:.1f}" y2="{ys[-1] + 18}" '
                     f'stroke="{"#9aa5b1" if t == 0 else "#e4e7eb"}" stroke-width="1.5"/>')
            b.append(text(x, ys[-1] + 46, f"{t:.2f}", 18, "middle", GREY))
        mv = tab[f][MAIN_ROW][k]
        x = X(f, mv)
        b.append(f'<line x1="{x:.1f}" y1="{y0 - 18}" x2="{x:.1f}" y2="{ys[-1] + 18}" stroke="{MAIN}" '
                 f'stroke-width="1.5" stroke-dasharray="5,5"/>')
    for (lab, row, g), y in zip(order, ys):
        b.append(text(410, y + 7, lab, 19, "end", MAIN if g == "main" else INK,
                      "bold" if g == "main" else "normal"))
        for f in (1, 2):
            r = tab[f][row]
            v, sdv = r[k], r[ksd]
            x = X(f, v)
            if sdv:
                b.append(f'<line x1="{X(f, v - sdv):.1f}" y1="{y}" x2="{X(f, v + sdv):.1f}" y2="{y}" '
                         f'stroke="{INK}" stroke-width="2"/>')
            if g == "main":
                b.append(f'<circle cx="{x:.1f}" cy="{y}" r="9" fill="{MAIN}"/>')
            elif g == "24" and not sig[f][row]:
                b.append(f'<circle cx="{x:.1f}" cy="{y}" r="8" fill="#ffffff" stroke="{TIE}" stroke-width="3"/>')
            else:
                b.append(f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="{WORSE}"/>')
    ly = H - 52
    b.append(f'<circle cx="80" cy="{ly - 6}" r="9" fill="{MAIN}"/>')
    b.append(text(98, ly, "MAGNET main result", 19))
    b.append(f'<circle cx="340" cy="{ly - 6}" r="8" fill="#ffffff" stroke="{TIE}" stroke-width="3"/>')
    b.append(text(358, ly, "Tie: not significant after Holm", 19))
    b.append(f'<circle cx="680" cy="{ly - 6}" r="7" fill="{WORSE}"/>')
    b.append(text(698, ly, "Significantly lower than MAGNET (Holm, 16 methods per fold)", 19))
    b.append(text(60, H - 18, "† Chosen after the ablation study (post hoc); its numbers are biased upward. "
                  "Neural baselines untuned, 3 seeds; [24] learners are deterministic.", 16, fill=GREY))
    return svg_wrap(b)


# ---------------------------------------------------------------- 圖 3：消融長條圖
def fig_ablation(tab) -> str:
    arms = [("MAGNET (main) †", MAIN_ROW, MAIN),
            ("Without same-industry relations (previous main)", PRIOR_ROW, "#d69e5e"),
            ("Single market: cross-market links off", SINGLE_ROW, WORSE)]
    groups = [(1, "ic", "Fold 1  IC"), (1, "rk", "Fold 1  RankIC"), (2, "ic", "Fold 2  IC"), (2, "rk", "Fold 2  RankIC")]
    base, top, hi = 720, 200, 0.18
    gx0, gw, bw = 160, 400, 96

    def Y(v):
        return base - (base - top) * v / hi

    b = [text(60, 70, "Ablation: where the gain comes from", 34, weight="bold"),
         text(60, 106, "Seed means with seed SD (10 seeds each)", 20, fill=GREY)]
    for t in (0.0, 0.05, 0.10, 0.15):
        y = Y(t)
        b.append(f'<line x1="{gx0 - 20}" y1="{y:.1f}" x2="{gx0 + 4 * gw}" y2="{y:.1f}" '
                 f'stroke="{"#9aa5b1" if t == 0 else "#e4e7eb"}" stroke-width="1.5"/>')
        b.append(text(gx0 - 30, y + 6, f"{t:.2f}", 18, "end", GREY))
    for gi, (f, k, lab) in enumerate(groups):
        cx = gx0 + gw * gi + gw / 2
        b.append(text(cx, base + 40, lab, 22, "middle", weight="bold"))
        for ai, (_, row, col) in enumerate(arms):
            r = tab[f][row]
            v, sdv = r[k], r[k + "_sd"]
            x = cx + (ai - 1) * (bw + 14) - bw / 2
            b.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw}" height="{base - Y(v):.1f}" fill="{col}"/>')
            xm = x + bw / 2
            b.append(f'<line x1="{xm:.1f}" y1="{Y(v - sdv):.1f}" x2="{xm:.1f}" y2="{Y(v + sdv):.1f}" '
                     f'stroke="{INK}" stroke-width="2"/>')
            b.append(text(xm, Y(v + sdv) - 10, f"{v:.3f}", 17, "middle"))
    lx, ly = 160, 818
    for lab, _, col in arms:
        b.append(f'<rect x="{lx}" y="{ly - 18}" width="26" height="22" fill="{col}"/>')
        b.append(text(lx + 36, ly, lab, 19))
        lx += 36 + _tw(19, lab) + 50
    b.append(text(60, H - 22, "† Chosen after the ablation study (post hoc): the gap to the previous main result "
                  "is not evidence for the relational layer.", 16, fill=GREY))
    return svg_wrap(b)


def main():
    ap = argparse.ArgumentParser(description="Performance evaluation 的三張圖")
    ap.add_argument("--metric", choices=["ic", "rankic"], default="ic")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    tab = load_tables()
    bad = 0
    suffix = "" if args.metric == "ic" else "_rankic"
    for name, svg in (("eval_timeline_en.svg", fig_timeline()),):
        (out / name).write_text(svg + "\n")
        bad += check_texts(name)
    name = f"eval_main_dots{suffix}_en.svg"
    (out / name).write_text(fig_dots(tab, args.metric) + "\n")
    bad += check_texts(name)
    (out / "eval_ablation_en.svg").write_text(fig_ablation(tab) + "\n")
    bad += check_texts("eval_ablation_en.svg")
    print(f"-> {out}；共 {bad} 處問題")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
