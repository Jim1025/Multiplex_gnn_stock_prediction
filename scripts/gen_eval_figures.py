"""
gen_eval_figures.py — Performance evaluation 的時間軸圖（英文版，2:1）。

    eval_timeline_en.svg      兩折 walk-forward 的 train / val / test 與 2026 探索性窗的時間軸
                              （proposal_1111.pptx 第 21 頁用的就是這張）

切分日期取自 data/graphs/snapshots_tw50 的檔名（依 configs/base.yaml 的 split 索引與第二折的 903 / 1150）。

原本還畫主結果點圖（eval_main_dots_en.svg、eval_main_dots_rankic_en.svg）與消融長條圖（eval_ablation_en.svg），
這三張已在 6ebf7f0（2026-11-08）刪除，消融改由 gen_discussion_figures.py 的 eval_plan_ablation.svg 取代；
2026-11-17 把對應的程式也拿掉，只留時間軸。舊程式見 git 歷史。

    .venv/bin/python scripts/gen_eval_figures.py [--out-dir DIR]

產生後檢查：文字兩兩不重疊、文字在畫布內。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"

W, H = 1800, 900
INK = "#1f2933"           # 與 gen_magnet_figure.py / gen_tw_relation_figure.py 相同
GREY = "#5b6672"
MAIN = "#b45309"          # 測試期：架構圖 v7 的琥珀深色
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

TEXTS: list[tuple[float, float, float, str, str]] = []   # (x, y, size, anchor, text) 供碰撞檢查


# ---------------------------------------------------------------- 讀資料
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


# ---------------------------------------------------------------- 時間軸
def fig_timeline() -> str:
    sd = split_dates()
    import datetime as dt

    t0, t1 = dt.date(2019, 1, 1), dt.date(2026, 12, 31)
    X0, X1 = 260, 1260

    def X(s):
        d = dt.date.fromisoformat(s)
        return X0 + (X1 - X0) * (d - t0).days / (t1 - t0).days

    b = [text(70, 150, "Walk-forward evaluation windows", 34, weight="bold"),
         text(70, 180, "Two non-overlapping test periods; the 2026 window was used after model selection (exploratory)",
              20, fill=GREY)]
    for yr in range(2019, 2027):
        x = X(f"{yr}-01-01")
        b.append(f'<line x1="{x:.1f}" y1="250" x2="{x:.1f}" y2="500" stroke="#e4e7eb" stroke-width="1.5"/>')
        b.append(text(x, 530, str(yr), 20, "middle", GREY))
    colors = {"Train": "#cbd2d9", "Validation": "#7d8a99", "Test": MAIN}
    rows = [("Fold 1 (main)", 1, 310), ("Fold 2", 2, 420)]
    for label, f, y in rows:
        a, b_, c, d_, e, g = sd[f]
        b.append(text(230, y + 8, label, 24, "end", weight="bold"))
        for (s, t, kind) in ((a, b_, "Train"), (c, d_, "Validation"), (e, g, "Test")):
            x1, x2 = X(s), X(t)
            b.append(f'<rect x="{x1:.1f}" y="{y - 26}" width="{x2 - x1:.1f}" height="52" rx="4" fill="{colors[kind]}"/>')
        b.append(text((X(a) + X(b_)) / 2, y + 8, "Train", 20, "middle", INK))
        b.append(text((X(c) + X(d_)) / 2, y + 8, "Val", 20, "middle", "#ffffff", "bold"))
        b.append(text((X(e) + X(g)) / 2, y + 8, "Test", 20, "middle", "#ffffff", "bold"))

    y = 580
    lx = 260
    for kind in ("Train", "Validation", "Test"):
        b.append(f'<rect x="{lx}" y="580" width="34" height="22" rx="3" fill="{colors[kind]}"/>')
        b.append(text(lx + 46, 600, kind, 20))
        lx += 200
    b.append(text(70, 660, "No shuffling; all models of a fold are trained and selected on the same train / validation days.",
                  17, fill=GREY))
    return svg_wrap(b)


def main():
    ap = argparse.ArgumentParser(description="Performance evaluation 的時間軸圖")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    name = "eval_timeline_en.svg"
    (out / name).write_text(fig_timeline() + "\n")
    bad = check_texts(name)
    print(f"-> {out / name}；共 {bad} 處問題")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
