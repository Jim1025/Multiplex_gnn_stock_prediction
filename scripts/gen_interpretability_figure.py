"""gen_interpretability_figure.py — 兩個可解釋性證據的投影片用圖（§55 / §55.7 / §55.8）

輸出**兩張獨立的純英文圖**（投影片一張講一件事，比並排的雙面板好用）：

  interp_b_industry_en.svg   ①  B 的欄間餘弦，產業 x 產業的區塊均值（參數矩陣層級）
  interp_tilt_en.svg         ②  w̄ 的逐檔持久傾斜（模型輸出層級）

**① 不畫 50x50 的原始格點。** 單格餘弦的 sd 是 0.224（同產業）/ 0.277（異產業），
遠大於區塊均值之間的差（+0.26 vs −0.04），原始格點必然看起來像雜訊——
結構是統計上的，不是逐格看得出來的（§55.5 的更正）。

與 `gen_magnet_figure.py` 的架構圖 house style 不同：**那張不放讀數，
這兩張是結果圖，讀數就是內容**。共通的是無表情符號、字不重疊。

用法：
    .venv/bin/python scripts/gen_interpretability_figure.py
    .venv/bin/python scripts/gen_interpretability_figure.py --arm tw50_beta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import beta_industry_clustering as bic   # noqa: E402
import static_tilt as stl                # noqa: E402
from src.dataset.config import load_universe   # noqa: E402

INK, MUTE, LINE = "#1a1a1a", "#8a8a8a", "#d8d8d8"
# 三個粗分組的顏色。避開紅綠，色盲可分。
GC = {"Electronics": "#1f6f8b", "Financials": "#b45309", "Other": "#9aa0a6"}

# 台股的產業別是中文，投影片要英文。只列 universe 實際用到的。
IND_EN = {
    "半導體業": "Semiconductors", "電腦及週邊設備業": "Computers & Peripherals",
    "電子零組件業": "Electronic Components", "其他電子業": "Other Electronics",
    "光電業": "Optoelectronics", "通信網路業": "Telecom & Networking",
    "金融保險業": "Financials", "塑膠工業": "Plastics", "水泥工業": "Cement",
    "鋼鐵工業": "Steel", "橡膠工業": "Rubber", "紡織纖維": "Textiles",
    "食品工業": "Food", "航運業": "Shipping", "貿易百貨": "Trading & Retail",
    "油電燃氣業": "Oil, Gas & Power", "運動休閒": "Sports & Leisure", "其他": "Other",
}
NAMED = {"2330": "TSMC", "3711": "ASE", "2303": "UMC",
         "4904": "FET", "3045": "Taiwan Mobile", "2412": "Chunghwa Telecom"}


def coarse(en: str) -> str:
    if en in ("Semiconductors", "Computers & Peripherals", "Electronic Components",
              "Other Electronics", "Optoelectronics", "Telecom & Networking"):
        return "Electronics"
    return "Financials" if en == "Financials" else "Other"


def heat(v: float, vmax: float = 0.6) -> str:
    """−vmax..vmax -> 暖橘 / 白 / 冷藍。0 附近是白的，讓對角線跳出來。"""
    t = max(-1.0, min(1.0, v / vmax))
    a = np.array([255, 255, 255])
    b = np.array([31, 111, 139]) if t >= 0 else np.array([180, 83, 9])
    c = (a + (b - a) * (abs(t) ** 0.7)).astype(int)
    return f"rgb({c[0]},{c[1]},{c[2]})"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w: int, h: int) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="Helvetica, Arial, sans-serif">',
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>']


def txt(s, x, y, t, sz=14, col=INK, anc="start", wt="400", extra=""):
    s.append(f'<text x="{x}" y="{y}" font-size="{sz}" fill="{col}" '
             f'text-anchor="{anc}" font-weight="{wt}"{extra}>{esc(t)}</text>')


# ─────────────────────────────────────────────────────────────────────
def fig_b(M, fine_en, n_seed, out: Path) -> None:
    """① 產業 x 產業的區塊均值 + 色標尺。"""
    W, H = 1180, 900
    groups = sorted({g for g in fine_en if fine_en.count(g) >= 2},
                    key=lambda g: ({"Electronics": 0, "Financials": 1, "Other": 2}[coarse(g)], g))
    gi = {g: np.array([f == g for f in fine_en]) for g in groups}
    K = len(groups)
    x0, y0, side = 390, 150, 430
    cell = side / K
    s = svg(W, H)

    txt(s, 56, 52, "Industry structure emerges in B without any industry label", 24, INK, "start", "600")
    txt(s, 56, 82, "Columns of the cross-layer weight matrix B, averaged into industry x industry blocks.", 15, MUTE)
    txt(s, 56, 106, "Each column is one Taiwanese stock's loading on the 30 US stocks.", 15, MUTE)

    Mn = M.copy()
    np.fill_diagonal(Mn, np.nan)
    for r, ga in enumerate(groups):
        for c_, gb in enumerate(groups):
            v = float(np.nanmean(Mn[np.ix_(gi[ga], gi[gb])]))
            s.append(f'<rect x="{x0+c_*cell:.1f}" y="{y0+r*cell:.1f}" '
                     f'width="{cell:.1f}" height="{cell:.1f}" fill="{heat(v)}" '
                     f'stroke="#ffffff" stroke-width="1"/>')
            if r == c_:
                txt(s, x0 + c_ * cell + cell / 2, y0 + r * cell + cell / 2 + 5,
                    f"{v:+.2f}", 13, "#ffffff", "middle", "700")
        txt(s, x0 - 12, y0 + r * cell + cell / 2 + 5,
            f"{ga} ({int(gi[ga].sum())})", 13, GC[coarse(ga)], "end", "600")
    for c_, gb in enumerate(groups):
        cx = x0 + c_ * cell + cell / 2
        s.append(f'<text x="{cx}" y="{y0+side+14}" font-size="12" fill="{GC[coarse(gb)]}" '
                 f'text-anchor="end" font-weight="600" '
                 f'transform="rotate(-40 {cx} {y0+side+14})">{esc(gb)}</text>')
    s.append(f'<rect x="{x0}" y="{y0}" width="{side}" height="{side}" '
             f'fill="none" stroke="{LINE}" stroke-width="1"/>')

    # 色標尺
    bx, by, bw, bh = x0, y0 + side + 150, side, 20
    txt(s, bx, by - 14, "Mean cosine similarity between columns of B", 14, INK, "start", "600")
    for k in range(240):
        v = -0.6 + 1.2 * k / 239
        s.append(f'<rect x="{bx+bw*k/240:.2f}" y="{by}" width="{bw/240+0.6:.2f}" '
                 f'height="{bh}" fill="{heat(v)}"/>')
    s.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="none" '
             f'stroke="{LINE}" stroke-width="1"/>')
    for v in (-0.6, -0.3, 0.0, 0.3, 0.6):
        cx = bx + bw * (v + 0.6) / 1.2
        s.append(f'<line x1="{cx:.1f}" y1="{by+bh}" x2="{cx:.1f}" y2="{by+bh+5}" '
                 f'stroke="{MUTE}" stroke-width="1"/>')
        txt(s, cx, by + bh + 20, f"{v:+.1f}", 13, MUTE, "middle")
    txt(s, bx - 8, by + bh / 2 + 5, "opposite", 12.5, MUTE, "end")
    txt(s, bx + bw + 8, by + bh / 2 + 5, "aligned", 12.5, MUTE, "start")

    y = by + bh + 58
    txt(s, 56, y, "Same industry +0.2610     Different industry -0.0378     "
                  "Separation +0.2988", 16, INK, "start", "600")
    txt(s, 56, y + 26, f"Positive in {n_seed}/{n_seed} seeds. "
                       "Permutation test (industry labels shuffled 1000x): p <= 0.001.", 15, MUTE)
    txt(s, 56, y + 50, "The diagonal is uniformly warm (+0.20 to +0.58); off-diagonal cells are near zero.", 15, MUTE)
    txt(s, 56, H - 26, "Industries with n >= 2 shown (9 of 18). Cell = mean cosine over all "
                       "cross-pairs of the two industries; diagonal = within-industry mean.", 13, MUTE)
    s.append("</svg>")
    out.write_text("\n".join(s), encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)}  ({W}x{H}, {W/H:.2f}:1)")


# ─────────────────────────────────────────────────────────────────────
def fig_tilt(w1, e1, w2, cols, fine_en, n_seed, out: Path) -> None:
    """② w̄ 的逐檔長條 + x 軸單位 + 誤差棒說明。"""
    W, H = 1180, 960
    o = np.argsort(-w1)
    rx, ry, rw, rh = 330, 190, 620, 560
    lo = float(min(w1.min(), np.nanmin(w2), (w1 - e1).min()))
    hi = float(max(w1.max(), np.nanmax(w2), (w1 + e1).max()))
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    X = lambda v: rx + (v - lo) / (hi - lo) * rw
    bh = rh / len(cols)
    s = svg(W, H)

    txt(s, 56, 52, "The model holds a persistent cross-sectional tilt", 24, INK, "start", "600")
    txt(s, 56, 82, "w-bar = time average of the daily cross-sectionally standardized prediction, "
                   "per Taiwanese stock.", 15, MUTE)
    txt(s, 56, 106, "Long semiconductors, short telecom. Same sign in both folds.", 15, MUTE)

    # 圖例
    lx = 640
    for k, g in enumerate(("Electronics", "Financials", "Other")):
        s.append(f'<rect x="{lx+k*128}" y="{ry-38}" width="12" height="12" fill="{GC[g]}"/>')
        txt(s, lx + k * 128 + 18, ry - 28, g, 13, MUTE)

    s.append(f'<line x1="{X(0):.1f}" y1="{ry}" x2="{X(0):.1f}" y2="{ry+rh}" '
             f'stroke="{LINE}" stroke-width="1.2"/>')
    for k, j in enumerate(o):
        yy = ry + k * bh
        g = coarse(fine_en[j])
        xa, xb = (X(0), X(w1[j])) if w1[j] >= 0 else (X(w1[j]), X(0))
        s.append(f'<rect x="{xa:.1f}" y="{yy+0.9:.1f}" width="{max(xb-xa,0.6):.1f}" '
                 f'height="{bh-1.8:.1f}" fill="{GC[g]}" opacity="0.88"/>')
        s.append(f'<line x1="{X(w1[j]-e1[j]):.1f}" y1="{yy+bh/2:.1f}" '
                 f'x2="{X(w1[j]+e1[j]):.1f}" y2="{yy+bh/2:.1f}" '
                 f'stroke="{INK}" stroke-width="1.0" opacity="0.45"/>')
        if np.isfinite(w2[j]):
            s.append(f'<circle cx="{X(w2[j]):.1f}" cy="{yy+bh/2:.1f}" r="2.8" '
                     f'fill="none" stroke="{INK}" stroke-width="1.1" opacity="0.8"/>')
        if cols[j] in NAMED:
            # 標籤要讓開這一列畫出來的最外緣——誤差棒的端點與第二折的圓，
            # 否則負值那幾檔的圓圈會壓在字上
            ends = [w1[j] - e1[j], w1[j] + e1[j]]
            if np.isfinite(w2[j]):
                ends.append(w2[j])
            sx = (X(max(ends)) + 16) if w1[j] >= 0 else (X(min(ends)) - 16)
            txt(s, sx, yy + bh / 2 + 4, f"{cols[j]} {NAMED[cols[j]]}", 12.5, INK,
                "start" if w1[j] >= 0 else "end", "600")

    for v in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4):
        cx = X(v)
        s.append(f'<line x1="{cx:.1f}" y1="{ry+rh}" x2="{cx:.1f}" y2="{ry+rh+5}" '
                 f'stroke="{MUTE}" stroke-width="1"/>')
        txt(s, cx, ry + rh + 22, f"{v:+.1f}", 13, MUTE, "middle")
    txt(s, rx + rw / 2, ry + rh + 48,
        "persistent tilt  w-bar   (units: cross-sectional standard deviations of the daily prediction)",
        15, INK, "middle", "600")

    # 圖例樣本：橫線 = ±1 跨種子 sd
    ex, ey = 56, ry + 40
    txt(s, ex, ey - 22, "How to read one row", 14, INK, "start", "600")
    s.append(f'<rect x="{ex}" y="{ey+4}" width="86" height="11" fill="{GC["Electronics"]}" opacity="0.88"/>')
    txt(s, ex + 96, ey + 13, "bar: fold-1 mean", 12.5, MUTE)
    s.append(f'<line x1="{ex+18}" y1="{ey+44}" x2="{ex+104}" y2="{ey+44}" '
             f'stroke="{INK}" stroke-width="1.0" opacity="0.45"/>')
    txt(s, ex, ey + 68, "grey horizontal line:", 12.5, MUTE)
    txt(s, ex, ey + 86, "+/- 1 SD across the 10 seeds", 12.5, INK, "start", "600")
    txt(s, ex, ey + 104, "(not a confidence interval)", 12.5, MUTE)
    s.append(f'<circle cx="{ex+61}" cy="{ey+134}" r="2.8" fill="none" '
             f'stroke="{INK}" stroke-width="1.1" opacity="0.8"/>')
    txt(s, ex, ey + 158, "open circle: fold 2", 12.5, INK, "start", "600")
    txt(s, ex, ey + 176, "(non-overlapping test window)", 12.5, MUTE)

    y = ry + rh + 92
    txt(s, 56, y, "Semiconductors vs rest    dw-bar +0.2227 (fold 1) / +0.1867 (fold 2), "
                  f"{n_seed}/{n_seed} seeds each", 16, INK, "start", "600")
    txt(s, 56, y + 26, "Reproducibility of w-bar: corr +0.751 across seeds, +0.718 across folds "
                       "(random-direction control SD 0.146).", 15, MUTE)
    txt(s, 56, H - 28, "Cross-seed SD is 0.064 at the median against a cross-sectional SD of 0.127 "
                       "in w-bar, so read groups, not individual pairs.", 13, MUTE)
    s.append("</svg>")
    out.write_text("\n".join(s), encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)}  ({W}x{H}, {W/H:.2f}:1)")


def main() -> None:
    ap = argparse.ArgumentParser(description="兩個可解釋性證據的投影片用圖（英文）")
    ap.add_argument("--arm", default="tw50_betaF1nA2r1")
    ap.add_argument("--arm-f2", default="f2_best")
    ap.add_argument("--outdir", default="docs/figures")
    a = ap.parse_args()

    u = load_universe("tw50")
    tw = [str(c) for c in u.tw_nodes]
    fine_cn = [u.industry.get(c, "其他") for c in tw]
    unknown = sorted({g for g in fine_cn if g not in IND_EN})
    if unknown:
        raise SystemExit(f"IND_EN 缺少這些產業的英文：{unknown}")
    fine_en = [IND_EN[g] for g in fine_cn]

    mats = []
    for _, d in sorted(bic.find_seeds(a.arm).items(), key=lambda kv: int(kv[0])):
        B = bic.load_B(d)
        if B is not None:
            mats.append(bic.separation(B, fine_cn)[3])
    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fig_b(np.mean(mats, axis=0), fine_en, len(mats), outdir / "interp_b_industry_en.svg")

    _, W1, cols = stl.tilts(a.arm)
    _, W2, _ = stl.tilts(a.arm_f2, fold2=True)
    idx = [cols.index(c) for c in tw]      # 對齊到 universe 的順序
    w1, e1 = W1.mean(0)[idx], W1.std(0, ddof=1)[idx]
    w2 = (W2.mean(0)[idx] if W2.size else np.full(len(tw), np.nan))
    fig_tilt(w1, e1, w2, tw, fine_en, len(W1), outdir / "interp_tilt_en.svg")


if __name__ == "__main__":
    main()
