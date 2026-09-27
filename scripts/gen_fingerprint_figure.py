"""gen_fingerprint_figure.py — 資料端 vs 模型端的對照圖（英文）

輸出 docs/figures/interp_fingerprint_en.svg：

  (a) 跨市場殘差領先指紋，算在第一折的**測試窗**（模型從未看過的期間）
  (b) B 的欄間相關（即圖①的內容），同一個版面、同一條色標尺

兩個面板量的是**同一個概念**（兩檔台股對 30 檔美股殘差的反應有多像），
一個直接從報酬算，一個從訓練好的模型讀。資料層只建市場內的圖（A₁、A₂），
跨市場結構從未餵給模型，所以兩者吻合是「模型從資料學到東西」的證據。

圖下的統計量全部由 residual_fingerprint.py 即時算出（兩折），不寫死。
判準與兩折的結果見該腳本的 docstring 與輸出。

版面沿用 gen_interpretability_figure.py 的繪圖工具與配色；那支腳本不動。

用法：
    .venv/bin/python scripts/gen_fingerprint_figure.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gen_interpretability_figure as gif   # noqa: E402  繪圖工具與配色
import residual_fingerprint as rf           # noqa: E402  計算

INK, MUTE, LINE = gif.INK, gif.MUTE, gif.LINE


def block_means(M: np.ndarray, gi: dict, groups: list[str]) -> np.ndarray:
    Mn = M.copy()
    np.fill_diagonal(Mn, np.nan)
    return np.array([[np.nanmean(Mn[np.ix_(gi[a], gi[b])]) for b in groups] for a in groups])


def panel(s, x0, y0, side, V, groups, head, sub):
    K = len(groups)
    cell = side / K
    gif.txt(s, x0, y0 - 40, head, 16, INK, "start", "700")
    gif.txt(s, x0, y0 - 18, sub, 13, MUTE)
    for r in range(K):
        for c in range(K):
            v = float(V[r, c])
            s.append(f'<rect x="{x0+c*cell:.1f}" y="{y0+r*cell:.1f}" width="{cell:.1f}" '
                     f'height="{cell:.1f}" fill="{gif.heat(v)}" stroke="#ffffff" stroke-width="1"/>')
            if r == c:
                # 白字只放在夠深的格子上；接近 0 的格子是白底，要改黑字
                col = "#ffffff" if abs(v) >= 0.25 else INK
                gif.txt(s, x0 + c * cell + cell / 2, y0 + r * cell + cell / 2 + 5,
                        f"{v:+.2f}", 12.5, col, "middle", "700")
    s.append(f'<rect x="{x0}" y="{y0}" width="{side}" height="{side}" fill="none" '
             f'stroke="{LINE}" stroke-width="1"/>')
    for c, g in enumerate(groups):
        cx = x0 + c * cell + cell / 2
        s.append(f'<text x="{cx:.1f}" y="{y0+side+14}" font-size="12" '
                 f'fill="{gif.TONE[gif.tone(g)]}" text-anchor="end" font-weight="600" '
                 f'transform="rotate(-40 {cx:.1f} {y0+side+14})">{gif.esc(g)}</text>')


def main() -> None:
    warnings.filterwarnings("ignore")
    tw, us, fine = rf.universe()
    unknown = sorted({g for g in fine if g not in gif.IND_EN})
    if unknown:
        raise SystemExit(f"IND_EN 缺少這些產業的英文：{unknown}")
    fine_en = [gif.IND_EN[g] for g in fine]
    Y, X, D, ok = rf.returns_panel(tw, us)

    res = {f: rf.analyse(f, arm, f2, Y, X, D, ok, fine) for f, (arm, f2) in rf.FOLDS.items()}
    r1, r2 = res["第一折"], res["第二折"]
    for f, r in res.items():
        m, p = r["測試窗"]["mantel"]
        if rf.verdict(m, p) == "失敗":
            raise SystemExit(f"{f} 的主判準未通過（r {m:+.4f}, p {p:.4f}），不出圖")

    groups = sorted({g for g in fine_en if fine_en.count(g) >= 2},
                    key=lambda g: ({"Electronics": 0, "Financials": 1, "Other": 2}[gif.coarse(g)], g))
    gi = {g: np.array([f == g for f in fine_en]) for g in groups}
    VA = block_means(r1["測試窗"]["M"], gi, groups)
    VB = block_means(r1["M_B"], gi, groups)
    te1, te2 = r1["w"]["測試窗"], r2["w"]["測試窗"]
    tr1, tr2, va1 = r1["w"]["訓練窗"], r2["w"]["訓練窗"], r1["w"]["驗證窗"]

    # 兩個面板**刻意不是同一段時間**：(a) 在模型定型之後、(b) 學自更早的年份。
    # 這是為了擋「模型當然會複製訓練資料」的質疑，代價是 Mantel r 同時混了
    # 「B 學得準不準」與「結構跨年持不持久」兩件事——所以圖下要並列同期的數字
    # 與資料自身跨期的一致性當參考，讀者才拆得開。
    W, H = 1180, 1108
    s = gif.svg(W, H)
    gif.txt(s, 56, 52, f"What B learned from {tr1[0][:4]}-{tr1[1][:4]} is still measurable in later "
                       "returns", 24, INK, "start", "600")
    gif.txt(s, 56, 82, "How similarly two Taiwanese stocks respond to the residual moves of the "
                       "30 US stocks.", 15, MUTE)
    gif.txt(s, 56, 106, "Left: measured from returns after the model was fixed. Right: read off the "
                        "trained model. Same layout, same colour scale.", 15, MUTE)

    y0, side, xa, xb = 204, 390, 278, 716
    panel(s, xa, y0, side, VA, groups, "(a) Measured from returns",
          f"{te1[0]} to {te1[1]}, not used in training or selection")
    panel(s, xb, y0, side, VB, groups, "(b) Learned by MAGNET (matrix B)",
          f"fit {tr1[0][:7]} to {tr1[1][:7]}, selected {va1[0][:7]} to {va1[1][:7]}, "
          f"{r1['n_seed']} seeds")
    cell = side / len(groups)
    for r, g in enumerate(groups):          # 列標籤兩個面板共用
        gif.txt(s, xa - 12, y0 + r * cell + cell / 2 + 5, f"{g} ({int(gi[g].sum())})", 13,
                gif.TONE[gif.tone(g)], "end", "600")

    # 共用色標尺
    bw, bh = 420, 20
    bx, by = (xa + xb + side) / 2 - bw / 2, y0 + side + 170
    gif.txt(s, bx, by - 14, "Mean correlation between two stocks' profiles", 14, INK, "start", "600")
    for k in range(240):
        v = -0.6 + 1.2 * k / 239
        s.append(f'<rect x="{bx+bw*k/240:.2f}" y="{by}" width="{bw/240+0.6:.2f}" '
                 f'height="{bh}" fill="{gif.heat(v)}"/>')
    s.append(f'<rect x="{bx:.1f}" y="{by}" width="{bw}" height="{bh}" fill="none" '
             f'stroke="{LINE}" stroke-width="1"/>')
    for v in (-0.6, -0.3, 0.0, 0.3, 0.6):
        cx = bx + bw * (v + 0.6) / 1.2
        s.append(f'<line x1="{cx:.1f}" y1="{by+bh}" x2="{cx:.1f}" y2="{by+bh+5}" '
                 f'stroke="{MUTE}" stroke-width="1"/>')
        gif.txt(s, cx, by + bh + 20, f"{v:+.1f}", 12.5, MUTE, "middle")
    gif.txt(s, bx - 10, by + 15, "opposed", 13, MUTE, "end")
    gif.txt(s, bx + bw + 10, by + 15, "aligned", 13, MUTE, "start")

    # 統計量（兩折，即時算）
    m1, m2 = r1["測試窗"]["mantel"][0], r2["測試窗"]["mantel"][0]
    pmax = max(r1["測試窗"]["mantel"][1], r2["測試窗"]["mantel"][1])
    sp = max(r1["測試窗"]["sep"][1], r1["B_sep"][1], r2["測試窗"]["sep"][1], r2["B_sep"][1])
    a1 = int((r1["測試窗"]["align"] > 0).sum())
    a2 = int((r2["測試窗"]["align"] > 0).sum())
    y = by + bh + 72
    gif.txt(s, 56, y, f"Agreement over all 1,225 stock pairs (Mantel r): {m1:+.2f} in fold 1, "
                      f"{m2:+.2f} in fold 2, p < {0.001 if pmax < 0.001 else pmax:g} in both.",
            16, INK, "start", "600")
    gif.txt(s, 56, y + 26, f"Reference: same period as training {r1['訓練窗']['mantel'][0]:+.2f} / "
                           f"{r2['訓練窗']['mantel'][0]:+.2f}; market structure itself, training vs "
                           f"test period {r1['測試窗']['stab']:+.2f} / {r2['測試窗']['stab']:+.2f}.",
            15, MUTE)
    gif.txt(s, 56, y + 50, f"Fold 2 repeats this on an earlier split: fit up to {tr2[1][:7]}, "
                           f"measured on {te2[0]} to {te2[1]}.", 15, MUTE)
    gif.txt(s, 56, y + 74, "Industry separation (same minus different industry), measured / learned: "
                           f"fold 1 {r1['測試窗']['sep'][0]:+.2f} / {r1['B_sep'][0]:+.2f}, "
                           f"fold 2 {r2['測試窗']['sep'][0]:+.2f} / {r2['B_sep'][0]:+.2f}, "
                           f"all p <= {sp:.3f}.", 15, MUTE)
    gif.txt(s, 56, y + 98, f"Per stock, the learned and measured profiles are positively correlated "
                           f"for {a1}/50 (fold 1) and {a2}/50 (fold 2) stocks.", 15, MUTE)
    # 穩健性比的是「兩種估計法在同一段訓練窗上」，不是跟面板 (a) 比，措辭要講清楚
    gif.txt(s, 56, y + 122, "Estimator check: a multivariate ridge instead of pairwise correlations "
                           f"gives the same structure (r = {r1['ridge']['vs_F']:+.2f} / "
                           f"{r2['ridge']['vs_F']:+.2f}, training window).", 15, MUTE)

    fy = y + 164
    gif.txt(s, 56, fy, "Left: correlation of each Taiwanese stock's return on day t with each US "
                       "stock's return on the previous shared trading day, both net of their market's daily mean.",
            12.5, MUTE)
    gif.txt(s, 56, fy + 18, "Right: columns of B, which starts at zero; industry labels never enter "
                            "training. Cells average over all cross-pairs of the two industries.", 12.5, MUTE)
    gif.txt(s, 56, fy + 36, "Industries with n >= 2 shown (9 of 18); Cement and Electronic Components "
                            "are one pair each.", 12.5, MUTE)
    # 原寫「Zero-volume Taiwan non-trading days」——6 天裡只有 3 天休市，另 3 天有交易、
    # 只是資料商缺資料（證交所 FMTQIK 查證，§55.9(h)）。
    gif.txt(s, 56, fy + 54, "Days on which all 50 Taiwanese returns are zero (market closures or missing "
                            "vendor data) are excluded.", 12.5, MUTE)
    s.append("</svg>")
    out = ROOT / "docs" / "figures" / "interp_fingerprint_en.svg"
    out.write_text("\n".join(s), encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)}  ({W}x{H})")
    print(f"   (a) 對角 {np.round(np.diag(VA), 2).tolist()}")
    print(f"   (b) 對角 {np.round(np.diag(VB), 2).tolist()}")


if __name__ == "__main__":
    main()
