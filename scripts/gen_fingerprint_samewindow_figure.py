"""gen_fingerprint_samewindow_figure.py — 同一段窗口、同一條公式：真實資料 vs 模型（英文）

輸出 docs/figures/interp_fingerprint_samewindow_en.svg：

  (a) 真實報酬：第一折**測試窗**的跨市場殘差領先指紋（與 interp_fingerprint_en.svg 的 (a) 同一套算法）
  (b) 模型：把 (a) 公式裡的「台股實際報酬」換成「MAGNET 對同一天的預測」（10 顆種子平均），其餘一字不改

與 gen_fingerprint_figure.py 的差別（那張圖與那支腳本都不動）：
  那張圖的右邊是 B 本身——學自 2019–2023 的係數，和左邊的 2025 年相關係數比，
  窗口不同、公式也不同（B 是原始係數；相關係數還乘上了那段期間美股之間的共變結構）。
  本圖兩邊同一段窗口、同一條公式，唯一的差別是「實際報酬」對「模型預測」。
  模型預測只用台股開盤前就有的輸入，從未碰到當天的實際報酬，所以沒有洩漏。

色標尺：兩邊共用 ±1.0（相關係數的自然界線）。模型那邊會比較深——預測裡沒有實際報酬中
預測不到的雜訊，而那些雜訊會把實際的相關係數往 0 拉。若沿用原圖的 ±0.6，第一折模型那邊
有 32/81 格會被截斷。

雜訊上限：測試窗的天數隨機對半拆，兩半各算一次 (a)，取 50 次平均後做 Spearman-Brown 校正，
得到整年那張圖的可信度；任何模型對 (a) 的一致性上限是它的平方根。

圖下的統計量全部即時算（兩折），不寫死。任一折的 Mantel 不顯著或不為正就不出圖。

只讀 data/processed 與 runs/ 的預測檔、checkpoint，不寫入任何資料層檔案。

用法：
    .venv/bin/python scripts/gen_fingerprint_samewindow_figure.py              # 原主結果（不帶關係）
    .venv/bin/python scripts/gen_fingerprint_samewindow_figure.py --arm tw50_rgcnSA --arm-f2 f2_rgcnSA \
        --out docs/figures/sa/interp_fingerprint_samewindow_en.svg              # 主結果（SA，§83）
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import beta_industry_clustering as bic      # noqa: E402
import gen_interpretability_figure as gif   # noqa: E402  繪圖工具與配色
import residual_fingerprint as rf           # noqa: E402  計算

INK, MUTE, LINE = gif.INK, gif.MUTE, gif.LINE
VMAX = 1.0          # 共用色標尺的上下界
N_SPLIT = 50        # 雜訊上限：隨機對半拆的次數
SEED = 42


def profile(Yv: np.ndarray, Xv: np.ndarray) -> np.ndarray:
    """與 rf.fingerprint 同一條公式：F[i, j] = corr(台股端_j,t , 美股殘差_i,t−1)，[30, 50]。"""
    Yz = (Yv - Yv.mean(0)) / Yv.std(0)
    Xz = (Xv - Xv.mean(0)) / Xv.std(0)
    return Xz.T @ Yz / len(Yv)


def pair_corr(Ma: np.ndarray, Mb: np.ndarray) -> float:
    iu = np.triu_indices(len(Ma), 1)
    return float(np.corrcoef(Ma[iu], Mb[iu])[0, 1])


def reliability(Yv: np.ndarray, Xv: np.ndarray, rng) -> float:
    """天數隨機對半拆，兩半各算相似度矩陣，相關取平均後做 Spearman-Brown。"""
    n, rs = len(Yv), []
    for _ in range(N_SPLIT):
        p = rng.permutation(n)
        a, b = p[: n // 2], p[n // 2: 2 * (n // 2)]
        rs.append(pair_corr(rf.similarity(profile(Yv[a], Xv[a])),
                            rf.similarity(profile(Yv[b], Xv[b]))))
    r = float(np.mean(rs))
    return 2 * r / (1 + r)


def predictions(arm: str, tw: list[str]) -> tuple[pd.DataFrame, int]:
    """該 arm 各種子的測試窗預測，跨種子平均。欄 = 台股代號（與 tw 同序）。"""
    Hs = []
    for _, d in sorted(bic.find_seeds(arm).items(), key=lambda kv: int(kv[0])):
        f = glob.glob(f"{d}/predictions/test_predictions.csv")
        if not f:
            continue
        df = pd.read_csv(f[0], dtype={"ticker": str})
        df["target_date"] = df["target_date"].astype(str).str[:10]
        Hs.append(df.pivot(index="target_date", columns="ticker", values="y_hat").reindex(columns=tw))
    return sum(Hs) / len(Hs), len(Hs)


def analyse(arm: str, fold2: bool, Y, X, D, ok, tw, fine, rng) -> dict:
    w = rf.windows(arm, fold2)
    H, n_seed = predictions(arm, tw)
    mk = rf.in_window(D, w, "測試窗") & ok & np.isin(D, H.index)
    Yh = H.loc[D[mk]].to_numpy(float)
    if not np.isfinite(Yh).all():
        raise SystemExit(f"{arm} 的預測在測試窗有缺值")
    Yh = Yh - Yh.mean(1, keepdims=True)                 # 與實際報酬同樣扣當日橫截面均值
    Yv, Xv = Y[mk], X[mk]
    Fd, Fm = profile(Yv, Xv), profile(Yh, Xv)
    Md, Mm = rf.similarity(Fd), rf.similarity(Fm)
    rel = reliability(Yv, Xv, rng)
    r, p = rf.mantel(Mm, Md)
    PD, PM = rf.unit_cols(Fd), rf.unit_cols(Fm)
    align = np.array([np.corrcoef(PM[:, j], PD[:, j])[0, 1] for j in range(len(tw))])
    M_B, _ = rf.b_similarity(arm, fine)
    return dict(w=w, days=D[mk], n_seed=n_seed, Md=Md, Mm=Mm, mantel=(r, p), rel=rel,
                align=align, med=(float(np.median(np.abs(Fd))), float(np.median(np.abs(Fm)))),
                b_ref=rf.mantel(M_B, Md)[0])


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
                     f'height="{cell:.1f}" fill="{gif.heat(v, VMAX)}" stroke="#ffffff" stroke-width="1"/>')
            if r == c:
                # 白字只放在夠深的格子上（heat 的深淺是 |v/VMAX|^0.7）。
                # 0.4 時 +0.46 的白字落在中淺藍上、對比不足（實際渲染檢查過），所以拉到 0.55
                col = "#ffffff" if abs(v) >= 0.55 * VMAX else INK
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
    # 2026-10-27 起主結果是 SA（tw50_rgcnSA / f2_rgcnSA，§83）；預設仍是原主結果，與 residual_fingerprint.py 相同
    ap = argparse.ArgumentParser(description="同窗指紋圖：真實資料 vs 模型")
    ap.add_argument("--arm", default=rf.FOLDS["第一折"][0])
    ap.add_argument("--arm-f2", default=rf.FOLDS["第二折"][0])
    ap.add_argument("--out", default="docs/figures/interp_fingerprint_samewindow_en.svg")
    args = ap.parse_args()
    rf.FOLDS["第一折"], rf.FOLDS["第二折"] = (args.arm, False), (args.arm_f2, True)
    warnings.filterwarnings("ignore")
    tw, us, fine = rf.universe()
    unknown = sorted({g for g in fine if g not in gif.IND_EN})
    if unknown:
        raise SystemExit(f"IND_EN 缺少這些產業的英文：{unknown}")
    fine_en = [gif.IND_EN[g] for g in fine]
    Y, X, D, ok = rf.returns_panel(tw, us)
    rng = np.random.default_rng(SEED)

    res = {f: analyse(arm, f2, Y, X, D, ok, tw, fine, rng) for f, (arm, f2) in rf.FOLDS.items()}
    r1, r2 = res["第一折"], res["第二折"]
    for f, r in res.items():
        m, p = r["mantel"]
        if m <= 0 or p >= 0.05:
            raise SystemExit(f"{f} 的 Mantel 未通過（r {m:+.4f}, p {p:.4f}），不出圖")

    groups = sorted({g for g in fine_en if fine_en.count(g) >= 2},
                    key=lambda g: ({"Electronics": 0, "Financials": 1, "Other": 2}[gif.coarse(g)], g))
    gi = {g: np.array([f == g for f in fine_en]) for g in groups}
    VA, VB = block_means(r1["Md"], gi, groups), block_means(r1["Mm"], gi, groups)
    te1, tr1 = r1["w"]["測試窗"], r1["w"]["訓練窗"]
    tr2 = r2["w"]["訓練窗"]
    d1, d2 = r1["days"], r2["days"]

    W, H = 1180, 1108
    s = gif.svg(W, H)
    # 標題刻意用 largely：一致性 +0.56 / +0.66，上限是 0.82 / 0.88，不是「一模一樣」
    gif.txt(s, 56, 52, "MAGNET's predicted responses to US moves largely match the market's",
            24, INK, "start", "600")
    gif.txt(s, 56, 82, "How similarly two Taiwanese stocks respond to the residual moves of the 30 US stocks, "
                       "in a period the model never saw.", 15, MUTE)
    gif.txt(s, 56, 106, "Left: from realized returns. Right: the same formula applied to MAGNET's predictions "
                        "for the same days. Same layout, same colour scale.", 15, MUTE)

    y0, side, xa, xb = 204, 390, 278, 716
    panel(s, xa, y0, side, VA, groups, "(a) Realized returns",
          f"{d1[0]} to {d1[-1]}, {len(d1)} trading days")
    panel(s, xb, y0, side, VB, groups, "(b) MAGNET's predictions",
          f"same days, same formula; fit {tr1[0][:7]} to {tr1[1][:7]}, {r1['n_seed']} seeds")
    cell = side / len(groups)
    for r, g in enumerate(groups):          # 列標籤兩個面板共用
        gif.txt(s, xa - 12, y0 + r * cell + cell / 2 + 5, f"{g} ({int(gi[g].sum())})", 13,
                gif.TONE[gif.tone(g)], "end", "600")

    # 共用色標尺
    bw, bh = 420, 20
    bx, by = (xa + xb + side) / 2 - bw / 2, y0 + side + 170
    gif.txt(s, bx, by - 14, "Mean correlation between two stocks' profiles", 14, INK, "start", "600")
    for k in range(240):
        v = -VMAX + 2 * VMAX * k / 239
        s.append(f'<rect x="{bx+bw*k/240:.2f}" y="{by}" width="{bw/240+0.6:.2f}" '
                 f'height="{bh}" fill="{gif.heat(v, VMAX)}"/>')
    s.append(f'<rect x="{bx:.1f}" y="{by}" width="{bw}" height="{bh}" fill="none" '
             f'stroke="{LINE}" stroke-width="1"/>')
    for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
        cx = bx + bw * (v + VMAX) / (2 * VMAX)
        s.append(f'<line x1="{cx:.1f}" y1="{by+bh}" x2="{cx:.1f}" y2="{by+bh+5}" '
                 f'stroke="{MUTE}" stroke-width="1"/>')
        gif.txt(s, cx, by + bh + 20, f"{v:+.1f}", 12.5, MUTE, "middle")
    gif.txt(s, bx - 10, by + 15, "opposed", 13, MUTE, "end")
    gif.txt(s, bx + bw + 10, by + 15, "aligned", 13, MUTE, "start")

    # 統計量（兩折，即時算）
    m1, m2 = r1["mantel"][0], r2["mantel"][0]
    pmax = max(r1["mantel"][1], r2["mantel"][1])
    c1, c2 = np.sqrt(r1["rel"]), np.sqrt(r2["rel"])
    a1, a2 = int((r1["align"] > 0).sum()), int((r2["align"] > 0).sum())
    y = by + bh + 72
    gif.txt(s, 56, y, f"Agreement over all 1,225 stock pairs (Mantel r): {m1:+.2f} in fold 1, "
                      f"{m2:+.2f} in fold 2, p < {0.001 if pmax < 0.001 else pmax:g} in both.",
            16, INK, "start", "600")
    gif.txt(s, 56, y + 26, f"Noise ceiling {c1:.2f} / {c2:.2f}: realized returns only partly agree with "
                           f"themselves (split halves of the days). Corrected agreement "
                           f"{m1 / c1:+.2f} / {m2 / c2:+.2f}.", 15, MUTE)
    gif.txt(s, 56, y + 50, f"Fold 2 repeats this on an earlier split: fit up to {tr2[1][:7]}, "
                           f"measured on {d2[0]} to {d2[-1]}.", 15, MUTE)
    gif.txt(s, 56, y + 74, f"Per stock, the model's and the market's profiles point the same way for "
                           f"{a1}/50 (fold 1) and {a2}/50 (fold 2) stocks.", 15, MUTE)
    gif.txt(s, 56, y + 98, "Right is darker: predictions lack the noise that pulls realized correlations "
                           f"toward zero (median |corr| {r1['med'][0]:.2f} / {r2['med'][0]:.2f} "
                           f"vs {r1['med'][1]:.2f} / {r2['med'][1]:.2f}).", 15, MUTE)
    gif.txt(s, 56, y + 122, f"For reference, this model's matrix B alone agrees with the same realized matrix at "
                            f"{r1['b_ref']:+.2f} / {r2['b_ref']:+.2f}.", 15, MUTE)

    fy = y + 164
    gif.txt(s, 56, fy, "Both panels: correlation of each Taiwanese stock's day-t value with each US stock's "
                       "return on the previous shared trading day, both net of their market's daily mean.",
            12.5, MUTE)
    gif.txt(s, 56, fy + 18, "Right: average of MAGNET's out-of-sample predictions (the whole model, not only B), "
                            "using only inputs available before the Taiwan open.", 12.5, MUTE)
    gif.txt(s, 56, fy + 36, "Cells average over all cross-pairs of the two industries. Industries with n >= 2 "
                            "shown (9 of 18); Cement and Electronic Components are one pair each.", 12.5, MUTE)
    gif.txt(s, 56, fy + 54, "Days on which all 50 Taiwanese returns are zero (market closures or missing "
                            "vendor data) are excluded.", 12.5, MUTE)
    s.append("</svg>")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(s), encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)}  ({W}x{H})")
    for f, r in res.items():
        print(f"   {f}: Mantel {r['mantel'][0]:+.4f} (p {r['mantel'][1]:.4f}), "
              f"reliability {r['rel']:.3f}, ceiling {np.sqrt(r['rel']):.3f}, "
              f"aligned {int((r['align'] > 0).sum())}/50, B ref {r['b_ref']:+.4f}")
    print(f"   (a) 對角 {np.round(np.diag(VA), 2).tolist()}")
    print(f"   (b) 對角 {np.round(np.diag(VB), 2).tolist()}")


if __name__ == "__main__":
    main()
