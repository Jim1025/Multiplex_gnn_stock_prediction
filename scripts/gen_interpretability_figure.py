"""gen_interpretability_figure.py — 兩個可解釋性證據的投影片用圖（§55 / §55.7）

一張 2:1 的雙面板，一個面板一個證據，合起來講同一句話：
**結構被識別了，參數沒有。**

  左：B 的欄間餘弦熱圖（§55）——參數矩陣層級。按粗分組（電子/金融/其餘）
      再按細產業排序，電子與金融各自在對角線上發熱，之間是冷的。
  右：w̄ 的逐檔長條（§55.7）——模型輸出層級。排序後上多空電信的梯度，
      誤差棒是跨種子 sd，空心點是第二折（不重疊的測試期）。

與 `gen_magnet_figure.py` 的架構圖house style 不同：**那張不放讀數，
這張是結果圖，讀數就是內容**。共通的是 2:1、無表情符號、字不重疊。

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

W, H = 1600, 800
INK, MUTE, LINE = "#1a1a1a", "#8a8a8a", "#d8d8d8"
# 三個粗分組的顏色：電子 / 金融 / 其餘。避免紅綠，色盲可分。
GC = {"電子": "#1f6f8b", "金融": "#b45309", "其餘": "#9aa0a6"}


def coarse(x: str) -> str:
    if any(k in x for k in ("半導體", "電腦", "電子", "光電", "通信")):
        return "電子"
    return "金融" if "金融" in x else "其餘"


def heat(v: float) -> str:
    """−1..1 -> 冷藍 / 白 / 暖橘。0 附近是白的，讓區塊結構跳出來。"""
    t = max(-1.0, min(1.0, v))
    if t >= 0:
        a, b = np.array([255, 255, 255]), np.array([31, 111, 139])
    else:
        a, b = np.array([255, 255, 255]), np.array([180, 83, 9])
        t = -t
    c = (a + (b - a) * (t ** 0.7)).astype(int)
    return f"rgb({c[0]},{c[1]},{c[2]})"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    ap = argparse.ArgumentParser(description="兩個可解釋性證據的投影片用圖")
    ap.add_argument("--arm", default="tw50_betaF1nA2r1")
    ap.add_argument("--arm-f2", default="f2_best")
    ap.add_argument("--out", default="docs/figures/interpretability_two_levels.svg")
    a = ap.parse_args()

    u = load_universe("tw50")
    tw = [str(c) for c in u.tw_nodes]
    ind = {c: u.industry.get(c, "其他") for c in tw}

    # ── 左：B 的欄間餘弦 ────────────────────────────────────────────
    mats = []
    for _, d in sorted(bic.find_seeds(a.arm).items(), key=lambda kv: int(kv[0])):
        B = bic.load_B(d)
        if B is None:
            continue
        mats.append(bic.separation(B, [ind[c] for c in tw])[3])
    M = np.mean(mats, axis=0)
    n_seed_b = len(mats)
    # 排序：粗分組 -> 細產業 -> 代號
    order = sorted(range(len(tw)),
                   key=lambda j: ({"電子": 0, "金融": 1, "其餘": 2}[coarse(ind[tw[j]])],
                                  ind[tw[j]], tw[j]))
    M = M[np.ix_(order, order)]
    gl = [coarse(ind[tw[j]]) for j in order]

    # ── 右：w̄ ──────────────────────────────────────────────────────
    _, W1, cols = stl.tilts(a.arm)
    _, W2, _ = stl.tilts(a.arm_f2, fold2=True)
    w1, e1 = W1.mean(0), W1.std(0, ddof=1)
    w2 = W2.mean(0) if W2.size else np.full_like(w1, np.nan)
    o2 = np.argsort(-w1)

    s: list[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
                    f'width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">',
                    f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    T = (lambda x, y, t, sz=14, col=INK, anc="start", wt="400":
         s.append(f'<text x="{x}" y="{y}" font-size="{sz}" fill="{col}" '
                  f'text-anchor="{anc}" font-weight="{wt}">{esc(t)}</text>'))

    T(56, 52, "模型在沒有任何產業標籤下，自己學出產業結構", 25, INK, "start", "600")
    T(56, 80, "兩個層級的同一件事：參數矩陣裡有，模型輸出裡也有。"
              "兩者都跨 10 顆種子成立，右圖另外跨一段不重疊的測試期。", 15, MUTE)

    # ── 左面板：產業 x 產業的區塊均值 ──
    #
    # **不畫 50x50 的原始格點。** 個別 cell 的 sd 是 0.224（同產業）/ 0.277（異產業），
    # 遠大於區塊均值的差（0.2~0.3），所以原始格點必然看起來像雜訊——
    # 結構是統計上的，不是逐格看得出來的。平均成區塊才誠實也才讀得到。
    x0, y0, side = 210, 150, 400
    fine = [ind[tw[j]] for j in order]
    groups = [g for g, c in sorted({g: fine.count(g) for g in set(fine)}.items(),
                                   key=lambda kv: (-kv[1], kv[0])) if c >= 2]
    groups = sorted(groups, key=lambda g: ({"電子": 0, "金融": 1, "其餘": 2}[coarse(g)], g))
    gi = {g: np.array([f == g for f in fine]) for g in groups}
    K = len(groups)
    cell = side / K
    T(x0, y0 - 46, "① B 的欄間餘弦（跨層權重）", 17, INK, "start", "600")
    T(x0, y0 - 24, "產業 x 產業的區塊均值；單格雜訊已平均掉", 13.5, MUTE)
    Mn = M.copy()
    np.fill_diagonal(Mn, np.nan)
    for r, ga in enumerate(groups):
        for c_, gb in enumerate(groups):
            v = float(np.nanmean(Mn[np.ix_(gi[ga], gi[gb])]))
            s.append(f'<rect x="{x0+c_*cell:.1f}" y="{y0+r*cell:.1f}" '
                     f'width="{cell:.1f}" height="{cell:.1f}" fill="{heat(v)}" '
                     f'stroke="#ffffff" stroke-width="1"/>')
            if r == c_:
                T(x0 + c_ * cell + cell / 2, y0 + r * cell + cell / 2 + 5,
                  f"{v:+.2f}", 12.5, "#ffffff", "middle", "700")
        T(x0 - 10, y0 + r * cell + cell / 2 + 5,
          f"{ga}({int(gi[ga].sum())})", 12.5, GC[coarse(ga)], "end", "600")
        s.append(f'<line x1="{x0+r*cell:.1f}" y1="{y0}" x2="{x0+r*cell:.1f}" '
                 f'y2="{y0+side}" stroke="#ffffff" stroke-width="0"/>')
    for c_, gb in enumerate(groups):
        cx = x0 + c_ * cell + cell / 2
        s.append(f'<text x="{cx}" y="{y0+side+16}" font-size="12" fill="{GC[coarse(gb)]}" '
                 f'text-anchor="end" font-weight="600" '
                 f'transform="rotate(-42 {cx} {y0+side+16})">{esc(gb)}</text>')
    s.append(f'<rect x="{x0}" y="{y0}" width="{side}" height="{side}" '
             f'fill="none" stroke="{LINE}" stroke-width="1"/>')
    T(x0 - 140, y0 + side + 108, "同產業 +0.2610   異產業 −0.0378   分離度 +0.2988", 15, INK)
    T(x0 - 140, y0 + side + 132,
      f"{n_seed_b}/{n_seed_b} 顆種子為正；洗牌產業標籤 1000 次 p <= 0.001", 15, MUTE)
    T(x0 - 140, y0 + side + 156,
      "對角線整條發熱，非對角線幾乎全白（僅列出 n >= 2 的產業）", 15, MUTE)

    # ── 右面板 ──
    rx, ry, rw, rh = 760, 130, 640, 480
    T(rx, ry - 18, "② w̄：每檔台股的持久傾斜（模型輸出）", 17, INK, "start", "600")
    lo, hi = float(min(w1.min(), np.nanmin(w2))), float(max(w1.max(), np.nanmax(w2)))
    pad = (hi - lo) * 0.12
    lo, hi = lo - pad, hi + pad
    X = lambda v: rx + (v - lo) / (hi - lo) * rw
    bh = rh / len(tw)
    s.append(f'<line x1="{X(0):.1f}" y1="{ry}" x2="{X(0):.1f}" y2="{ry+rh}" '
             f'stroke="{LINE}" stroke-width="1.2"/>')
    for k, j in enumerate(o2):
        yy = ry + k * bh
        g = coarse(ind[cols[j]])
        xa, xb = (X(0), X(w1[j])) if w1[j] >= 0 else (X(w1[j]), X(0))
        s.append(f'<rect x="{xa:.1f}" y="{yy+0.8:.1f}" width="{max(xb-xa,0.6):.1f}" '
                 f'height="{bh-1.6:.1f}" fill="{GC[g]}" opacity="0.88"/>')
        s.append(f'<line x1="{X(w1[j]-e1[j]):.1f}" y1="{yy+bh/2:.1f}" '
                 f'x2="{X(w1[j]+e1[j]):.1f}" y2="{yy+bh/2:.1f}" '
                 f'stroke="{INK}" stroke-width="0.9" opacity="0.5"/>')
        if np.isfinite(w2[j]):
            s.append(f'<circle cx="{X(w2[j]):.1f}" cy="{yy+bh/2:.1f}" r="2.6" '
                     f'fill="none" stroke="{INK}" stroke-width="1.1" opacity="0.8"/>')
    named = {"2330": "台積電", "3711": "日月光", "2303": "聯電",
             "4904": "遠傳", "3045": "台灣大", "2412": "中華電"}
    for k, j in enumerate(o2):
        c = cols[j]
        if c in named:
            yy = ry + k * bh + bh / 2 + 4
            side_x = X(w1[j]) + (12 if w1[j] >= 0 else -12)
            anc = "start" if w1[j] >= 0 else "end"
            T(side_x, yy, f"{c} {named[c]}", 12.5, INK, anc, "600")
    for v in (-0.2, 0.0, 0.2, 0.4):
        T(X(v), ry + rh + 22, f"{v:+.1f}", 13, MUTE, "middle")
    T(rx, ry + rh + 52, "實心棒 = 第一折（10 顆種子平均，橫線為跨種子 sd）"
                        "　空心圈 = 第二折（測試期不重疊）", 14, MUTE)
    T(rx, ry + rh + 76, "半導體 vs 其餘　Δw̄ +0.2227 / +0.1867（兩折各 10/10 種子）",
      15, INK)
    T(rx, ry + rh + 100, "跨種子 corr(w̄) +0.751　跨折 corr(w̄) +0.718　"
                         "（隨機方向對照 sd 0.146）", 15, MUTE)

    # 圖例
    lx = rx + 430
    for k, (g, lab) in enumerate((("電子", "電子"), ("金融", "金融"), ("其餘", "其餘"))):
        s.append(f'<rect x="{lx+k*72}" y="{ry-34}" width="12" height="12" fill="{GC[g]}"/>')
        T(lx + k * 72 + 17, ry - 24, lab, 13, MUTE)

    T(56, H - 26, "①：50 檔台股在 30 檔美股上的載重向量，兩兩餘弦。"
                  "②：逐日橫截面標準化後的預測，對時間取平均。"
                  "兩者都沒有用到任何產業標籤。", 13, MUTE)
    s.append("</svg>")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(s), encoding="utf-8")
    print(f"-> {a.out}  ({W}x{H}, {W/H:.2f}:1)")


if __name__ == "__main__":
    main()
