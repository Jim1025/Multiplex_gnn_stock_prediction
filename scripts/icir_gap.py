"""icir_gap.py — 主結果為什麼在 ICIR / Rank ICIR 上輸給 KTW+（proposal §60.2a）

兩折都是同一個形狀：**分子我們贏，分母我們輸**。
  第一折 IC +0.1090 vs +0.1077，ICIR 0.4368 vs 0.4810
  第二折 IC +0.1537 vs +0.1482，ICIR 0.6275 vs 0.6651

本腳本用排除法把 sd_t(IC) 的差距逐項拆掉，七個區塊全部可獨立重跑：

  A. 差距的形狀          IC / sd_t / ICIR，兩折、全部種子
  B. 種子雜訊            逐 seed vs 跨 seed 平均 vs 集成預測。
                         若集成後就追平，那只是隨機性，不是訊號問題
  C. 抽樣雜訊 vs 訊號     每天把 n 檔隨機對半，兩半 IC 的歧異估「橫截面抽樣雜訊」，
                         其餘歸「押注報酬的逐日變動」。
                         **注意這個量有地板**：每天押同一個固定向量（RC 常數對照）
                         也會得到約 0.176，所以絕對值無意義，只有相互比較有意義
  D. 換手                 對 ŷ 做因果移動平均（只用當日與過去）。
                         若平滑能拉高 ICIR，代表預測抖；實測是殺 IC 更快
  E. 集中度               winsorize / 名次 / sign。若極端預測是雜訊，壓掉應該變好
  F. 跨 arm               神經 vs 線性都放進來。用來檢查「神經比較不穩」這個說法
  G. 控制實驗             同一組特徵只換擬合目標（原始 y vs 逐日橫截面 rank）。
                         `factor_vs_graph.py` 的 K 前綴就是這件事，
                         第一折有 R2/KR2、第二折有 TW+/KTW+

用法：
    .venv/bin/python scripts/icir_gap.py
    .venv/bin/python scripts/icir_gap.py --blocks A C G
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 第二折的評估窗上界。與 results_table.py:_fold2_rows 的 CUT 同值。
CUT = "2024-12-25"
SEED = 42                 # 半樣本拆分的抽樣種子
HALF_REPEATS = 40         # 每天重複幾次對半拆分

BEST_F1 = "runs/**/*tw50_betaF1nA2r1_s*/predictions/test_predictions.csv"
BEST_F2 = "runs/**/*f2_best_s*/predictions/test_predictions.csv"
KTW_F1 = "runs/**/*fvg_KTWp/predictions/*.csv"
KTW_F2 = "runs_f2/*fvg_KTWp/predictions/*.csv"


def load(pat: str, fold2: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """pattern -> 逐 run 的 (ŷ, y)，形狀都是 [T, n]。"""
    out = []
    for f in sorted(glob.glob(str(ROOT / pat), recursive=True)):
        df = pd.read_csv(f)
        df["target_date"] = df["target_date"].astype(str).str[:10]
        if fold2:
            df = df[df.target_date <= CUT]
        H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
        Y = df.pivot(index="target_date", columns="ticker", values="y").sort_index()
        if len(H) > 3:
            out.append((H.to_numpy(), Y.to_numpy()))
    return out


def ic1(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def ic_series(H, Y) -> np.ndarray:
    return np.array([ic1(H[t], Y[t]) for t in range(len(Y))])


def icir(a) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean() / a.std(ddof=1)) if a.size > 1 else np.nan


def zc(A: np.ndarray) -> np.ndarray:
    """逐日橫截面標準化。"""
    s = A.std(1, keepdims=True)
    return (A - A.mean(1, keepdims=True)) / np.where(s > 0, s, np.nan)


def decompose(H, Y, rng) -> tuple[float, float, float, float]:
    """(mean IC, sd_t, 抽樣雜訊 sd, 押注報酬 sd)。

    半樣本拆分：同一天把 n 檔隨機分兩半，兩半各算一次 IC。
    兩者的差只含抽樣雜訊（當天的真實訊號強度對兩半是同一個），所以
        Var(差) = 2 x Var_雜訊(半樣本)
    而半樣本的 n 是全樣本的一半，相關係數的抽樣變異約與 n 成反比，故
        Var_雜訊(全樣本) ~ Var_雜訊(半樣本) / 2 = Var(差) / 4
    """
    n = Y.shape[1]
    full, dif = [], []
    for t in range(len(Y)):
        v = ic1(H[t], Y[t])
        if not np.isfinite(v):
            continue
        full.append(v)
        for _ in range(HALF_REPEATS):
            p = rng.permutation(n)
            a, b = ic1(H[t, p[: n // 2]], Y[t, p[: n // 2]]), ic1(H[t, p[n // 2:]], Y[t, p[n // 2:]])
            if np.isfinite(a) and np.isfinite(b):
                dif.append(a - b)
    full = np.array(full)
    v_tot = float(full.var(ddof=1))
    v_noise = float(np.array(dif).var(ddof=1)) / 4.0
    return (float(full.mean()), np.sqrt(v_tot), np.sqrt(v_noise),
            np.sqrt(max(v_tot - v_noise, 0.0)))


def agg(runs, fn):
    return np.array([fn(H, Y) for H, Y in runs], float).mean(axis=0)


FOLDS = (("第一折", BEST_F1, KTW_F1, False), ("第二折", BEST_F2, KTW_F2, True))


def block_A():
    print("\n" + "=" * 74)
    print("A. 差距的形狀：分子我們贏，分母我們輸")
    print("=" * 74)
    print(f"\n  {'':22s} {'n':>3s} {'mean IC':>9s} {'sd_t(IC)':>9s} {'ICIR':>8s}")
    for fold, mp, kp, f2 in FOLDS:
        for nm, pat in ((f"{fold} MAGNET 本版", mp), (f"{fold} KTW+", kp)):
            rs = load(pat, f2)
            if not rs:
                print(f"  {nm:22s} (找不到)")
                continue
            r = agg(rs, lambda H, Y: (np.nanmean(ic_series(H, Y)),
                                      np.nanstd(ic_series(H, Y), ddof=1),
                                      icir(ic_series(H, Y))))
            print(f"  {nm:22s} {len(rs):3d} {r[0]:+9.4f} {r[1]:9.4f} {r[2]:8.4f}")
    print("\n  -> 兩折都是 IC 小贏、sd 大輸。整個 ICIR 的差距在分母。")


def block_B():
    print("\n" + "=" * 74)
    print("B. 是不是種子雜訊：不是，只佔 sd 的 3.6%")
    print("=" * 74)
    for fold, mp, kp, f2 in FOLDS:
        rs = load(mp, f2)
        if not rs:
            continue
        IC = np.array([ic_series(H, Y) for H, Y in rs])
        Y = rs[0][1]
        mu = np.nanmean(IC, 0)                       # 跨 seed 平均的逐日 IC
        ens = ic_series(np.array([H for H, _ in rs]).mean(0), Y)   # 先平均 ŷ 再算 IC
        k = load(kp, f2)
        kir = icir(ic_series(*k[0])) if k else np.nan
        print(f"\n  [{fold}]  KTW+ ICIR = {kir:.4f}")
        for nm, a in (("逐 seed（表上的慣例）", None), ("跨 seed 平均的日序列", mu),
                      ("集成預測（先平均 ŷ）", ens)):
            if a is None:
                v = float(np.mean([icir(r) for r in IC]))
                sd = float(np.nanstd(IC, 1, ddof=1).mean())
                m = float(np.nanmean(IC))
            else:
                v, sd, m = icir(a), float(np.nanstd(a, ddof=1)), float(np.nanmean(a))
            print(f"    {nm:22s} IC {m:+.4f}  sd {sd:.4f}  ICIR {v:.4f}"
                  f"  {'仍輸' if v < kir else '贏'}")


def block_C():
    print("\n" + "=" * 74)
    print("C. 抽樣雜訊 vs 押注報酬的逐日變動")
    print("=" * 74)
    print("\n  地板警告：每天押同一個固定向量也會有約 0.176 的押注報酬 sd")
    print("  （見 F 區塊的 RC 常數對照）。絕對值無意義，只比相對大小。")
    rng = np.random.default_rng(SEED)
    print(f"\n  {'':22s} {'mean IC':>9s} {'sd_t':>8s} {'抽樣雜訊':>9s} {'押注報酬 sd':>11s}")
    for fold, mp, kp, f2 in FOLDS:
        for nm, pat in ((f"{fold} MAGNET 本版", mp), (f"{fold} KTW+", kp)):
            rs = load(pat, f2)
            if not rs:
                continue
            r = agg(rs, lambda H, Y: decompose(H, Y, rng))
            print(f"  {nm:22s} {r[0]:+9.4f} {r[1]:8.4f} {r[2]:9.4f} {r[3]:11.4f}")
    print("\n  -> 抽樣雜訊兩邊幾乎相同。差距全部在押注報酬的逐日變動。")


def _smooth(zh, w):
    """因果移動平均：只用當日與過去 w−1 天。"""
    if w <= 1:
        return zh
    return np.array([np.nanmean(zh[max(0, t - w + 1): t + 1], axis=0) for t in range(len(zh))])


def _ic_after(H, Y, f):
    zh = f(zc(H))
    out = np.full(len(Y), np.nan)
    for t in range(len(Y)):
        if np.isfinite(zh[t]).all() and zh[t].std() > 0 and Y[t].std() > 0:
            out[t] = np.corrcoef(zh[t], Y[t])[0, 1]
    return np.nanmean(out), np.nanstd(out, ddof=1), icir(out)


def block_D():
    print("\n" + "=" * 74)
    print("D. 是不是換手太快：不是，平滑殺 IC 比殺 sd 快得多")
    print("=" * 74)
    for fold, mp, kp, f2 in FOLDS:
        M, K = load(mp, f2), load(kp, f2)
        if not M or not K:
            continue
        kir = _ic_after(*K[0], lambda z: z)[2]
        print(f"\n  [{fold}]  KTW+（未平滑）ICIR = {kir:.4f}")
        print(f"    {'平滑窗 w':>9s} {'IC':>9s} {'sd':>8s} {'ICIR':>8s}")
        for w in (1, 2, 3, 5, 10, 20):
            r = agg(M, lambda H, Y, _w=w: _ic_after(H, Y, lambda z: _smooth(z, _w)))
            print(f"    {w:>9d} {r[0]:+9.4f} {r[1]:8.4f} {r[2]:8.4f}")
    print("\n  -> 訊號本身是一天壽命的（§58：77.9% 在隔夜），不是預測在抖。")


def block_E():
    print("\n" + "=" * 74)
    print("E. 是不是押注太集中：不是，壓掉極端值兩邊都變差")
    print("=" * 74)
    TF = {"原樣": lambda z: z,
          "winsor ±1.5": lambda z: np.clip(z, -1.5, 1.5),
          "winsor ±1.0": lambda z: np.clip(z, -1.0, 1.0),
          "橫截面名次": lambda z: np.apply_along_axis(stats.rankdata, 1, z),
          "sign(±1)": lambda z: np.sign(z)}
    for fold, mp, kp, f2 in FOLDS:
        M, K = load(mp, f2), load(kp, f2)
        if not M or not K:
            continue
        print(f"\n  [{fold}]   {'轉換':>14s} {'MAGNET ICIR':>12s} {'KTW+ ICIR':>11s}")
        for nm, f in TF.items():
            m = agg(M, lambda H, Y, _f=f: _ic_after(H, Y, _f))[2]
            k = _ic_after(*K[0], f)[2]
            print(f"  {'':9s}   {nm:>14s} {m:12.4f} {k:11.4f}")
    print("\n  -> KTW+ 自己也一樣掉。極端預測對兩邊都是有資訊的。")


def block_F():
    print("\n" + "=" * 74)
    print("F. 跨 arm：這不是「神經 vs 線性」")
    print("=" * 74)
    ARMS = [("MAGNET 本版（神經）", "runs/**/*tw50_betaF1nA2r1_s*/predictions/test_predictions.csv"),
            ("F1 + rank1.0（神經）", "runs/**/*tw50_betaF1r1_s*/predictions/test_predictions.csv"),
            ("F1 單獨（神經）", "runs/**/*tw50_betaF1_s*/predictions/test_predictions.csv"),
            ("beta 層前版（神經）", "runs/**/*tw50_beta_s*/predictions/test_predictions.csv"),
            ("KTW+（線性）", KTW_F1),
            ("[24] LASSO（線性）", "runs/**/*bipartite*t2_LASSO/predictions/*.csv"),
            ("R2 ridge（線性）", "runs/**/*ridge_R2/predictions/*.csv"),
            ("RC 常數對照（線性）", "runs/**/*ridge_RC/predictions/*.csv")]
    rng = np.random.default_rng(SEED)
    print(f"\n  第一折   {'':22s} {'n':>2s} {'mean IC':>9s} {'抽樣雜訊':>9s} {'押注報酬 sd':>11s} {'ICIR':>8s}")
    rows = []
    for nm, pat in ARMS:
        rs = load(pat)
        if not rs:
            print(f"  {'':9s}{nm:22s} (找不到)")
            continue
        r = agg(rs[:5], lambda H, Y: decompose(H, Y, rng))
        ir = agg(rs[:5], lambda H, Y: (icir(ic_series(H, Y)),))[0]
        rows.append((nm, r[3]))
        print(f"  {'':9s}{nm:22s} {min(len(rs),5):2d} {r[0]:+9.4f} {r[2]:9.4f} {r[3]:11.4f} {ir:8.4f}")
    if rows:
        rows.sort(key=lambda x: x[1])
        print(f"\n  押注報酬 sd 由小到大：{'  <  '.join(n for n, _ in rows)}")
    print("  -> 線性模型分布在最好與最差兩端。KTW+ 是離群值，不是「線性比較穩」。")


def block_G():
    print("\n" + "=" * 74)
    print("G. 控制實驗：同一組特徵，只換擬合目標")
    print("=" * 74)
    print("\n  factor_vs_graph.py 的 K 前綴 = 把 y 先做逐日橫截面 rank（單位變異）再擬合。")
    rng = np.random.default_rng(SEED)
    PAIRS = (("第一折（30 檔美股報酬）",
              [("R2   原始 y 目標", "runs/**/*fvg_R2/predictions/*.csv", False),
               ("KR2  逐日 rank 目標", "runs/**/*fvg_KR2/predictions/*.csv", False)]),
             ("第二折（美股+台股 80 維）",
              [("TW+  原始 y 目標", "runs_f2/*fvg_TWp/predictions/*.csv", True),
               ("KTW+ 逐日 rank 目標", KTW_F2, True)]))
    for fold, pair in PAIRS:
        print(f"\n  [{fold}]")
        print(f"    {'':22s} {'mean IC':>9s} {'sd_t':>8s} {'押注報酬 sd':>11s} {'ICIR':>8s}")
        for nm, pat, f2 in pair:
            rs = load(pat, f2)
            if not rs:
                print(f"    {nm:22s} (找不到)")
                continue
            r = agg(rs, lambda H, Y: decompose(H, Y, rng))
            ir = agg(rs, lambda H, Y: (icir(ic_series(H, Y)),))[0]
            print(f"    {nm:22s} {r[0]:+9.4f} {r[1]:8.4f} {r[3]:11.4f} {ir:8.4f}")
    print("\n  -> rank 目標買到的是**分子**（IC +4% / +11%），sd 幾乎不動。")
    print("     我們的 mse: 1.0 擬合的是原始 y。這一格是 §52.3 P8 的依據。")


BLOCKS = {"A": block_A, "B": block_B, "C": block_C,
          "D": block_D, "E": block_E, "F": block_F, "G": block_G}


def main() -> None:
    ap = argparse.ArgumentParser(description="ICIR 差距的拆解（proposal §60.2a）")
    ap.add_argument("--blocks", nargs="*", default=list(BLOCKS), choices=list(BLOCKS))
    for b in ap.parse_args().blocks:
        BLOCKS[b]()


if __name__ == "__main__":
    main()
