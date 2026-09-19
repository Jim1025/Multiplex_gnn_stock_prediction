"""p9_report.py — 靜態傾斜的因果檢驗（proposal §52.3 P9、§60.2a(k)）

2x2x2折：`A₂ 開/關` x `mse_target raw/zscore` x `第一折/第二折`。

問的是一個**介入**問題，不是相關：§60.2a(k) 量到 `無A₂` 對靜態 IC 的 2³
主效應是 −0.0098（僅第一折），本腳本把 A₂ 換回來直接看傾斜回不回來。

事前判準（寫於跑之前，見 §52.3）：
  主指標 = A₂ 對**靜態 IC** 的主效應，**兩折同向且為正**才算通過。
  次要   = ICIR 是否跟著動。靜態 IC 回來但 ICIR 不動 -> 機制假說證偽。
  交互   = A₂ 與 zscore 若可加，代表兩者修的是不同東西（靜態側 / 動態側）。

用法：
    .venv/bin/python scripts/p9_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import icir_gap as ig   # noqa: E402

# (A₂ 開?, zscore?) -> (第一折 arm, 第二折 arm)
CELLS = {
    (0, 0): ("tw50_betaF1nA2r1", "f2_best"),      # 主結果 arm
    (1, 0): ("tw50_betaF1r1",    "f2_a2on"),      # 只恢復 A₂
    (0, 1): ("tw50_p8zbal",      "f2_p8zbal"),    # 只換目標（= P8b）
    (1, 1): ("tw50_a2onz",       "f2_a2onz"),     # 兩個都做
}
LAB = {(0, 0): "主 arm（A₂ 關、raw）", (1, 0): "A₂ 開、raw",
       (0, 1): "A₂ 關、zscore（P8b）", (1, 1): "A₂ 開、zscore"}


def zc(A):
    s = A.std(1, keepdims=True)
    return (A - A.mean(1, keepdims=True)) / np.where(s > 0, s, np.nan)


def metrics(H, Y) -> dict:
    """單一 run 的指標。靜態 = 押注的時間平均方向單獨去預測。"""
    Z = zc(H)
    w = np.nanmean(Z, 0)
    T = len(Y)
    ic = ig.ic_series(H, Y)
    return dict(ic=float(np.nanmean(ic)), icir=ig.icir(ic),
                static=float(np.nanmean(ig.ic_series(np.tile(w, (T, 1)), Y))),
                dynamic=float(np.nanmean(ig.ic_series(Z - w, Y))),
                frac=float(np.linalg.norm(w) / np.sqrt(np.nanmean(np.sum(Z ** 2, 1)))),
                w=w)


def cell(arm: str, fold: int) -> dict | None:
    pat = (f"runs/**/*{arm}_s*/predictions/test_predictions.csv" if "/" not in arm else arm)
    rs = ig.load(pat, fold == 2)
    if not rs:
        return None
    ms = [metrics(H, Y) for H, Y in rs]
    out = {k: float(np.mean([m[k] for m in ms])) for k in ("ic", "icir", "static", "dynamic", "frac")}
    out["sd_static"] = float(np.std([m["static"] for m in ms], ddof=1)) if len(ms) > 1 else np.nan
    out["sd_icir"] = float(np.std([m["icir"] for m in ms], ddof=1)) if len(ms) > 1 else np.nan
    out["w"] = np.mean([m["w"] for m in ms], axis=0)
    out["per_static"] = {s: m["static"] for s, m in zip(sorted(range(len(ms))), ms)}
    out["n"] = len(ms)
    out["raw"] = ms
    return out


def main() -> None:
    R = {}
    for k, (a1, a2) in CELLS.items():
        for f, a in ((1, a1), (2, a2)):
            R[(k, f)] = cell(a, f)

    missing = [f"{LAB[k]} 第{f}折" for (k, f), v in R.items() if v is None]
    if missing:
        print("缺：" + "、".join(missing) + "\n")

    for f in (1, 2):
        print("=" * 76)
        print(f"第{'一二'[f-1]}折")
        print("=" * 76)
        print(f"  {'':24s} {'n':>3s} {'mean IC':>9s} {'ICIR':>8s} {'靜態 IC':>9s} "
              f"{'動態 IC':>9s} {'w̄ 佔範數':>9s}")
        for k in sorted(CELLS):
            v = R[(k, f)]
            if v is None:
                print(f"  {LAB[k]:24s}  (缺)")
                continue
            print(f"  {LAB[k]:24s} {v['n']:3d} {v['ic']:+9.4f} {v['icir']:8.4f} "
                  f"{v['static']:+9.4f} {v['dynamic']:+9.4f} {v['frac']:9.3f}")
        print()

    # 2x2 的主效應與交互（逐折）
    print("=" * 76)
    print("2x2 的效應：(1/2)Σ x·y。A = 恢復 A₂，Z = mse_target zscore")
    print("=" * 76)
    for metric in ("static", "icir", "ic", "dynamic", "frac"):
        print(f"\n  [{metric}]")
        for f in (1, 2):
            if any(R[(k, f)] is None for k in CELLS):
                print(f"    第{'一二'[f-1]}折  (缺格，跳過)")
                continue
            v = {k: R[(k, f)][metric] for k in CELLS}
            A = (v[(1, 0)] + v[(1, 1)] - v[(0, 0)] - v[(0, 1)]) / 2
            Z = (v[(0, 1)] + v[(1, 1)] - v[(0, 0)] - v[(1, 0)]) / 2
            AZ = (v[(1, 1)] + v[(0, 0)] - v[(1, 0)] - v[(0, 1)]) / 2
            print(f"    第{'一二'[f-1]}折  A {A:+.4f}   Z {Z:+.4f}   AxZ {AZ:+.4f}")

    # 跨折的傾斜方向持久性
    print("\n" + "=" * 76)
    print("傾斜方向的跨折持久性 corr(w̄_第一折, w̄_第二折)")
    print("=" * 76)
    rng = np.random.default_rng(0)
    null = np.std([np.corrcoef(rng.standard_normal(50), rng.standard_normal(50))[0, 1]
                   for _ in range(2000)])
    for k in sorted(CELLS):
        a, b = R[(k, 1)], R[(k, 2)]
        if a is None or b is None:
            continue
        c = float(np.corrcoef(a["w"], b["w"])[0, 1])
        print(f"  {LAB[k]:24s} {c:+.3f}   (z = {c/null:+.1f})")
    print(f"  （隨機對照 sd {null:.3f}）")

    # 逐種子配對：恢復 A₂ 對靜態 IC
    print("\n" + "=" * 76)
    print("恢復 A₂ 的逐種子配對（靜態 IC），raw 與 zscore 各一組")
    print("=" * 76)
    for zf, zlab in ((0, "raw 目標"), (1, "zscore 目標")):
        for f in (1, 2):
            on, off = R[((1, zf), f)], R[((0, zf), f)]
            if on is None or off is None:
                continue
            a = np.array([m["static"] for m in on["raw"]])
            b = np.array([m["static"] for m in off["raw"]])
            n = min(len(a), len(b))
            d = a[:n] - b[:n]
            p = stats.ttest_rel(a[:n], b[:n]).pvalue
            print(f"  {zlab:12s} 第{'一二'[f-1]}折  Δ靜態IC {d.mean():+.4f}  p {p:.4f}  "
                  f"同號 {int(np.sum(np.sign(d) == np.sign(d.mean())))}/{n}")


if __name__ == "__main__":
    main()
