"""
factorial_2x2x2.py — 本版三個改動的 2^3 完整效應分解（proposal §35.3）

**問題**：§35.3 原本只有六個 cell，而 2^3 有八個。缺兩格時
`+0.0079` 到底是 F1 x rank1.0 的二階、無A₂ x rank1.0 的二階、
還是真正的三階交互作用，**在數學上分不出來**。

三個因子（全部相對於 `tw50_beta` 這個 000 基準）：

    A = F1        只留 log_return（vs F3 = log_return + RSI_14 + BB_pos）
    B = 無A₂      graph_ablate=empty_l2，台股層圖只剩 self-loop
    C = rank1.0   loss_weights.rank 0.5 -> 1.0

八格對應的 arm tag 見 CELLS。六個既有 + 兩個補跑（2026-09-15）。

**估計方式**：十顆種子在八格之間是**配對**的（同一組 seed），
所以逐 seed 算完七個對比再跨 seed 取平均，並用配對 t 檢定。
這比把八格當獨立樣本做 ANOVA 更有力，也與本專案其他地方的
跨種子慣例一致。

對比用 ±1 編碼的**效應（effect）**參數化：

    effect_A   = (1/4) * sum over 8 corners of  x_A * y
    effect_AB  = (1/4) * sum of  x_A * x_B * y
    effect_ABC = (1/4) * sum of  x_A * x_B * x_C * y

其中 x = 2*level - 1。這個尺度讀作「把該因子打開，平均讓 y 變多少」，
與迴歸係數參數化（1/8）差一個 2 倍，報表時要講明。

用法：
    .venv/bin/python scripts/factorial_2x2x2.py
    .venv/bin/python scripts/factorial_2x2x2.py --metric RankIC
"""

from __future__ import annotations

import argparse
import glob
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# (A=F1, B=無A₂, C=rank1.0) -> arm tag
CELLS: dict[tuple[int, int, int], str] = {
    (0, 0, 0): "tw50_beta",              # F3 / A₂on  / r0.5   基準
    (1, 0, 0): "tw50_betaF1",            # F1 / A₂on  / r0.5
    (0, 1, 0): "tw50_beta_g_empty_l2",   # F3 / A₂off / r0.5
    (0, 0, 1): "tw50_betaR1",            # F3 / A₂on  / r1.0
    (1, 1, 0): "tw50_betaF1nA2",         # F1 / A₂off / r0.5
    (1, 0, 1): "tw50_betaF1r1",          # F1 / A₂on  / r1.0   2026-09-15 補
    (0, 1, 1): "tw50_betanA2r1",         # F3 / A₂off / r1.0   2026-09-15 補
    (1, 1, 1): "tw50_betaF1nA2r1",       # F1 / A₂off / r1.0   本版
}
NAMES = {"A": "F1", "B": "無A₂", "C": "rank1.0"}


def per_seed(tag: str, metric: str) -> dict[str, float]:
    """逐 seed 的指標值。IC / RankIC 由預測檔重算，ICIR 用逐日 IC 的訊噪比。"""
    out: dict[str, float] = {}
    pat = str(ROOT / "runs" / "**" / f"*{tag}_s*" / "predictions"
              / "test_predictions.csv")
    for f in sorted(glob.glob(pat, recursive=True)):
        m = re.search(rf"{re.escape(tag)}_s(\d+)/", f)
        if not m:
            continue
        df = pd.read_csv(f)
        ic, ric = [], []
        for _, g in df.groupby("target_date"):
            y, p = g.y.to_numpy(float), g.y_hat.to_numpy(float)
            if y.std() <= 1e-12 or p.std() <= 1e-12:
                continue
            ic.append(np.corrcoef(p, y)[0, 1])
            ric.append(stats.spearmanr(p, y).statistic)
        a = np.asarray(ic if metric != "RankIC" else ric, float)
        if metric == "ICIR":
            out[m.group(1)] = float(np.mean(ic) / np.std(ic, ddof=1))
        else:
            out[m.group(1)] = float(a.mean())
    return out


def contrasts(y: dict[tuple[int, int, int], float]) -> dict[str, float]:
    """一顆種子的八格 -> 七個效應。y 的鍵是 (A,B,C)。"""
    eff = {}
    for r in (1, 2, 3):
        for combo in itertools.combinations("ABC", r):
            idx = {"A": 0, "B": 1, "C": 2}
            s = 0.0
            for corner, v in y.items():
                sign = 1.0
                for f in combo:
                    sign *= (2 * corner[idx[f]] - 1)
                s += sign * v
            eff["x".join(combo)] = s / 4.0
    return eff


def _self_check() -> None:
    """對比的符號與尺度必須對得上，錯了整張表都是錯的。發散就中止。"""
    C = list(itertools.product((0, 1), repeat=3))
    cases = [
        ("只有 A", lambda a, b, c: 1.0 * a, {"A": 1.0}),
        ("只有 AB 交互", lambda a, b, c: 1.0 * a * b,
         {"A": 0.5, "B": 0.5, "AxB": 0.5}),
        ("純三階", lambda a, b, c: 1.0 * a * b * c,
         {k: 0.25 for k in ("A", "B", "C", "AxB", "AxC", "BxC", "AxBxC")}),
        ("可加 A+2B+3C", lambda a, b, c: a + 2 * b + 3 * c,
         {"A": 1.0, "B": 2.0, "C": 3.0}),
    ]
    for name, f, exp in cases:
        e = contrasts({c: f(*c) for c in C})
        for k, v in e.items():
            if abs(v - exp.get(k, 0.0)) > 1e-12:
                raise SystemExit(f"[self-check] {name}：{k} 得 {v}，應為 "
                                 f"{exp.get(k, 0.0)}")
    # 還原 y(111) − y(000) 的是**奇數階**：二階項在兩個全同角落之間相消。
    rng = np.random.default_rng(0)
    y = {c: float(rng.normal()) for c in C}
    e = contrasts(y)
    odd = e["A"] + e["B"] + e["C"] + e["AxBxC"]
    if abs(odd - (y[(1, 1, 1)] - y[(0, 0, 0)])) > 1e-12:
        raise SystemExit("[self-check] 奇數階重建失敗")


def main() -> None:
    ap = argparse.ArgumentParser(description="2^3 效應分解（§35.3）")
    ap.add_argument("--metric", choices=["IC", "RankIC", "ICIR"], default="IC")
    args = ap.parse_args()
    _self_check()

    data = {c: per_seed(t, args.metric) for c, t in CELLS.items()}
    missing = [f"{CELLS[c]} {c}" for c, d in data.items() if not d]
    if missing:
        sys.exit("缺這幾格，無法分解：\n  " + "\n  ".join(missing))

    counts = {CELLS[c]: len(d) for c, d in data.items()}
    seeds = sorted(set.intersection(*[set(d) for d in data.values()]), key=int)
    if len(seeds) < 2:
        sys.exit(f"共同種子只有 {len(seeds)} 顆，無法做配對檢定")
    if len(set(counts.values())) > 1 or len(seeds) < max(counts.values()):
        print("八格的種子數不齊——分解只會用共同的那幾顆，"
              "**先把格子補齊再引用這張表**：")
        for t, k in sorted(counts.items(), key=lambda kv: kv[1]):
            print(f"    {t:24s} {k:2d} 顆" + ("   <-- 不足" if k < max(counts.values()) else ""))
        print(f"    共同種子 {len(seeds)} 顆\n")

    print(f"\n指標 = {args.metric}    共同種子 {len(seeds)} 顆："
          f"{' '.join('s' + s for s in seeds)}\n")

    print("八個角落（跨 seed 平均，括號為 ddof=1 的 sd）")
    print(f"  {'A=F1':>5s} {'B=無A₂':>6s} {'C=r1.0':>6s}  {'arm':24s} "
          f"{args.metric:>9s} {'sd':>8s} {'vs 基準':>9s}")
    b0 = np.mean([data[(0, 0, 0)][s] for s in seeds])
    for c in sorted(CELLS, key=lambda k: (k[0], k[1], k[2])):
        v = np.array([data[c][s] for s in seeds])
        print(f"  {c[0]:>5d} {c[1]:>6d} {c[2]:>6d}  {CELLS[c]:24s} "
              f"{v.mean():+9.4f} {v.std(ddof=1):8.4f} {v.mean() - b0:+9.4f}")

    per = [contrasts({c: data[c][s] for c in CELLS}) for s in seeds]
    keys = list(per[0])
    print(f"\n七個效應（±1 編碼的 effect 尺度；配對 t 檢定，n={len(seeds)}）")
    print(f"  {'效應':22s} {'估計':>9s} {'sd':>8s} {'t':>7s} {'配對 p':>9s} {'同號種子':>9s}")
    for k in keys:
        v = np.array([p[k] for p in per])
        t, pv = stats.ttest_1samp(v, 0.0)
        agree = int(max((v > 0).sum(), (v < 0).sum()))
        lab = "x".join(NAMES[f] for f in k.split("x"))
        star = " **" if pv < 0.05 else "   "
        print(f"  {lab:22s} {v.mean():+9.4f} {v.std(ddof=1):8.4f} "
              f"{t:7.2f} {pv:9.4f}{star} {agree:6d}/{len(seeds)}")

    # 還原「本版 − 基準」的是**奇數階**（A、B、C、三階）。
    # 二階項在兩個全同角落之間符號相消，不參與這個重建——
    # 這不是近似，是恆等式，殘差應該是浮點誤差等級。
    corner = np.array([data[(1, 1, 1)][s] - data[(0, 0, 0)][s] for s in seeds])
    odd = np.array([p["A"] + p["B"] + p["C"] + p["AxBxC"] for p in per])
    print(f"\n  重建檢查（恆等式，殘差應為浮點誤差等級）")
    print(f"    本版 − 基準                = {corner.mean():+.6f}")
    print(f"    奇數階和 A+B+C+三階        = {odd.mean():+.6f}"
          f"   殘差 {abs(odd.mean() - corner.mean()):.2e}")
    print("    （二階項在兩個全同角落之間相消，故不入此式）")

    # 條件效應：rank1.0 在不同脈絡下值多少——這是 §35.3 敘事真正要的東西
    print(f"\n  條件效應（比整體 effect 更好讀）")
    for f, lab in (("C", "rank1.0"), ("B", "無A₂"), ("A", "F1")):
        i = {"A": 0, "B": 1, "C": 2}[f]
        print(f"    {lab} 的效果，依其餘兩因子而定：")
        for other in itertools.product((0, 1), repeat=2):
            on, off = [0, 0, 0], [0, 0, 0]
            rest = [j for j in range(3) if j != i]
            for j, v in zip(rest, other):
                on[j] = off[j] = v
            on[i], off[i] = 1, 0
            d = np.array([data[tuple(on)][s] - data[tuple(off)][s] for s in seeds])
            _, pv = stats.ttest_1samp(d, 0.0)
            tag = " / ".join(f"{NAMES[k]}={'開' if v else '關'}"
                             for k, v in zip([("A","B","C")[j] for j in rest], other))
            star = " **" if pv < 0.05 else ""
            print(f"      {tag:24s} {d.mean():+9.4f}  p={pv:.4f}"
                  f"  {(d > 0).sum():2d}/{len(seeds)}{star}")


if __name__ == "__main__":
    main()
