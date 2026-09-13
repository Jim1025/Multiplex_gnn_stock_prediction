"""
portfolio_readout.py — IC 的組合讀法與成本敏感度（proposal §60）

**這不是回測腳本。** 它回答的是「IC = 0.109 在組合上是什麼意思」，
以及「在什麼交易成本下這個訊號還成立」。不宣稱策略績效。

四個區塊：

  A. IC / ICIR / 兩種組合建構的對照
     §57.8 的恆等式：z 加權金額中性組合的日報酬 r_t = IC_t x sigma_t。
     所以 IC 只決定**分子**；若 sigma_t 為常數，Sharpe = ICIR x sqrt(252)。
     與回測 Sharpe 對應的是 ICIR，不是 IC。

  B. Top-K 多空的週轉率與打平成本
     打平成本 = 每日毛報酬 / 每日週轉率。低於這個來回成本才有淨收益。

  C. 成本敏感度
     台灣證交稅為賣出邊 0.3%（= 30 bp/來回），手續費 0.1425%/邊
     （常見折扣後 3~6 bp/邊），借券費另計。**稅率請自行核對現行規定。**

  D. 為什麼 IC 的排名不保證回測的排名
     拆成三個可量測的來源：IC 的逐日波動、整個橫截面 vs 只用兩端、
     選到的股票本身的波動。

全部不重訓，只讀 runs/**/predictions/test_predictions.csv。

用法：
    .venv/bin/python scripts/portfolio_readout.py
    .venv/bin/python scripts/portfolio_readout.py --blocks A D
    .venv/bin/python scripts/portfolio_readout.py --topk 5
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 與 results_table.py 的基準一致
MODELS = [
    ("MAGNET 本版 betaF1nA2r1", "runs/**/*tw50_betaF1nA2r1_s*/predictions/test_predictions.csv"),
    ("MAGNET beta 前版",        "runs/**/*tw50_beta_s*/predictions/test_predictions.csv"),
    ("KTW+ 線性最高標",          "runs/linear/*fvg_KTWp/predictions/test_predictions.csv"),
    ("RC 常數對照",              "runs/linear/*ridge_RC/predictions/test_predictions.csv"),
]
TOPK_DEFAULT = 10
COSTS_BP = (0, 10, 20, 30, 40, 60)


def files_for(pat: str) -> list[str]:
    fs = sorted(glob.glob(str(ROOT / pat), recursive=True))
    # tw50_beta 的 glob 會誤抓 tw50_beta_no1 等，這裡精確過濾
    if "tw50_beta_s" in pat:
        fs = [f for f in fs if re.search(r"tw50_beta_s\d+/", f)]
    return fs


def icir(a: np.ndarray) -> float:
    """ICIR = mean(IC)/sd(IC)，ddof=1，與 src/train/metrics.py:176 同一定義。"""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if a.size < 2:
        return float("nan")
    sd = a.std(ddof=1)
    return float("nan") if sd < 1e-12 else float(a.mean() / sd)


def per_day(df: pd.DataFrame, k: int) -> dict[str, np.ndarray]:
    """逐日算所有需要的量。跳過 std(y)=0 的日子（第一折有 1 天）。"""
    ic, ric, sig, r_z, r_k, hit, sd_long, turn = [], [], [], [], [], [], [], []
    prev: dict = {}
    for d in sorted(df.target_date.unique()):
        g = df[df.target_date == d]
        y = g.y.to_numpy(float)
        p = g.y_hat.to_numpy(float)
        tk = g.ticker.to_numpy()
        if y.std(ddof=0) <= 1e-12 or p.std(ddof=0) <= 1e-12:
            continue
        ic.append(float(np.corrcoef(p, y)[0, 1]))
        ric.append(float(stats.spearmanr(p, y).statistic))
        sig.append(float(y.std(ddof=0)))
        # z 加權、金額中性：用全部節點（§57.8 的恆等式就是這個建構）
        z = (p - p.mean()) / p.std(ddof=0)
        r_z.append(float((z / len(z)) @ y))
        # Top-K 多空，等權
        o = np.argsort(-p)
        L, S = o[:k], o[-k:]
        r_k.append(float(y[L].mean() - y[S].mean()) / 2.0)
        hit.append(float((y[L] >= np.quantile(y, 0.8)).mean()))
        sd_long.append(float(y[L].std(ddof=0)))
        w = {t: 1.0 / (2 * k) for t in tk[L]}
        for t in tk[S]:
            w[t] = w.get(t, 0.0) - 1.0 / (2 * k)
        keys = set(w) | set(prev)
        turn.append(sum(abs(w.get(t, 0.0) - prev.get(t, 0.0)) for t in keys) / 2.0)
        prev = w
    out = {kk: np.array(v) for kk, v in
           dict(ic=ic, ric=ric, sig=sig, r_z=r_z, r_k=r_k,
                hit=hit, sd_long=sd_long, turn=turn).items()}
    out["turn"] = out["turn"][1:]          # 第一天沒有前一日權重
    return out


def collect(k: int) -> dict[str, list[dict]]:
    out = {}
    for lab, pat in MODELS:
        fs = files_for(pat)
        if fs:
            out[lab] = [per_day(pd.read_csv(f), k) for f in fs]
    return out


def avg(runs: list[dict], fn) -> float:
    return float(np.mean([fn(r) for r in runs]))


ANN = np.sqrt(252)


def block_A(data, k):
    print("\n" + "=" * 78)
    print("A. IC / ICIR / 兩種組合建構")
    print("=" * 78)
    print("\n  §57.8：z 加權金額中性組合的日報酬 r_t = IC_t x sigma_t。")
    print("  IC 只決定分子；sigma_t 為常數時 Sharpe = ICIR x sqrt(252)。")
    print(f"\n  {'模型':24s} {'n':>3s} {'IC':>8s} {'RankIC':>8s} {'ICIR':>7s}"
          f" {'ICIRx√252':>10s} {'z加權 Sharpe':>12s} {f'Top-{k} Sharpe':>13s}")
    for lab, runs in data.items():
        n = len(runs)
        ic = avg(runs, lambda r: r["ic"].mean())
        ric = avg(runs, lambda r: r["ric"].mean())
        ir = avg(runs, lambda r: icir(r["ic"]))
        shz = avg(runs, lambda r: r["r_z"].mean() / r["r_z"].std(ddof=1) * ANN)
        shk = avg(runs, lambda r: r["r_k"].mean() / r["r_k"].std(ddof=1) * ANN)
        print(f"  {lab:24s} {n:3d} {ic:+8.4f} {ric:+8.4f} {ir:7.4f}"
              f" {ir * ANN:10.2f} {shz:12.2f} {shk:13.2f}")
    print("\n  ICIR 逐 seed 算完再平均（不是用跨 seed 平均後的日序列——那是集成的 ICIR）。")


def block_B(data, k):
    print("\n" + "=" * 78)
    print(f"B. Top-{k} 多空的週轉率與打平成本（每日再平衡、等權、**無成本**）")
    print("=" * 78)
    print(f"\n  {'模型':24s} {'毛報酬/日':>10s} {'年化':>8s} {'Sharpe':>7s}"
          f" {'週轉/日':>8s} {'打平(來回)':>11s}")
    for lab, runs in data.items():
        g = avg(runs, lambda r: r["r_k"].mean())
        sh = avg(runs, lambda r: r["r_k"].mean() / r["r_k"].std(ddof=1) * ANN)
        t = avg(runs, lambda r: r["turn"].mean())
        be = f"{g / t * 1e4:8.0f} bp" if t > 1e-9 else "    無上限"
        print(f"  {lab:24s} {g * 100:9.4f}% {g * 252 * 100:7.1f}% {sh:7.2f}"
              f" {t * 100:7.1f}% {be:>11s}")
    print("\n  打平成本 = 毛報酬 / 週轉率。來回成本高於此，淨報酬為負。")


def block_C(data, k):
    print("\n" + "=" * 78)
    print("C. 成本敏感度（年化淨報酬 = (毛報酬 − 週轉率 x 來回成本) x 252）")
    print("=" * 78)
    print("\n  台灣證交稅賣出邊 0.3% = 30 bp/來回；手續費 0.1425%/邊"
          "（折扣後常見 3~6 bp/邊）；")
    print("  借券費另計。**稅率請自行核對現行規定。**")
    labs = list(data)
    print(f"\n  {'來回成本':>9s} " + " ".join(f"{l[:20]:>21s}" for l in labs))
    base = {l: (avg(r, lambda x: x["r_k"].mean()), avg(r, lambda x: x["turn"].mean()))
            for l, r in data.items()}
    for c in COSTS_BP:
        cells = [f"{(g - t * c / 1e4) * 252 * 100:+20.1f}%" for g, t in
                 (base[l] for l in labs)]
        print(f"  {c:7d}bp " + " ".join(cells))


def block_D(data, k):
    print("\n" + "=" * 78)
    print("D. 為什麼 IC 的排名不保證回測的排名")
    print("=" * 78)
    print("\n  (1) IC 的逐日波動——Sharpe 是訊噪比，IC 只是平均")
    print(f"      {'模型':24s} {'mean IC':>9s} {'sd(IC)':>8s} {'ICIR':>8s}")
    for lab, runs in data.items():
        m = avg(runs, lambda r: r["ic"].mean())
        s = avg(runs, lambda r: r["ic"].std(ddof=1))
        print(f"      {lab:24s} {m:+9.4f} {s:8.4f} {avg(runs, lambda r: icir(r['ic'])):8.4f}")

    print("\n  (2) IC 用整個橫截面，Top-K 只用兩端——平均報酬可能翻轉")
    print(f"      {'模型':24s} {'z加權(全部)':>13s} {f'Top-{k}(兩端)':>13s}")
    for lab, runs in data.items():
        print(f"      {lab:24s} {avg(runs, lambda r: r['r_z'].mean()) * 100:12.4f}%"
              f" {avg(runs, lambda r: r['r_k'].mean()) * 100:12.4f}%")

    print("\n  (3) 選到的是哪些股票——同樣的平均報酬，分母可以差很多")
    print(f"      {'模型':24s} {'多頭 sd':>9s} {'全體 sd':>9s} {'比值':>7s} {'命中率':>8s}")
    for lab, runs in data.items():
        sl = avg(runs, lambda r: r["sd_long"].mean())
        su = avg(runs, lambda r: r["sig"].mean())
        print(f"      {lab:24s} {sl * 100:8.3f}% {su * 100:8.3f}% {sl / su:7.3f}"
              f" {avg(runs, lambda r: r['hit'].mean()) * 100:7.1f}%")

    print("\n  (4) corr(IC_t, sigma_t)——「IC 出現在哪一天」的影響")
    print(f"      {'模型':24s} {'corr':>8s} {'E[IC]xE[sig]':>13s} {'E[ICxsig]':>11s}")
    for lab, runs in data.items():
        c = avg(runs, lambda r: np.corrcoef(r["ic"], r["sig"])[0, 1])
        prod = avg(runs, lambda r: r["ic"].mean() * r["sig"].mean())
        act = avg(runs, lambda r: r["r_z"].mean())
        print(f"      {lab:24s} {c:+8.4f} {prod * 100:12.4f}% {act * 100:10.4f}%")
    print("\n  註：本專案 (4) 的貢獻只有報酬的 2~3%，不是主因；(1) 才是。")
    print("      但在別的資料上 (4) 可能很大——若模型只在低波動日準，")
    print("      組合報酬會遠低於平均 IC 的暗示。")


BLOCKS = {"A": block_A, "B": block_B, "C": block_C, "D": block_D}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", nargs="*", default=list(BLOCKS), choices=list(BLOCKS))
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    args = ap.parse_args()
    data = collect(args.topk)
    if not data:
        raise SystemExit("找不到任何預測檔")
    for kk in args.blocks:
        BLOCKS[kk](data, args.topk)
    print()


if __name__ == "__main__":
    main()
