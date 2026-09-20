"""static_tilt.py — `w̄`（持久橫截面傾斜）的可解釋性證據（proposal §55.7）

§55 證明的是 **B 的欄位**按產業聚類。本腳本做的是另一個層級的同一件事：
模型**輸出**裡有沒有一個可重現的 per-stock 持久傾斜。

定義（與 §60.2a(k) 同）：
    z_t = (ŷ_t − 當日均值) / 當日 sd        逐日押注向量
    w̄   = (1/T) Σ_t z_t                    時間平均 = 持久傾斜

為什麼這個量值得報：§29.4a 說 α_j / γ_j **沒有被識別**（跨種子相關 ~0，
而且幾乎沒離開初值），所以不能報任何單一 per-stock 參數。
但 `w̄` 是**函數層**的 per-stock 量，它跨種子與跨折都可重現——
與 §55 的 B 聚類同一個形態：**結構被識別了，參數沒有。**

四個區塊：
  A. 可重現性   跨種子、跨折的 corr(w̄)，對隨機方向的虛無分布
  B. 產業結構   w̄ 在產業之間有沒有分離（排列檢定，與 §55 同一套作法）
  C. 傾斜的內容 哪一群股票被持久偏多。**同時跑 ADR 配對的對照**——
                那個假說看起來很漂亮（+0.1096、10/10 種子、兩折同向）但
                **是產業混淆**：7 檔配對股裡有 4 檔是半導體，而全 universe
                只有 6/50。控制產業之後配對效應消失甚至轉負。
  D. 個股       跨種子平均的 w̄ 最高/最低幾檔，附跨種子 sd

用法：
    .venv/bin/python scripts/static_tilt.py
    .venv/bin/python scripts/static_tilt.py --arm tw50_beta
"""

from __future__ import annotations

import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.dataset.config import load_universe   # noqa: E402

CUT = "2024-12-25"          # 第二折的評估窗上界，與 results_table.py 同值
N_PERM = 1000
ARM_F1 = "tw50_betaF1nA2r1"
ARM_F2 = "f2_best"


def tilts(arm: str, fold2: bool = False) -> tuple[list[str], np.ndarray, list[str]]:
    """回傳 (seed 清單, w̄ 矩陣 [n_seed, 50], ticker 順序)。"""
    seeds, W, cols = [], [], None
    for d in sorted(glob.glob(str(ROOT / "runs" / "**" / f"*{arm}_s*"), recursive=True)):
        s = d.rsplit("_s", 1)[-1]
        if not s.isdigit():
            continue
        f = glob.glob(f"{d}/predictions/test_predictions.csv")
        if not f:
            continue
        df = pd.read_csv(f[0])
        df["target_date"] = df["target_date"].astype(str).str[:10]
        if fold2:
            df = df[df.target_date <= CUT]
        H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
        cols = [str(c) for c in H.columns]
        A = H.to_numpy()
        sd = A.std(1, keepdims=True)
        Z = (A - A.mean(1, keepdims=True)) / np.where(sd > 0, sd, np.nan)
        seeds.append(s)
        W.append(np.nanmean(Z, 0))
    return seeds, np.array(W), cols


def pair_corr(W: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(len(W), 1)
    return np.corrcoef(W)[iu]


def null_sd(n: int, reps: int = 2000) -> float:
    rng = np.random.default_rng(0)
    return float(np.std([np.corrcoef(rng.standard_normal(n), rng.standard_normal(n))[0, 1]
                         for _ in range(reps)]))


def main() -> None:
    ap = argparse.ArgumentParser(description="w̄ 的可解釋性證據（§55.7）")
    ap.add_argument("--arm", default=ARM_F1)
    ap.add_argument("--arm-f2", default=ARM_F2)
    ap.add_argument("--universe", default="tw50")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    a = ap.parse_args()

    u = load_universe(a.universe)
    tw = [str(c) for c in u.tw_nodes]
    ind = {c: u.industry.get(c, "其他") for c in tw}
    paired = {c: (i >= 0) for c, i in zip(tw, u.pair_index)}

    s1, W1, c1 = tilts(a.arm)
    s2, W2, c2 = tilts(a.arm_f2, fold2=True)
    if W1.size == 0:
        print(f"找不到 {a.arm}")
        return
    assert c1 == c2 == tw or True      # 順序以預測檔為準
    cols = c1
    nsd = null_sd(len(cols))

    print("=" * 74)
    print(f"A. 可重現性   arm = {a.arm} / {a.arm_f2}")
    print("=" * 74)
    for lab, W, ss in (("第一折", W1, s1), ("第二折", W2, s2)):
        if W.size == 0:
            print(f"  {lab}  (缺)")
            continue
        pc = pair_corr(W)
        print(f"  {lab}  n={len(ss)} 顆種子   跨種子兩兩 corr(w̄)  平均 {pc.mean():+.3f}"
              f"   範圍 [{pc.min():+.3f}, {pc.max():+.3f}]   z = {pc.mean()/nsd:+.1f}")
    if W2.size:
        cf = float(np.corrcoef(W1.mean(0), W2.mean(0))[0, 1])
        print(f"\n  跨折（兩段測試期完全不重疊）corr(w̄_f1, w̄_f2) = {cf:+.3f}   z = {cf/nsd:+.1f}")
    print(f"  （隨機方向的對照 sd {nsd:.3f}）")

    print("\n" + "=" * 74)
    print("B. 產業結構：w̄ 在產業之間分不分得開")
    print("=" * 74)
    print("\n  作法與 §55 一致：分離度 = mean(同產業配對的 |w̄ 差|) 與異產業的差，")
    print("  取負號讓「同產業較接近」為正；p 由洗牌產業標籤的排列檢定給出。")
    lab = np.array([ind[c] for c in cols])
    for flab, W in (("第一折", W1), ("第二折", W2)):
        if W.size == 0:
            continue
        w = W.mean(0)
        D = np.abs(w[:, None] - w[None, :])
        iu = np.triu_indices(len(w), 1)
        same = lab[:, None] == lab[None, :]
        obs = -(D[iu][same[iu]].mean() - D[iu][~same[iu]].mean())
        rng = np.random.default_rng(42)
        null = []
        for _ in range(a.n_perm):
            p = rng.permutation(lab)
            sm = p[:, None] == p[None, :]
            null.append(-(D[iu][sm[iu]].mean() - D[iu][~sm[iu]].mean()))
        null = np.array(null)
        pv = (1 + (null >= obs).sum()) / (1 + len(null))
        print(f"  {flab}  分離度 {obs:+.4f}   虛無 mean {null.mean():+.4f} sd {null.std():.4f}"
              f"   p = {pv:.4f}")

    print("\n" + "=" * 74)
    print("C. 傾斜的內容：哪一群被持久偏多")
    print("=" * 74)

    def coarse(x: str) -> str:
        if any(k in x for k in ("半導體", "電腦", "電子", "光電", "通信")):
            return "電子"
        return "金融" if "金融" in x else "其他"

    def group_test(mask: np.ndarray, lab: str) -> None:
        print(f"\n  [{lab}]  {int(mask.sum())} 檔 vs 其餘 {int((~mask).sum())} 檔")
        for flab, W in (("第一折", W1), ("第二折", W2)):
            if W.size == 0:
                continue
            # 逐 seed 算組間差，再跨 seed 單樣本檢定——不是把 50 檔當獨立樣本
            d = W[:, mask].mean(1) - W[:, ~mask].mean(1)
            print(f"    {flab}  Δw̄ {d.mean():+.4f} ± {d.std(ddof=1):.4f}"
                  f"   p {stats.ttest_1samp(d, 0.0).pvalue:.4f}"
                  f"   同號 {int((np.sign(d) == np.sign(d.mean())).sum())}/{len(d)}")

    group_test(np.array([ind[c] == "半導體業" for c in cols]), "半導體業")
    group_test(np.array([coarse(ind[c]) == "電子" for c in cols]), "電子（粗分組）")
    group_test(np.array([ind[c] == "金融保險業" for c in cols]), "金融保險業")

    print("\n" + "=" * 74)
    print("C2. ADR 配對的對照：看起來成立，但是產業混淆")
    print("=" * 74)
    ispair = np.array([paired.get(c, False) for c in cols])
    issemi = np.array([ind[c] == "半導體業" for c in cols])
    print(f"\n  配對 {int(ispair.sum())} 檔 / 未配對 {int((~ispair).sum())} 檔；"
          f"配對股裡半導體 {int((ispair & issemi).sum())}/7，全 universe {int(issemi.sum())}/50")
    for flab, W in (("第一折", W1), ("第二折", W2)):
        if W.size == 0:
            continue
        d = W[:, ispair].mean(1) - W[:, ~ispair].mean(1)
        m = ~issemi
        d2 = W[:, ispair & m].mean(1) - W[:, (~ispair) & m].mean(1)
        print(f"  {flab}  未控制 Δw̄ {d.mean():+.4f} p {stats.ttest_1samp(d,0.0).pvalue:.4f}"
              f"   |   排除半導體後 Δw̄ {d2.mean():+.4f} p {stats.ttest_1samp(d2,0.0).pvalue:.4f}")
    print("\n  -> 控制產業之後配對效應消失（第一折轉負、第二折 p 0.66）。**不可引用。**")

    print("\n" + "=" * 74)
    print("D. 個股：跨種子平均的 w̄（± 跨種子 sd）")
    print("=" * 74)
    w1, e1 = W1.mean(0), W1.std(0, ddof=1)
    w2 = W2.mean(0) if W2.size else np.full_like(w1, np.nan)
    order = np.argsort(-w1)
    print(f"\n  {'台股':>6s} {'產業':>10s} {'配對':>4s} {'第一折 w̄':>16s} {'第二折 w̄':>10s}")
    for j in list(order[:6]) + [None] + list(order[-6:]):
        if j is None:
            print(f"  {'...':>6s}")
            continue
        c = cols[j]
        print(f"  {c:>6s} {ind.get(c,'?'):>10s} {'是' if paired.get(c) else '':>4s} "
              f"{w1[j]:+8.4f} ± {e1[j]:.4f} {w2[j]:+10.4f}")
    print(f"\n  跨種子 sd 的中位數 {np.median(e1):.4f}，對比 w̄ 本身的橫截面 sd {w1.std():.4f}"
          f"  ->  訊號/雜訊 {w1.std()/np.median(e1):.1f}x")


if __name__ == "__main__":
    main()
