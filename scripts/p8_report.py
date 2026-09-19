"""p8_report.py — P8（`mse_target: zscore`）的結果表（proposal §60.2a(i)）

對主結果 baseline 做跨種子配對比較，並對 KTW+ 做單樣本檢定
（KTW+ 是確定性擬合，沒有種子可配對）。

兩格的用意見 §60.2a(i)：`zscore` 同時改了目標與梯度平衡（後者約 50 倍），
所以要跑兩個權重設定才分得開。

用法：
    .venv/bin/python scripts/p8_report.py            # 兩折都跑（缺的那折會標示）
    .venv/bin/python scripts/p8_report.py --fold 1
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import icir_gap as ig            # noqa: E402
import results_table as rt       # noqa: E402

# 第二折的評估窗上界，與 results_table.py:_fold2_rows 的 CUT 同值。
CUT = "2024-12-25"

# 權重縮放用的 train 逐日橫截面 sd(y)。只用於在表頭印出來，不影響計算。
SD_Y_TRAIN = {1: 0.012459, 2: 0.012795}

FOLDS = {
    1: dict(base="tw50_betaF1nA2r1", bal="tw50_p8zbal", raw="tw50_p8z",
            ktw="runs/linear/*fvg_KTWp/predictions/*.csv", cut=False),
    2: dict(base="f2_best", bal="f2_p8zbal", raw="f2_p8z",
            ktw="runs_f2/*fvg_KTWp/predictions/*.csv", cut=True),
}


def per_seed(arm: str, cut: bool) -> dict:
    """arm -> {seed: 指標 dict}。逐 seed 由預測檔重算，不讀 meta.json。"""
    out: dict[str, dict] = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / "**" / f"*{arm}_s*"), recursive=True)):
        s = d.rsplit("_s", 1)[-1]
        if not s.isdigit():
            continue
        f = glob.glob(f"{d}/predictions/test_predictions.csv")
        if not f:
            continue
        df = pd.read_csv(f[0])
        df["target_date"] = df["target_date"].astype(str).str[:10]
        if cut:
            df = df[df.target_date <= CUT]
        H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index().to_numpy()
        Y = df.pivot(index="target_date", columns="ticker", values="y").sort_index().to_numpy()
        ic = ig.ic_series(H, Y)
        ric = np.array([stats.spearmanr(H[t], Y[t]).statistic
                        if H[t].std() > 0 and Y[t].std() > 0 else np.nan
                        for t in range(len(Y))])
        sh, sy = H.std(1), Y.std(1)
        ok = sy > 1e-15
        out[s] = dict(ic=float(np.nanmean(ic)), ric=float(np.nanmean(ric)),
                      icir=ig.icir(ic), ricir=ig.icir(ric),
                      disp=float((sh[ok] / sy[ok]).mean()), sd_hat=float(sh.mean()),
                      cal=rt.calib_stats(H, Y), daily_ic=ic,
                      rk=np.array([rt.topk_ret(H[t], Y[t]) for t in range(len(Y))]))
    return out


def hac_p(d: np.ndarray) -> tuple[float, float, int]:
    n = len(d)
    L = int(4 * (n / 100) ** (2 / 9))
    s2 = np.var(d, ddof=1)
    for l in range(1, L + 1):
        s2 += 2 * (1 - l / (L + 1)) * np.cov(d[l:], d[:-l], ddof=1)[0, 1]
    tt = d.mean() / (s2 / n) ** 0.5
    return float(d.mean()), float(2 * (1 - stats.norm.cdf(abs(tt)))), L


def report(fold: int) -> None:
    c = FOLDS[fold]
    print("\n" + "=" * 78)
    print(f"第{'一二'[fold-1]}折   （權重縮放用的 train sd(y) = {SD_Y_TRAIN[fold]}）")
    print("=" * 78)
    A = [("主結果 baseline", c["base"]), ("P8b 平衡保持", c["bal"]), ("P8a 權重不動", c["raw"])]
    R = {nm: per_seed(a, c["cut"]) for nm, a in A}
    missing = [nm for nm, _ in A if not R[nm]]
    if missing:
        print(f"  缺：{'、'.join(missing)}")
        if R["主結果 baseline"] is None or not R["主結果 baseline"]:
            return

    print(f"\n  {'arm':16s} {'n':>3s} {'IC':>16s} {'RankIC':>16s} {'ICIR':>16s} {'RankICIR':>16s}")
    for nm, _ in A:
        d = R[nm]
        if not d:
            continue
        g = lambda k: (float(np.mean([v[k] for v in d.values()])),
                       float(np.std([v[k] for v in d.values()], ddof=1)) if len(d) > 1 else np.nan)
        print(f"  {nm:16s} {len(d):3d} " +
              " ".join(f"{m:+8.4f} ±{s:6.4f}" for m, s in (g('ic'), g('ric'), g('icir'), g('ricir'))))

    print(f"\n  {'arm':16s} {'離散比 vs y':>11s} {'ŷ 日 sd':>9s} {'ŷ/訓練目標':>11s}"
          f" {'MSE(1e-3)':>10s} {'校正後 R²':>10s} {'Sharpe':>8s} {'MDD(複)':>8s}")
    for nm, _ in A:
        d = R[nm]
        if not d:
            continue
        tgt = SD_Y_TRAIN[fold] if nm.startswith("主") else 1.0
        sd_hat = float(np.mean([v['sd_hat'] for v in d.values()]))
        pf = rt.portfolio_stats([v['rk'] for v in d.values()])
        print(f"  {nm:16s} {np.mean([v['disp'] for v in d.values()]):11.2f} {sd_hat:9.4f}"
              f" {sd_hat/tgt:11.2f} {np.mean([v['cal']['mse'] for v in d.values()])*1e3:10.3f}"
              f" {np.mean([v['cal']['r2cal'] for v in d.values()]):+10.4f}"
              f" {pf['sharpe']:8.2f} {pf['mdd_c']*100:7.2f}%")

    base = R["主結果 baseline"]
    for nm, _ in A[1:]:
        d = R[nm]
        if not d or not base:
            continue
        common = sorted(set(d) & set(base), key=int)
        if len(common) < 3:
            print(f"\n  [{nm}] 共同種子只有 {len(common)} 顆，跳過檢定")
            continue
        print(f"\n  [{nm}] vs 主結果 baseline —— 跨種子配對，共同種子 n={len(common)}")
        print(f"    {'指標':>10s} {'差':>9s} {'配對 t p':>10s} {'同號':>8s}")
        for k, lab in (('ic', 'IC'), ('ric', 'RankIC'), ('icir', 'ICIR'),
                       ('ricir', 'RankICIR'), ('disp', '離散比')):
            a = np.array([d[s][k] for s in common])
            b = np.array([base[s][k] for s in common])
            p = stats.ttest_rel(a, b).pvalue
            same = int(np.sum(np.sign(a - b) == np.sign(np.mean(a - b))))
            print(f"    {lab:>10s} {np.mean(a-b):+9.4f} {p:10.4f} {same:4d}/{len(common)}")
        da = np.nanmean([d[s]['daily_ic'] for s in common], 0)
        db = np.nanmean([base[s]['daily_ic'] for s in common], 0)
        ok = np.isfinite(da) & np.isfinite(db)
        m, p, L = hac_p(da[ok] - db[ok])
        print(f"    逐日 IC HAC：Δ {m:+.4f}  p {p:.4f}  (lag={L})")

    # KTW+ 是確定性擬合（固定設計矩陣上的 Ridge.fit），沒有種子可配對，故用單樣本 t
    K = ig.load(c["ktw"], c["cut"])
    if not K:
        print(f"\n  （找不到 KTW+：{c['ktw']}）")
        return
    H, Y = K[0]
    kic = ig.ic_series(H, Y)
    kric = np.array([stats.spearmanr(H[t], Y[t]).statistic
                     if H[t].std() > 0 and Y[t].std() > 0 else np.nan for t in range(len(Y))])
    KT = {'ic': float(np.nanmean(kic)), 'ric': float(np.nanmean(kric)),
          'icir': ig.icir(kic), 'ricir': ig.icir(kric)}
    print(f"\n  KTW+（確定性 n=1）  IC {KT['ic']:+.4f}  RankIC {KT['ric']:+.4f}"
          f"  ICIR {KT['icir']:.4f}  RankICIR {KT['ricir']:.4f}")
    print(f"\n  {'arm':16s} {'指標':>9s} {'我方':>9s} {'差':>9s} {'單樣本 t p':>11s} {'勝出種子':>9s}")
    for nm, _ in A:
        d = R[nm]
        if not d:
            continue
        for k, lab in (('icir', 'ICIR'), ('ricir', 'RankICIR'), ('ic', 'IC')):
            v = np.array([x[k] for x in d.values()])
            p = stats.ttest_1samp(v, KT[k]).pvalue
            print(f"  {nm:16s} {lab:>9s} {v.mean():+9.4f} {v.mean()-KT[k]:+9.4f} {p:11.4f}"
                  f" {int((v > KT[k]).sum()):6d}/{len(v)}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="P8 的結果表（proposal §60.2a(i)）")
    ap.add_argument("--fold", type=int, nargs="*", default=[1, 2], choices=[1, 2])
    for f in ap.parse_args().fold:
        report(f)


if __name__ == "__main__":
    main()
