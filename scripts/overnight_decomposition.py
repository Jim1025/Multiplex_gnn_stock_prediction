"""
overnight_decomposition.py — 跨市場訊號的傳導管道（proposal §58 的 P5）

把次日報酬拆成隔夜跳空與盤中兩段，問：模型賺到的錢來自哪一段？

    r_tot = log(close_t / close_{t-1})
    r_gap = log(open_t  / close_{t-1})     <- 美股盤發生在這一段
    r_int = log(close_t / open_t)
    r_tot = r_gap + r_int                   （恆等）

**不重訓**，只用既有預測 + data/processed 的 OHLC（唯讀）。

兩個關鍵設計，都是踩過坑之後才定的：

1. **必須用 data/processed，不能用 data/raw。** raw 與 pipeline 在極端日
   不一致——2025-04-07/08（關稅衝擊後台股跌停）有 42 筆差到 9.97%。
   本腳本啟動時會強制跑對齊閘門，不過就中止。

2. **歸因用組合報酬，不是比較 IC。** IC 是相關係數，對兩邊的仿射變換
   都不變，所以「先標準化再比 IC」是無效操作。而組合報酬是可加的：
       Σ w·r_tot = Σ w·r_gap + Σ w·r_int
   這是精確歸因，沒有尺度混淆。

用法：
    .venv/bin/python scripts/overnight_decomposition.py
    .venv/bin/python scripts/overnight_decomposition.py --arm tw50_beta --no-probe
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.config import load_universe   # noqa: E402

ARM_DEFAULT = "tw50_betaF1nA2r1"
GATE_TOL = 1e-6


def tw_panel(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        d = (pd.read_csv(ROOT / f"data/processed/tw/{t}.csv", parse_dates=["Date"])
             .sort_values("Date").set_index("Date"))
        rows.append(pd.DataFrame({
            "target_date": d.index, "ticker": t,
            "r_tot": np.log(d.Close / d.Close.shift(1)).values,
            "r_gap": np.log(d.Open / d.Close.shift(1)).values,
            "r_int": np.log(d.Close / d.Open).values,
            "is_imputed": d.is_imputed.values,
        }))
    return pd.concat(rows)


def find_seeds(arm: str) -> list[str]:
    return [d for d in sorted(glob.glob(str(ROOT / "runs" / "**" / f"*{arm}_s*"), recursive=True))
            if re.fullmatch(rf"\d{{8}}_\d{{4}}_{re.escape(arm)}_s\d+", os.path.basename(d))]


def gate(m: pd.DataFrame) -> None:
    """對齊閘門。差一天就全錯，所以不過就中止。"""
    add = float(np.abs(m.r_tot - (m.r_gap + m.r_int)).max())
    rec = float(np.abs(m.y - m.r_tot).max())
    bad = int((np.abs(m.y - m.r_tot) > GATE_TOL).sum())
    print("=== 對齊閘門 ===")
    print(f"  可加恆等式  max|r_tot − (r_gap+r_int)| = {add:.3e}")
    print(f"  重建 r_tot vs 預測檔 y   max|差| = {rec:.3e}   超過 {GATE_TOL:g} 的筆數 {bad}/{len(m)}")
    print(f"  被標記 is_imputed  {int(m.is_imputed.sum())}/{len(m)}")
    if add > 1e-12 or bad > 0:
        raise SystemExit("閘門未通過——先確認資料來源是 data/processed 而非 data/raw")
    print("  通過\n")


def hac_p(d: np.ndarray) -> float:
    n = len(d)
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    x = d - d.mean()
    var = float((x @ x) / n)
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float((x[k:] @ x[:-k]) / n)
    return float(2 * stats.t.sf(abs(d.mean() / np.sqrt(max(var, 1e-24) / n)), n - 1))


def legs(m: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳每日的 (總報酬, 隔夜腿, 盤中腿)。w = 標準化分數 / n，金額中性。"""
    T, G, I = [], [], []
    for _, g in m.groupby("target_date"):
        yh = g.y_hat.values
        if yh.std() < 1e-12:
            continue
        z = (yh - yh.mean()) / yh.std()
        n = len(g)
        T.append(z @ g.r_tot.values / n)
        G.append(z @ g.r_gap.values / n)
        I.append(z @ g.r_int.values / n)
    return np.array(T), np.array(G), np.array(I)


def channel_probe(u, va, te) -> None:
    """各通道的線性探針上限：30 檔美股 t−1 報酬 -> 該通道。協定同 §53。"""
    X = pd.concat([(pd.read_csv(ROOT / f"data/processed/adr/{t}.csv", parse_dates=["Date"])
                    .sort_values("Date").set_index("Date")["log_return"].rename(t))
                   for t in u.us_nodes], axis=1)

    def target(col: str) -> pd.DataFrame:
        out = {}
        for t in u.tw_nodes:
            d = (pd.read_csv(ROOT / f"data/processed/tw/{t}.csv", parse_dates=["Date"])
                 .sort_values("Date").set_index("Date"))
            out[t] = {"r_tot": np.log(d.Close / d.Close.shift(1)),
                      "r_gap": np.log(d.Open / d.Close.shift(1)),
                      "r_int": np.log(d.Close / d.Open)}[col]
        return pd.DataFrame(out)

    print("=== 各通道的線性探針上限（30 檔美股 t−1 報酬 -> 該通道）===")
    for col, lab in (("r_gap", "隔夜跳空"), ("r_tot", "總報酬"), ("r_int", "盤中")):
        Y = target(col)
        idx = X.index.intersection(Y.index)
        Xl, Ya = X.loc[idx].shift(1), Y.loc[idx]
        tr = idx[idx < va.min()]
        mu, sd = Xl.loc[tr].mean(), Xl.loc[tr].std().replace(0, 1)
        Z = ((Xl - mu) / sd).fillna(0.0)
        A = Z.loc[tr].values
        best = None
        for al in (1, 10, 100, 1000, 10000):
            P = pd.DataFrame(index=idx, columns=Ya.columns, dtype=float)
            for c in Ya.columns:
                b = Ya[c].loc[tr].values
                ok = np.isfinite(b) & np.isfinite(A).all(1)
                if ok.sum() < 50:
                    continue
                w = np.linalg.solve(A[ok].T @ A[ok] + al * np.eye(A.shape[1]), A[ok].T @ b[ok])
                P[c] = Z.values @ w

            def ic(dates):
                v = []
                for dt in dates:
                    if dt not in P.index:
                        continue
                    a = P.loc[dt].values.astype(float)
                    b = Ya.loc[dt].values.astype(float)
                    ok2 = np.isfinite(a) & np.isfinite(b)
                    if ok2.sum() < 10 or a[ok2].std() < 1e-12 or b[ok2].std() < 1e-12:
                        continue
                    v.append(np.corrcoef(a[ok2], b[ok2])[0, 1])
                return float(np.mean(v)) if v else np.nan

            v = ic(va)
            if best is None or v > best[0]:
                best = (v, al, ic(te))
        print(f"  {lab:10} alpha={best[1]:<6} val IC {best[0]:+.4f}   test IC {best[2]:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="次日報酬的隔夜／盤中歸因")
    ap.add_argument("--arm", default=ARM_DEFAULT)
    ap.add_argument("--universe", default="tw50")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()

    u = load_universe(a.universe)
    runs = find_seeds(a.arm)
    if not runs:
        raise SystemExit(f"找不到 {a.arm} 的 run")

    preds = []
    for d in runs:
        p = pd.read_csv(os.path.join(d, "predictions", "test_predictions.csv"))
        p["target_date"] = pd.to_datetime(p.target_date)
        preds.append(p)
    P = tw_panel(sorted(preds[0].ticker.unique()))

    ens = (pd.concat(preds).groupby(["target_date", "ticker"])
           .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
    m = ens.merge(P, on=["target_date", "ticker"], how="left").dropna(subset=["r_tot"])
    gate(m)

    T, G, I = legs(m)
    print(f"=== 組合報酬的可加分解（種子集成，{len(T)} 天）===")
    print(f"  恆等式  max|T − (G+I)| = {np.max(np.abs(T - (G + I))):.2e}\n")
    print(f"  {'':16}{'日均報酬':>11}{'佔總額':>9}{'日 sd':>10}{'逐日 HAC p':>12}")
    for lab, p in (("總報酬", T), ("  隔夜跳空腿", G), ("  盤中腿", I)):
        print(f"  {lab:16}{p.mean()*100:>+10.4f}%{p.mean()/T.mean()*100:>8.1f}%"
              f"{p.std(ddof=1)*100:>9.4f}%{hac_p(p):>12.4f}")
    d = G - I
    print(f"\n  隔夜 − 盤中 = {d.mean()*100:+.4f}%/日   逐日 HAC p = {hac_p(d):.4f}   "
          f"隔夜較高 {int((G > I).sum())}/{len(G)} 天")

    per = []
    for p in preds:
        mm = p.merge(P, on=["target_date", "ticker"], how="left").dropna(subset=["r_tot"])
        _, g, i = legs(mm)
        per.append((g.mean(), i.mean()))
    arr = np.array(per)
    dd = arr[:, 0] - arr[:, 1]
    print(f"\n=== 逐種子（{len(arr)} 顆）===")
    print(f"  隔夜腿 {arr[:,0].mean()*100:+.4f}% (sd {arr[:,0].std(ddof=1)*100:.4f}%)   "
          f"盤中腿 {arr[:,1].mean()*100:+.4f}% (sd {arr[:,1].std(ddof=1)*100:.4f}%)")
    print(f"  差 {dd.mean()*100:+.4f}%   同號 {int((dd>0).sum())}/{len(dd)}   "
          f"配對 t p = {stats.ttest_rel(arr[:,0], arr[:,1])[1]:.2e}")
    print(f"  隔夜腿佔總報酬 {arr[:,0].mean()/(arr[:,0]+arr[:,1]).mean()*100:.1f}%")

    vg = np.mean([g.r_gap.var() for _, g in m.groupby("target_date")])
    vi = np.mean([g.r_int.var() for _, g in m.groupby("target_date")])
    cv = np.mean([np.cov(g.r_gap, g.r_int)[0, 1] for _, g in m.groupby("target_date") if len(g) > 2])
    tot = vg + vi + 2 * cv
    print(f"\n=== 橫截面變異分解 ===")
    print(f"  隔夜 {vg/tot*100:.1f}%   盤中 {vi/tot*100:.1f}%   2xCov {2*cv/tot*100:.1f}%")
    print(f"  （隔夜只承擔 {vg/tot*100:.1f}% 的變異，卻貢獻 {arr[:,0].mean()/(arr[:,0]+arr[:,1]).mean()*100:.1f}% 的報酬）\n")

    if not a.no_probe:
        d0 = runs[0]
        va = pd.to_datetime(pd.read_csv(os.path.join(d0, "predictions", "val_predictions.csv"))
                            .target_date.unique())
        te = pd.to_datetime(m.target_date.unique())
        channel_probe(u, va, te)


if __name__ == "__main__":
    main()
