"""
bipartite_screen.py — [24] Bipartite screening baseline 的重現

Liu, Grith, Dong & Cucuringu (2026), "A Bipartite Graph Approach to U.S.-China
Cross-Market Return Forecasting", arXiv:2603.10559

這是 proposal p9 related work 表上與本研究最接近的一篇：同樣是 entity level、
跨市場、利用非重疊交易時段、來源市場 -> 目標市場的有向耦合。差別在於它用
逐對 t 檢定篩邊，我們用「已知固定 + 估計可學」的兩級邊。不重現它，
「兩級邊優於逐對檢定」這個主張沒有對照。

原文方法（§4.1、§5.2）：
    1. 回看視窗 w = 250 個交易日
    2. 對每個有序配對 (X_j, Y_i) 在視窗內做單變量 OLS：y 對 x
           t_beta = beta * sqrt(Sxx) / se ,  se = sqrt(SSE / (w-2))
    3. |t_beta| > tau 才連邊，原文 tau = 2
       原文明說這只是稀疏化手段，「not a formal multiple-testing correction」
    4. 被選中的來源股票即為該目標股票的預測特徵
    5. 每 10 天重建圖並重訓所有模型（rolling）
    6. 十個模型：OLS / LASSO / RIDGE / SVM / XGBoost / LGBM / RF / AdaBoost
       / ensemble-avg / ensemble-med
    7. 報酬 winsorize 至 0.5 / 99.5 百分位
    8. 落後 l = 1（來源市場收盤在目標市場開盤之前）

本重現與原文的差異，比較時必須揭露：
    - 目標：原文是 open-to-close，本研究是 next-day close-to-close
      （由凍結的資料層決定，不改）
    - 指標：原文報 Sharpe / PnL，這裡報 IC / RankIC 以便與其餘 baseline 並列
    - 模型：xgboost 與 lightgbm 未安裝，故十個取六個
      （OLS / LASSO / RIDGE / SVM / RF / AdaBoost）+ 兩個 ensemble
    - 特徵：原文用來源市場的日報酬，這裡取快照裡的 log_return，
      與神經模型看到的是同一份張量
    - universe：原文自承用全樣本市值選股、有 look-ahead；本研究在第一個
      訓練日選定，這一點我們比原文乾淨，不是缺陷

兩種評估方案（--scheme）：
    rolling      忠實重現：250 天滾動視窗、每 10 天重訓
                 -> 回答「[24] 的方法在這個資料上多強」
    walkforward  與其餘 baseline 同協定：只在 train split 上擬合一次
                 -> 回答「同協定下兩級邊是否優於逐對檢定」
    兩者都要報。rolling 讓模型看到離預測日很近的資料，對它有利；
    只報 rolling 會低估我們，只報 walkforward 會低估原文。

用法：
    .venv/bin/python scripts/bipartite_screen.py --scheme rolling
    .venv/bin/python scripts/bipartite_screen.py --scheme walkforward --tau 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.features import TECH_FEATURE_COLS  # noqa: E402
from src.dataset.multiplex_dataset import MultiplexDataset  # noqa: E402

RET = TECH_FEATURE_COLS.index("log_return")

W_DEFAULT = 250      # 原文 §5.2
REFIT_EVERY = 10     # 原文 §5.2
TAU_DEFAULT = 2.0    # 原文 §4.1
WINSOR = 0.5         # 原文 §3：0.5 / 99.5 百分位


def models() -> dict:
    """原文十個模型中本機可用的六個 + 兩個 ensemble（在 predict 端合成）。"""
    return {
        "OLS":      LinearRegression(),
        "LASSO":    Lasso(alpha=1e-4, max_iter=5000),
        "RIDGE":    Ridge(alpha=1.0),
        "SVM":      SVR(kernel="rbf", C=1.0, epsilon=1e-3),
        "RF":       RandomForestRegressor(n_estimators=100, max_depth=5,
                                          random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=42),
    }


def tstats(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    視窗內所有有序配對的迴歸 t 統計量，回傳 [n_x, n_y]。

    向量化版本，等價於對每個 (i, j) 各做一次單變量 OLS：
        beta = Sxy / Sxx
        SSE  = Syy - beta^2 * Sxx
        se   = sqrt(SSE / (w - 2))
        t    = beta * sqrt(Sxx) / se
    """
    w = X.shape[0]
    xc = X - X.mean(0)
    yc = Y - Y.mean(0)
    Sxx = (xc ** 2).sum(0)                     # [n_x]
    Syy = (yc ** 2).sum(0)                     # [n_y]
    Sxy = xc.T @ yc                            # [n_x, n_y]
    Sxx_safe = np.where(Sxx < 1e-12, np.nan, Sxx)
    beta = Sxy / Sxx_safe[:, None]
    SSE = np.clip(Syy[None, :] - beta ** 2 * Sxx_safe[:, None], 1e-18, None)
    se = np.sqrt(SSE / max(w - 2, 1))
    t = beta * np.sqrt(Sxx_safe)[:, None] / se
    return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def winsorize_by(a: np.ndarray, ref: np.ndarray, q: float) -> np.ndarray:
    """
    用 ref（僅訓練期）的百分位截尾 a（全期）。

    百分位若用全樣本算就會把測試期的極端值資訊洩漏進來——雖然影響小，
    但它是可避免的前視，而本研究的其餘管線都嚴格避免。
    """
    lo, hi = np.percentile(ref, [q, 100 - q], axis=0)
    return np.clip(a, lo, hi)


def fit_predict(Xtr, Ytr, Xte, tau, mdl_names, standardize=True):
    """
    對每個目標股票：t 檢定篩邊 -> 在被選中的來源股票上擬合 -> 預測。

    篩不到任何邊時退回截距（視窗均值）——這是 tau 的直接後果，
    不是實作缺陷，原文的稀疏化本來就允許空鄰域。
    """
    n_y = Ytr.shape[1]
    T = tstats(Xtr, Ytr)
    out = {m: np.empty((Xte.shape[0], n_y)) for m in mdl_names}
    n_edges = 0
    for j in range(n_y):
        sel = np.flatnonzero(np.abs(T[:, j]) > tau)
        n_edges += len(sel)
        if len(sel) == 0:
            for m in mdl_names:
                out[m][:, j] = Ytr[:, j].mean()
            continue
        xtr, xte, ytr = Xtr[:, sel], Xte[:, sel], Ytr[:, j]
        if standardize:
            # 報酬的量級是 1e-2。SVR 的 epsilon/C 與 Ridge 的 alpha 都是在
            # 「特徵尺度為 O(1)」的假設下取預設值的，不標準化會讓核方法
            # 完全失效（實測 SVM 從 +0.093 掉到 +0.046）。統計量只用訓練窗。
            mu, sd = xtr.mean(0), xtr.std(0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        for m, est in models().items():
            if m not in mdl_names:
                continue
            est.fit(xtr, ytr)
            out[m][:, j] = est.predict(xte)
    return out, n_edges


def daily_ic(Yhat, Y, cols=None):
    yh = Yhat if cols is None else Yhat[:, cols]
    y = Y if cols is None else Y[:, cols]
    p, s = [], []
    for t in range(len(y)):
        a, b = yh[t], y[t]
        if len(a) < 3 or np.std(b) == 0 or np.std(a) == 0:
            continue
        p.append(np.corrcoef(a, b)[0, 1])
        v = stats.spearmanr(a, b).statistic
        if not np.isnan(v):
            s.append(v)
    return float(np.mean(p)), float(np.mean(s))


def main() -> None:
    ap = argparse.ArgumentParser(description="[24] Bipartite screening 重現")
    ap.add_argument("--config", default="configs/tw50.yaml")
    ap.add_argument("--scheme", choices=["rolling", "walkforward"], default="rolling")
    ap.add_argument("--tau", type=float, default=TAU_DEFAULT)
    ap.add_argument("--window", type=int, default=W_DEFAULT)
    ap.add_argument("--models", nargs="*", default=["OLS", "LASSO", "RIDGE", "RF"])
    ap.add_argument("--no-standardize", action="store_true",
                    help="關閉特徵標準化（原文未載明；預設開啟，見 fit_predict 註解）")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    kw = dict(snapshot_dir=cfg["data"]["snapshot_dir"],
              features_dir=cfg["data"]["features_dir"],
              T=cfg["model"]["lstm"]["T_history"],
              config_path=args.config)
    ds_all = MultiplexDataset(split="all", **kw)
    n_te = len(MultiplexDataset(split="test", **kw))
    n_tr = len(MultiplexDataset(split="train", **kw))
    n_va = len(MultiplexDataset(split="val", **kw))
    pair_index = list(ds_all.pair_index)
    tw_codes = list(ds_all.tw_codes)
    paired = [j for j, i in enumerate(pair_index) if i >= 0]
    unpaired = [j for j, i in enumerate(pair_index) if i < 0]

    N = len(ds_all)
    X = np.empty((N, ds_all.n_l1), dtype=np.float64)   # 來源市場：t-1 的日報酬
    Y = np.empty((N, ds_all.n_l2), dtype=np.float64)   # 目標市場：t 的報酬
    dates = []
    for i in range(N):
        s = ds_all[i]
        X[i] = s["x_seq_L1"][-1, :, RET].numpy()
        Y[i] = s["y"].numpy()
        dates.append(s["target_date"])
    dates = np.asarray(dates)
    te0 = n_tr + n_va                                  # 測試期第一個索引
    X = winsorize_by(X, X[:n_tr], WINSOR)              # 百分位只用訓練期
    Y = winsorize_by(Y, Y[:n_tr], WINSOR)
    print(f"[24] N={N} 天，train {n_tr} / val {n_va} / test {n_te}，"
          f"來源 {ds_all.n_l1} 檔 -> 目標 {ds_all.n_l2} 檔")
    print(f"     scheme={args.scheme}  tau={args.tau}  window={args.window}  "
          f"refit_every={REFIT_EVERY}  models={args.models}")

    P = {m: np.empty((n_te, ds_all.n_l2)) for m in args.models}
    edges = []
    t0 = time.time()
    if args.scheme == "rolling":
        for blk in range(0, n_te, REFIT_EVERY):
            t = te0 + blk
            lo = max(0, t - args.window)
            out, ne = fit_predict(X[lo:t], Y[lo:t], X[t:t + REFIT_EVERY],
                                  args.tau, args.models, not args.no_standardize)
            edges.append(ne)
            for m in args.models:
                P[m][blk:blk + out[m].shape[0]] = out[m]
    else:
        out, ne = fit_predict(X[:n_tr], Y[:n_tr], X[te0:te0 + n_te],
                              args.tau, args.models, not args.no_standardize)
        edges.append(ne)
        for m in args.models:
            P[m][:] = out[m]

    tot_pairs = ds_all.n_l1 * ds_all.n_l2
    print(f"     選中的邊：平均 {np.mean(edges):.0f} / {tot_pairs} "
          f"（{np.mean(edges)/tot_pairs:.1%}），{len(edges)} 次重建，{time.time()-t0:.0f}s")

    Yte = Y[te0:te0 + n_te]
    dte = dates[te0:te0 + n_te]
    stamp = time.strftime("%Y%m%d_%H%M")
    print(f"\n{'model':10s} {'test IC':>9s} {'RankIC':>9s} {'paired7':>9s} {'unpair43':>9s}")
    preds = {}
    for m in args.models:
        preds[m] = P[m]
    if len(args.models) > 1:
        preds["ens-avg"] = np.mean([P[m] for m in args.models], axis=0)
        preds["ens-med"] = np.median([P[m] for m in args.models], axis=0)
    for m, Yh in preds.items():
        ic, ric = daily_ic(Yh, Yte)
        icp, _ = daily_ic(Yh, Yte, paired)
        icu, _ = daily_ic(Yh, Yte, unpaired)
        print(f"{m:10s} {ic:+9.4f} {ric:+9.4f} {icp:+9.4f} {icu:+9.4f}")
        # tau 進目錄名：掃 tau 時 stamp 只到分鐘，不帶 tau 會互相覆蓋
        d = ROOT / args.out / f"{stamp}_bipartite_{args.scheme}_t{args.tau:g}_{m}"
        (d / "predictions").mkdir(parents=True, exist_ok=True)
        rows = [(dte[t], tw_codes[j], Yh[t, j], Yte[t, j])
                for t in range(n_te) for j in range(len(tw_codes))]
        pd.DataFrame(rows, columns=["target_date", "ticker", "y_hat", "y"]).to_csv(
            d / "predictions" / "test_predictions.csv", index=False)
        json.dump({"slug": d.name, "tag": f"bipartite_{args.scheme}_{m}",
                   "status": "FINISHED", "scheme": args.scheme, "model": m,
                   "tau": args.tau, "window": args.window,
                   "mean_edges": float(np.mean(edges)),
                   "test_metrics": {"IC": ic, "RankIC": ric,
                                    "IC_paired7": icp, "IC_unpaired43": icu}},
                  open(d / "meta.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[24] 預測已寫入 {args.out}/{stamp}_bipartite_{args.scheme}_*/")


if __name__ == "__main__":
    main()
