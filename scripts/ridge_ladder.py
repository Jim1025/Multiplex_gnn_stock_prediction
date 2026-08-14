"""
ridge_ladder.py — 線性基準的資訊階梯 R0–R4

存在理由（回應口試對 ridge baseline 的質疑）：
    原本的 ridge 只用「前一日 30 檔美股的報酬」共 30 維，而神經模型看的是
    20 天 x 9 特徵。輸入集不同，並排比較不成立。

    修法不是「加個窗口」了事，而是排成一條階梯，把「線性模型的能力」與
    「餵給它多少資訊」分開：

        R0  台股自己 20x9                      180 維   無跨市場資訊的線性底線
        R1  配對 ADR 前一日報酬                  1 維    恆等邊單獨值多少
        R2  30 檔美股前一日報酬                  30 維   原本的版本
        R3  30 檔美股 x 5 日報酬                150 維   跨市場的時序結構有沒有用
        R4  30 檔美股 x 20 天 x 9 特徵 + R0    5580 維   與神經模型輸入完全相同

    R4 是關鍵那格：它和 MAGNET 吃一模一樣的資料，所以
        R4 勝 MAGNET  -> 非線性與圖結構沒有貢獻，必須誠實面對
        MAGNET 勝 R4  -> 我們的貢獻有了乾淨的歸屬
        R2 約等於 R4  -> 線性在 30 維就飽和，更高維只是雜訊

協定（與神經模型逐項對齊，否則比較無意義）：
    - 特徵直接取自 MultiplexDataset，與神經模型看到的張量是同一份，
      no-lookahead 由 dataset 的建構保證，不另外實作一套
    - 同一組 walk-forward 索引（train 0-1149 / val 1150-1396 / test 1397-）
    - 標準化只用 train 的統計量
    - alpha 只在 val 上選，且選的是「橫截面 val IC」——與神經模型的
      early_stop_metric=IC 同一個判準。不是每檔各自選 MSE 最好的 alpha，
      那會變成 50 個獨立模型各自調參，與神經模型的單一 checkpoint 不對等

    每檔台股一個獨立迴歸（per-stock），與原始設定一致。

用法：
    .venv/bin/python scripts/ridge_ladder.py --config configs/tw50.yaml
    .venv/bin/python scripts/ridge_ladder.py --rungs R0 R2 R4     只跑部分
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
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.multiplex_dataset import MultiplexDataset  # noqa: E402

# log_return 在 TECH_FEATURE_COLS 的位置。改動特徵順序時這裡要跟著改，
# 故從 features.py 取而非寫死索引。
from src.dataset.features import TECH_FEATURE_COLS  # noqa: E402

RET = TECH_FEATURE_COLS.index("log_return")

# 上界要夠高：R0（台股自身特徵）的最佳 alpha 在 1e8 才收斂，用 1e6 封頂會
# 選在邊界上、低報 R0 的表現（+0.0121 vs 收斂後的 +0.0149）。
ALPHAS = np.logspace(-2, 9, 23)

RUNGS = {
    # RC 不是 ridge，是這個任務真正的空模型：不看任何特徵，只照 train 的
    # 平均報酬把 50 檔排一個固定順序，每天照抄。橫截面排序任務必須先贏過
    # 它才有意義——實測 R0 在最佳 alpha（1e9）下係數被壓到 1.6e-09，
    # 跨日變動 5.3e-08，本來就已經退化成這個模型，兩者 IC 逐位相同。
    "RC": "constant: train-mean rank (0)",
    "R0": "TW own 20x9 (180)",
    "R1": "paired ADR t-1 return (1)",
    "R2": "30 US t-1 returns (30)",
    "R3": "30 US x 5d returns (150)",
    "R4": "30 US x 20d x 9F + TW own (5580)",
}


def collect(ds: MultiplexDataset) -> dict:
    """把一個 split 的所有樣本讀成密集陣列。"""
    X1 = np.empty((len(ds), ds.T, ds.n_l1, len(TECH_FEATURE_COLS)), dtype=np.float32)
    X2 = np.empty((len(ds), ds.T, ds.n_l2, len(TECH_FEATURE_COLS)), dtype=np.float32)
    Y = np.empty((len(ds), ds.n_l2), dtype=np.float32)
    dates = []
    for i in range(len(ds)):
        s = ds[i]
        X1[i] = s["x_seq_L1"].numpy()
        X2[i] = s["x_seq_L2"].numpy()
        Y[i] = s["y"].numpy()
        dates.append(s["target_date"])
    return {"X1": X1, "X2": X2, "Y": Y, "dates": np.asarray(dates)}


def design(rung: str, d: dict, j: int, pair_index) -> np.ndarray:
    """第 j 檔台股在某一階梯下的設計矩陣 [n_days, p]。"""
    X1, X2 = d["X1"], d["X2"]
    n = X1.shape[0]
    if rung == "R0":
        return X2[:, :, j, :].reshape(n, -1)
    if rung == "R1":
        i = pair_index[j]
        if i < 0:
            # 無配對台股在這一階梯下沒有任何輸入——預測退化為截距。
            # 這不是實作缺陷，是 R1 這個資訊集的定義使然，也正是
            # 「恆等邊到不了 43 檔」在線性模型上的對應。
            return np.zeros((n, 1), dtype=np.float32)
        return X1[:, -1, i, RET].reshape(n, 1)
    if rung == "R2":
        return X1[:, -1, :, RET]
    if rung == "R3":
        return X1[:, -5:, :, RET].reshape(n, -1)
    if rung == "R4":
        return np.hstack([X1.reshape(n, -1), X2[:, :, j, :].reshape(n, -1)])
    raise ValueError(f"未知 rung={rung!r}")


def daily_ic(Yhat: np.ndarray, Y: np.ndarray, cols=None) -> tuple[float, float]:
    """回傳 (mean Pearson IC, mean Spearman RankIC)；跳過 y 全同的日子。"""
    p, s = [], []
    yh = Yhat if cols is None else Yhat[:, cols]
    y = Y if cols is None else Y[:, cols]
    for t in range(len(y)):
        a, b = yh[t], y[t]
        if len(a) < 3 or np.std(b) == 0 or np.std(a) == 0:
            continue
        p.append(np.corrcoef(a, b)[0, 1])
        s.append(stats.spearmanr(a, b).statistic)
    return float(np.mean(p)), float(np.mean(s))


def fit_rung(rung: str, tr: dict, va: dict, te: dict, pair_index) -> dict:
    """對一個階梯：掃 alpha、以 val 橫截面 IC 選點、回傳 test 預測。"""
    n2 = tr["Y"].shape[1]
    if rung == "RC":
        const = tr["Y"].mean(0)                       # [n2]，只用 train
        Yva = np.tile(const, (len(va["Y"]), 1))
        return {"alpha": float("nan"), "val_IC": daily_ic(Yva, va["Y"])[0],
                "Yte": np.tile(const, (len(te["Y"]), 1)), "n_params": 0}
    # 每檔股票的設計矩陣：R2/R3 全體共用（不含台股自身特徵），可一次擬合
    shared = rung in ("R2", "R3")

    def build(dset):
        if shared:
            X = design(rung, dset, 0, pair_index)
            return [X] * n2
        return [design(rung, dset, j, pair_index) for j in range(n2)]

    Xtr, Xva, Xte = build(tr), build(va), build(te)
    # 標準化：只用 train 的均值與標準差
    stats_ = []
    for j in range(n2 if not shared else 1):
        mu = Xtr[j].mean(0)
        sd = Xtr[j].std(0)
        sd[sd < 1e-8] = 1.0
        stats_.append((mu, sd))
    def z(Xs, j):
        mu, sd = stats_[0 if shared else j]
        return (Xs[j] - mu) / sd

    best = None
    for a in ALPHAS:
        Yva = np.empty_like(va["Y"])
        models = []
        if shared:
            m = Ridge(alpha=a).fit(z(Xtr, 0), tr["Y"])
            Yva[:] = m.predict(z(Xva, 0))
            models = [m]
        else:
            for j in range(n2):
                m = Ridge(alpha=a).fit(z(Xtr, j), tr["Y"][:, j])
                Yva[:, j] = m.predict(z(Xva, j))
                models.append(m)
        ic, _ = daily_ic(Yva, va["Y"])
        if best is None or ic > best[1]:
            best = (a, ic, models)
    alpha, val_ic, models = best

    Yte = np.empty_like(te["Y"])
    if shared:
        Yte[:] = models[0].predict(z(Xte, 0))
    else:
        for j in range(n2):
            Yte[:, j] = models[j].predict(z(Xte, j))

    p = sum(m.coef_.size for m in models) if not shared else models[0].coef_.size
    return {"alpha": alpha, "val_IC": val_ic, "Yte": Yte, "n_params": int(p)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ridge 資訊階梯 R0-R4")
    ap.add_argument("--config", default="configs/tw50.yaml")
    ap.add_argument("--rungs", nargs="*", default=list(RUNGS))
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    kw = dict(snapshot_dir=cfg["data"]["snapshot_dir"],
              features_dir=cfg["data"]["features_dir"],
              T=cfg["model"]["lstm"]["T_history"],
              config_path=args.config)
    print("[ridge] 讀取資料 ...")
    ds_tr = MultiplexDataset(split="train", **kw)
    ds_va = MultiplexDataset(split="val", **kw)
    ds_te = MultiplexDataset(split="test", **kw)
    pair_index = list(ds_tr.pair_index)
    tw_codes = list(ds_tr.tw_codes)
    paired = [j for j, i in enumerate(pair_index) if i >= 0]
    unpaired = [j for j, i in enumerate(pair_index) if i < 0]
    tr, va, te = collect(ds_tr), collect(ds_va), collect(ds_te)
    print(f"[ridge] train {len(tr['dates'])} / val {len(va['dates'])} / test {len(te['dates'])} 天，"
          f"n1={ds_tr.n_l1} n2={ds_tr.n_l2}，配對 {len(paired)} 檔")

    stamp = time.strftime("%Y%m%d_%H%M")
    print(f"\n{'rung':5s} {'輸入':34s} {'參數':>8s} {'alpha':>9s} {'val IC':>8s} "
          f"{'test IC':>9s} {'RankIC':>8s} {'paired7':>9s} {'unpair43':>9s}")
    for rung in args.rungs:
        t0 = time.time()
        r = fit_rung(rung, tr, va, te, pair_index)
        ic, ric = daily_ic(r["Yte"], te["Y"])
        icp, _ = daily_ic(r["Yte"], te["Y"], paired)
        icu, _ = daily_ic(r["Yte"], te["Y"], unpaired)
        print(f"{rung:5s} {RUNGS[rung]:34s} {r['n_params']:8,} {r['alpha']:9.2g} "
              f"{r['val_IC']:+8.4f} {ic:+9.4f} {ric:+8.4f} {icp:+9.4f} {icu:+9.4f}"
              f"   ({time.time()-t0:.0f}s)")

        # 與訓練 run 相同的目錄結構，讓既有分析腳本可以直接吃
        d = ROOT / args.out / f"{stamp}_ridge_{rung}"
        (d / "predictions").mkdir(parents=True, exist_ok=True)
        rows = [(te["dates"][t], tw_codes[j], r["Yte"][t, j], te["Y"][t, j])
                for t in range(len(te["dates"])) for j in range(len(tw_codes))]
        pd.DataFrame(rows, columns=["target_date", "ticker", "y_hat", "y"]).to_csv(
            d / "predictions" / "test_predictions.csv", index=False)
        json.dump({"slug": d.name, "tag": f"ridge_{rung}", "status": "FINISHED",
                   "rung": rung, "input": RUNGS[rung], "n_params": r["n_params"],
                   "alpha": float(r["alpha"]), "best_val_IC": r["val_IC"],
                   "test_metrics": {"IC": ic, "RankIC": ric,
                                    "IC_paired7": icp, "IC_unpaired43": icu}},
                  open(d / "meta.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[ridge] 預測已寫入 {args.out}/{stamp}_ridge_*/predictions/")


if __name__ == "__main__":
    main()
