"""
factor_vs_graph.py — [24] 的跨市場訊號是「圖」還是「一個因子」？

ridge_ladder 已經定位到全部訊號都在 R2（30 檔美股前一日報酬，30 維，
test IC +0.1020），而 [24] 的完整管線（t 檢定篩邊 + 十個模型 + 集成）
與 R2 統計上無法區分（p = 0.53~0.59）。

那就要問下一層問題：那 30 維裡真正在做事的是什麼？

  M1    只有美股等權平均（1 維）
        -> 每檔台股學一個 beta_j，橫截面排序由 beta_j 與當日市場方向決定。
           如果 M1 就逼近 R2，那 [24] 的「二部圖」實際上是單因子模型，
           per-pair 的 t 檢定篩邊在描述一件本來就只有一維的事。
  M30r  30 檔減去等權平均後的殘差（30 維，秩 29）
        -> 個股層級的特異訊號。M1 + M30r 的增量才是「圖」真正的貢獻。

第二層問題是 [24] 的估計方式：它對 50 個目標各配一個獨立迴歸，
每個用 w=250 天配最多 30 個係數，總共 1,500 個自由參數，目標間不共享
任何東西。如果瓶頸是樣本量而非模型類別（該文十個模型全距只有 0.013，
本身就是「模型類別不是瓶頸」的證據），那把影響矩陣約束成低秩應該會贏：

  PCRr  先在訓練期把 30 維壓到 r 個主成分（列空間共享），再 per-target
        配 r 個係數 -> 參數量 50r，r=1,2,3,5,10 對照滿秩的 1,500。

第三、四層是兩個 [24] 結構上沒有的東西：

  RANK  把 y 先做逐日橫截面 rank 再擬合。[24] 全用點預測損失，評估卻是
        IC/RankIC，目標與評估不一致。
  TW+   X = 30 檔美股 + 50 檔台股前一日報酬（80 維）。[24] 只從美股預測
        台股，目標側之間沒有任何連結。注意 R0（台股自身 20x9）已證實
        無效，但那是「自身歷史」，與「同日其他台股」是不同的資訊集。

全部沿用 ridge_ladder 的協定：alpha 以 val 橫截面 IC 選點、標準化只用
train 統計量、walk-forward 不重切。

用法：
    .venv/bin/python scripts/factor_vs_graph.py
    .venv/bin/python scripts/factor_vs_graph.py --arms M1 R2 PCR3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.multiplex_dataset import MultiplexDataset  # noqa: E402
from src.dataset.features import TECH_FEATURE_COLS  # noqa: E402

RET = TECH_FEATURE_COLS.index("log_return")
ALPHAS = np.logspace(-2, 9, 23)

# arm 名稱是可組合的：[K] + 設計矩陣 + [/r]
#   前綴 K   = 擬合目標改成逐日橫截面 rank
#   /r 後綴  = 先把設計矩陣壓到 r 個主成分（列空間由 50 檔目標共享）
ARMS = {
    "R2":      "30 US t-1 returns (reference)",
    "M1":      "US equal-weight mean only (1)",
    "M30r":    "30 US minus mean, residual (30)",
    "M1+r":    "mean + residuals, explicit (31)",
    "R2/1":    "shared rank-1 subspace",
    "R2/2":    "shared rank-2 subspace",
    "R2/3":    "shared rank-3 subspace",
    "R2/5":    "shared rank-5 subspace",
    "R2/10":   "shared rank-10 subspace",
    "KR2":     "rank target, 30 US",
    "TW+":     "30 US + 50 TW t-1 returns (80)",
    "KTW+":    "rank target, US+TW (80)",
    "TW+/10":  "US+TW, shared rank-10",
    "KTW+/10": "rank target, US+TW, rank-10",
    "KTW+/20": "rank target, US+TW, rank-20",
    "KR2/10":  "rank target, 30 US, rank-10",
}


def parse_arm(arm: str) -> tuple[bool, str, int | None]:
    """arm -> (是否 rank 目標, 設計矩陣名稱, 主成分數或 None)。"""
    use_rank = arm.startswith("K")
    body = arm[1:] if use_rank else arm
    if "/" in body:
        body, r = body.split("/")
        return use_rank, body, int(r)
    return use_rank, body, None


def collect(ds: MultiplexDataset) -> dict:
    X1 = np.empty((len(ds), ds.T, ds.n_l1, len(TECH_FEATURE_COLS)), dtype=np.float32)
    X2 = np.empty((len(ds), ds.T, ds.n_l2, len(TECH_FEATURE_COLS)), dtype=np.float32)
    Y = np.empty((len(ds), ds.n_l2), dtype=np.float32)
    for i in range(len(ds)):
        s = ds[i]
        X1[i], X2[i], Y[i] = s["x_seq_L1"].numpy(), s["x_seq_L2"].numpy(), s["y"].numpy()
    return {"X1": X1, "X2": X2, "Y": Y}


def design(arm: str, d: dict) -> np.ndarray:
    """所有 arm 的設計矩陣都是全體台股共用的 [n_days, p]。"""
    us = d["X1"][:, -1, :, RET]                        # [n, 30]
    if arm == "R2":
        return us
    if arm == "M1":
        return us.mean(1, keepdims=True)                # [n, 1]
    if arm == "M30r":
        return us - us.mean(1, keepdims=True)           # [n, 30]，秩 29
    if arm == "M1+r":
        return np.hstack([us.mean(1, keepdims=True), us - us.mean(1, keepdims=True)])
    if arm == "TW+":
        return np.hstack([us, d["X2"][:, -1, :, RET]])  # [n, 80]
    raise ValueError(f"未知 arm={arm!r}")


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


def daily_ic_series(Yhat, Y):
    """逐日 (IC, RankIC) 序列，供配對檢定用。"""
    p, s = [], []
    for t in range(len(Y)):
        a, b = Yhat[t], Y[t]
        if np.std(a) == 0 or np.std(b) == 0:
            p.append(np.nan)
            s.append(np.nan)
        else:
            p.append(np.corrcoef(a, b)[0, 1])
            s.append(stats.spearmanr(a, b).statistic)
    return np.asarray(p), np.asarray(s)


def rank_transform(Y: np.ndarray) -> np.ndarray:
    """逐日橫截面 rank，再標準化到零均值單位變異，讓 ridge 看到一致尺度。"""
    R = np.apply_along_axis(stats.rankdata, 1, Y).astype(np.float64)
    R -= R.mean(1, keepdims=True)
    sd = R.std(1, keepdims=True)
    return R / np.where(sd < 1e-12, 1.0, sd)


def fit_arm(arm: str, tr: dict, va: dict, te: dict) -> dict:
    use_rank, body, r = parse_arm(arm)
    Xtr, Xva, Xte = design(body, tr), design(body, va), design(body, te)

    # 標準化只用 train 統計量
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    if r is not None:
        # 列空間共享：主成分只在 train 上配，r 個方向由 50 檔目標共用，
        # per-target 只剩 r 個係數。這是「共享參數」最乾淨的線性版本。
        pca = PCA(n_components=r, random_state=42).fit(Xtr)
        Xtr, Xva, Xte = pca.transform(Xtr), pca.transform(Xva), pca.transform(Xte)

    # 擬合目標：K 前綴用逐日橫截面 rank，其餘用原始報酬
    Ttr = rank_transform(tr["Y"]) if use_rank else tr["Y"]

    best = None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, Ttr)
        ic, _ = daily_ic(m.predict(Xva), va["Y"])
        if best is None or ic > best[1]:
            best = (a, ic, m)
    alpha, val_ic, m = best
    return {"alpha": alpha, "val_IC": val_ic, "Yte": m.predict(Xte),
            "n_params": int(m.coef_.size)}


def main() -> None:
    ap = argparse.ArgumentParser(description="[24] 的訊號是圖還是因子")
    ap.add_argument("--config", default="configs/tw50.yaml")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    kw = dict(snapshot_dir=cfg["data"]["snapshot_dir"],
              features_dir=cfg["data"]["features_dir"],
              T=cfg["model"]["lstm"]["T_history"], config_path=args.config)
    ds_tr = MultiplexDataset(split="train", **kw)
    ds_va = MultiplexDataset(split="val", **kw)
    ds_te = MultiplexDataset(split="test", **kw)
    pair_index = list(ds_tr.pair_index)
    paired = [j for j, i in enumerate(pair_index) if i >= 0]
    unpaired = [j for j, i in enumerate(pair_index) if i < 0]
    tr, va, te = collect(ds_tr), collect(ds_va), collect(ds_te)
    print(f"[fvg] train {len(tr['Y'])} / val {len(va['Y'])} / test {len(te['Y'])} 天，"
          f"n1={ds_tr.n_l1} n2={ds_tr.n_l2}，配對 {len(paired)} 檔")

    # 美股橫截面本身的因子結構：第一主成分解釋了多少
    us_tr = tr["X1"][:, -1, :, RET]
    ev = PCA(n_components=min(10, us_tr.shape[1]), random_state=42).fit(
        (us_tr - us_tr.mean(0)) / np.where(us_tr.std(0) < 1e-8, 1.0, us_tr.std(0))
    ).explained_variance_ratio_
    print(f"[fvg] 美股 30 維（train）主成分解釋比例："
          f"{' '.join(f'{v:.3f}' for v in ev[:5])} ... 前 3 累計 {ev[:3].sum():.3f}")

    print(f"\n{'arm':8s} {'輸入':34s} {'參數':>6s} {'alpha':>9s} {'val IC':>8s} "
          f"{'test IC':>9s} {'RankIC':>8s} {'paired7':>9s} {'unpair43':>9s}")
    series = {}
    for arm in args.arms:
        t0 = time.time()
        r = fit_arm(arm, tr, va, te)
        ic, ric = daily_ic(r["Yte"], te["Y"])
        pic, _ = daily_ic(r["Yte"], te["Y"], paired)
        uic, _ = daily_ic(r["Yte"], te["Y"], unpaired)
        series[arm] = daily_ic_series(r["Yte"], te["Y"])
        print(f"{arm:8s} {ARMS[arm]:34s} {r['n_params']:6,} {r['alpha']:9.2g} "
              f"{r['val_IC']:+8.4f} {ic:+9.4f} {ric:+8.4f} {pic:+9.4f} {uic:+9.4f}"
              f"   [{time.time()-t0:.1f}s]")

    # 對照 R2 的逐日配對檢定（IC 與 RankIC 各一）
    if "R2" in series:
        print(f"\n{'arm':8s} {'dIC':>8s} {'t':>7s} {'p':>8s}    "
              f"{'dRankIC':>8s} {'t':>7s} {'p':>8s}   （vs R2 逐日配對）")
        for arm in args.arms:
            if arm == "R2":
                continue
            row = f"{arm:8s}"
            for k in (0, 1):
                base, s = series["R2"][k], series[arm][k]
                ok = ~(np.isnan(base) | np.isnan(s))
                t, p = stats.ttest_rel(s[ok], base[ok])
                row += (f" {(s[ok]-base[ok]).mean():+8.4f} {t:+7.2f} {p:8.4f}"
                        + ("   " if k == 0 else ""))
            print(row)


if __name__ == "__main__":
    main()
