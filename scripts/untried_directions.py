"""
untried_directions.py — proposal §64 夜間批次（2026-10-11）的檢驗程式

§63 之後依建議順序執行「還沒嘗試的方向」。每個區塊都在結果出來之前寫好事前登記（§64.13），
本檔把當時 scratchpad 裡的探針整併成一支，數字與 §64 逐位元相同。

區塊：

  N     非線性強基線（同 80 個輸入）：GBM（主）、MLP（次）；事後：延伸網格、靜態傾斜拆解、事後組合
  R     全域資料與市場狀態：VIX、匯率、|SPY|、美股離散度的交互作用與水準特徵；三分位描述
  A     不對稱傳導：正負報酬分開
  S     供應鏈先驗圖：先驗懲罰 ridge + 100 組同度數隨機先驗；事後：只用先驗邊、MAGNET 的 B 對齊
  S2    只跑 S 的事後 2（MAGNET 的 B 對齊）；配 --arm / --arm-f2 換主 arm
  E     early fusion 的三方組合（第二折重現）與 D3 公平性；事後：預測的持續性
  L     IC 損失（`loss_weights.ic`）的判定
  P2    第二折的單市場消融
  P3    beta_init_std 0.1 / 1.0 的敏感度
  POOL  兩折 test 窗串接後的描述性檢定（非登記）

只讀 MultiplexDataset、`runs/` 的預測檔、`data/processed/sentiment/` 與 checkpoint，不寫入 repo；
GBM / MLP 的預測快取在 --cache（預設在系統暫存目錄）。

用法：
    .venv/bin/python scripts/untried_directions.py --blocks N R A S
    .venv/bin/python scripts/untried_directions.py --blocks E L P2 P3 POOL     # run 完成之後
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import pickle
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
from scipy import stats       # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import tw_lag_channel as T                       # noqa: E402
from src.dataset.config import load_universe     # noqa: E402

U = load_universe("tw50")
US, TW = list(U.us_nodes), list(U.tw_nodes)
ETF = {"SPY", "QQQ", "SMH", "XLF", "XLB", "XLE"}
SPY = US.index("SPY")
SEEDS = [42, 7, 123, 11, 99, 314, 555, 777, 888, 2026]
FOLDS = ("第一折", "第二折")
MAIN = {"第一折": "tw50_betaF1nA2r1", "第二折": "f2_best"}
CACHE = Path(tempfile.gettempdir()) / "untried_directions_cache"
_FOLD: dict = {}


# ---------------------------------------------------------------- 資料與共用
def fold(f: str) -> dict:
    if f not in _FOLD:
        p = CACHE / f"{f}.pkl"
        if p.exists():
            _FOLD[f] = pickle.load(open(p, "rb"))
        else:
            c = T.FOLDS[f]
            _FOLD[f] = T.load_fold(c["cfg"], c["f2"])
            pickle.dump(_FOLD[f], open(p, "wb"))
    return _FOLD[f]


def X80(D, sp):
    return np.hstack([D[f"us_{sp}"], D[f"tw_{sp}"]])


def zrow(Y):
    R = Y - Y.mean(1, keepdims=True)
    s = R.std(1, keepdims=True)
    return R / np.where(s < 1e-12, 1.0, s)


def ser(P, Y):
    return T.ic_ser(P, Y)


def summ(P, Y):
    return T.summ(P, Y)


def hac(a, b):
    d = np.asarray(a) - np.asarray(b)
    return T.hac_p(d[np.isfinite(d)])


def ens(dct):
    return np.nanmean(np.stack([dct[s] for s in sorted(dct)]), 0)


def seedmean_ser(dct, Y):
    return np.nanmean(np.stack([ser(dct[s], Y) for s in sorted(dct)]), 0)


def line(lab, P, Y, ref=None):
    s = summ(P, Y)
    out = f"  {lab:34s} IC {s['IC']:+.4f}  ICIR {s['ICIR']:.4f}  RankIC {s['RIC']:+.4f}"
    if ref is not None:
        d = s["ser"] - ref
        out += f"  dIC {np.nanmean(d):+.4f} (HAC p {T.hac_p(d[np.isfinite(d)]):.4f})"
    return out


def load(pattern, split, D, reeval=False):
    """神經 arm 的逐種子預測 {seed: [T, 50]}；reeval=True 時 test 讀 CPU 重評的 *_reeval.csv。"""
    out = {}
    fn = f"{split}_predictions_reeval.csv" if reeval and split == "test" else f"{split}_predictions.csv"
    for p in sorted(glob.glob(str(ROOT / "runs" / "**" / f"*{pattern}_s*" / "predictions" / fn), recursive=True)):
        tail = Path(p).parents[1].name.rsplit("_s", 1)[-1]
        if not tail.isdigit():
            continue
        df = pd.read_csv(p, dtype={"ticker": str})
        df["target_date"] = df["target_date"].astype(str).str[:10]
        H = df.pivot(index="target_date", columns="ticker", values="y_hat")
        out[int(tail)] = H.reindex(index=D[f"d_{split}"], columns=D["codes"]).to_numpy(float)
    return out


def ktw(f, D):
    return T.preds(T.FOLDS[f]["ktw"], "", D["d_test"], D["codes"])[0][0]


def linz(D):
    """線性-z：per-target ridge，美股 30 + 台股 50，z 目標，無截距（= ceiling/probe_3way 的 L）。"""
    return T.ridge(D, "US30+TW50", "zscore", False)


def ridge_design(Xs, Y):
    """共用設計的 per-target ridge：z 目標、無截距、alpha 以 val 平均 IC 選（同 linz 的協定）。"""
    Ttr = zrow(Y["train"])
    mu, sd = Xs["train"].mean(0), Xs["train"].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = {k: (v - mu) / sd for k, v in Xs.items()}
    Tm = Ttr.mean(0)
    s, V = np.linalg.eigh(Z["train"].T @ Z["train"])
    VtXtT = V.T @ (Z["train"].T @ (Ttr - Tm))
    best = None
    for a in T.ALPHAS:
        C = V @ (VtXtT / (s[:, None] + a))
        v = float(np.nanmean(ser(Z["val"] @ C, Y["val"])))
        if best is None or v > best[0]:
            best = (v, a, C)
    v, a, C = best
    return dict(alpha=a, val_ic=v, val=Z["val"] @ C, test=Z["test"] @ C)


def seed_stats(dct, Y):
    r = [summ(dct[s], Y) for s in sorted(dct)]
    return np.array([x["IC"] for x in r]), np.array([x["ICIR"] for x in r]), np.array([x["RIC"] for x in r])


# ---------------------------------------------------------------- N：非線性強基線
PAIR = [US.index(k) for k in U.pairing]
PAIR_TW = [TW.index(v) for v in U.pairing.values()]
GBM_ITS = (50, 100, 200, 300, 500, 750, 1000)
GBM_GRID = list(itertools.product((7, 15, 31), (100, 400, 1600), (0.0, 10.0), (0.5, 1.0)))
GBM_ITS_EXT = (300, 500, 1000, 1500, 2000, 3000)
GBM_GRID_EXT = list(itertools.product((7, 31, 63), (1600, 3200, 6400), (10.0, 50.0), (0.5,)))


def gbm_rows(D, sp):
    """[T, 80] -> [(T*50), 83]：x_t、代號、自身昨日報酬、配對 ADR 昨日報酬（無配對 NaN）。"""
    us, tw = D[f"us_{sp}"], D[f"tw_{sp}"]
    T_, n = tw.shape
    x = np.repeat(np.hstack([us, tw]), n, axis=0)
    jid = np.tile(np.arange(n), T_)[:, None].astype(float)
    adr = np.full((T_, n), np.nan)
    for a, j in zip(PAIR, PAIR_TW):
        adr[:, j] = us[:, a]
    return np.hstack([x, jid, tw.reshape(-1, 1), adr.reshape(-1, 1)])


def run_gbm(grid, its, name):
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = {}
    for f in FOLDS:
        D = fold(f)
        Xtr, Xva, Xte = gbm_rows(D, "train"), gbm_rows(D, "val"), gbm_rows(D, "test")
        ytr = zrow(D["y_train"]).reshape(-1)
        Tv, Tt = len(D["y_val"]), len(D["y_test"])
        cat = np.zeros(Xtr.shape[1], bool)
        cat[80] = True
        best, t0 = None, time.time()
        for leaf, msl, l2, mf in grid:
            m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=max(its), max_leaf_nodes=leaf,
                                              min_samples_leaf=msl, l2_regularization=l2, max_features=mf,
                                              categorical_features=cat, early_stopping=False, random_state=42)
            m.fit(Xtr, ytr)
            for k, pv in enumerate(m.staged_predict(Xva), start=1):
                if k in its:
                    v = float(np.nanmean(ser(pv.reshape(Tv, 50), D["y_val"])))
                    if best is None or v > best[-1]:
                        best = (leaf, msl, l2, mf, k, v)
        leaf, msl, l2, mf, k, v = best
        print(f"  {f} val 選出：leaf {leaf} msl {msl} l2 {l2} max_features {mf} iters {k}  val IC {v:+.4f}"
              f"（{time.time() - t0:.0f}s）", flush=True)
        Pv, Pt = [], []
        for s in (SEEDS if mf < 1.0 else [42]):
            m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=k, max_leaf_nodes=leaf, min_samples_leaf=msl,
                                              l2_regularization=l2, max_features=mf, categorical_features=cat,
                                              early_stopping=False, random_state=s)
            m.fit(Xtr, ytr)
            Pv.append(m.predict(Xva).reshape(Tv, 50))
            Pt.append(m.predict(Xte).reshape(Tt, 50))
        out[f] = dict(val=np.mean(Pv, 0), test=np.mean(Pt, 0), best=best)
    pickle.dump(out, open(CACHE / f"{name}.pkl", "wb"))
    return out


def run_mlp():
    import torch
    import torch.nn as nn
    torch.set_num_threads(2)
    grid = list(itertools.product((32, 128), (0.1, 0.3), (1e-3, 1e-1)))

    def train_one(Xtr, Ttr, Xva, Yva, h, p, wd, seed, max_ep=200, pat=20, bs=32):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], h), nn.GELU(), nn.Dropout(p), nn.Linear(h, Ttr.shape[1]))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
        xt, tt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ttr, dtype=torch.float32)
        xv = torch.tensor(Xva, dtype=torch.float32)
        best, state, wait = -np.inf, None, 0
        for _ in range(max_ep):
            net.train()
            for i in range(0, len(xt), bs):                     # shuffle=False：依時間順序
                opt.zero_grad()
                ((net(xt[i:i + bs]) - tt[i:i + bs]) ** 2).mean().backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(np.nanmean(ser(net(xv).numpy(), Yva)))
            if v > best:
                best, wait, state = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
            else:
                wait += 1
                if wait >= pat:
                    break
        net.load_state_dict(state)
        net.eval()
        return net, best

    out = {}
    for f in FOLDS:
        D = fold(f)
        Xtr, Xva, Xte = X80(D, "train"), X80(D, "val"), X80(D, "test")
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        Ttr = zrow(D["y_train"])
        res = []
        for h, p, wd in grid:
            vs = [train_one(Xtr, Ttr, Xva, D["y_val"], h, p, wd, s)[1] for s in (42, 7, 123)]
            res.append((np.mean(vs), h, p, wd))
        _, h, p, wd = max(res)
        print(f"  {f} MLP val 選出：h {h} dropout {p} wd {wd}", flush=True)
        Pv, Pt = {}, {}
        for s in SEEDS:
            net, _ = train_one(Xtr, Ttr, Xva, D["y_val"], h, p, wd, s)
            with torch.no_grad():
                Pv[s] = net(torch.tensor(Xva, dtype=torch.float32)).numpy()
                Pt[s] = net(torch.tensor(Xte, dtype=torch.float32)).numpy()
        out[f] = dict(val=Pv, test=Pt, cfg=(h, p, wd))
    pickle.dump(out, open(CACHE / "N_mlp.pkl", "wb"))
    return out


def cached(name, fn):
    p = CACHE / f"{name}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else fn()


def block_n(ext: bool):
    print("\n==================== N：非線性強基線（同 80 個輸入）")
    gbm = cached("N_gbm", lambda: run_gbm(GBM_GRID, GBM_ITS, "N_gbm"))
    mlp = cached("N_mlp", run_mlp)
    gext = cached("N_gbm_ext", lambda: run_gbm(GBM_GRID_EXT, GBM_ITS_EXT, "N_gbm_ext")) if ext else None
    for f in FOLDS:
        D = fold(f)
        Y = D["y_test"]
        L, K = linz(D), ktw(f, D)
        M = load(MAIN[f], "test", D)
        rL, rK, rM = ser(L["test"], Y), ser(K, Y), seedmean_ser(M, Y)
        G = gbm[f]["test"]
        print(f"\n######## {f}  GBM 選出 {gbm[f]['best']}")
        print(line("線性-z", L["test"], Y))
        print(line("KTW+", K, Y))
        print(line("主 arm（集成）", ens(M), Y))
        print(f"  主 arm 種子平均 IC {np.nanmean(rM):+.4f}")
        print(line("[主] N-GBM 對 線性-z", G, Y, rL))
        print(line("N-GBM 對 KTW+", G, Y, rK))
        print(line("N-GBM 對 主 arm（種子平均序列）", G, Y, rM))
        print(f"  逐日預測相關 GBM vs 線性-z {np.nanmean([np.corrcoef(a, b)[0, 1] for a, b in zip(G, L['test'])]):+.3f}"
              f"  GBM vs 主 arm 集成 {np.nanmean([np.corrcoef(a, b)[0, 1] for a, b in zip(G, ens(M))]):+.3f}")
        Pt = mlp[f]["test"]
        per = [summ(Pt[s], Y) for s in SEEDS]
        rP = seedmean_ser(Pt, Y)
        print(f"  N-MLP {mlp[f]['cfg']} 種子平均 IC {np.mean([r['IC'] for r in per]):+.4f}"
              f"（sd {np.std([r['IC'] for r in per], ddof=1):.4f}）ICIR {np.mean([r['ICIR'] for r in per]):.4f}；"
              f"種子平均序列對線性-z {np.nanmean(rP - rL):+.4f}（HAC p {hac(rP, rL):.4f}）")
        print(line("N-MLP 集成 對 線性-z", ens(Pt), Y, rL))
        if gext is not None:
            print(line(f"[事後] GBM 延伸網格 {gext[f]['best'][:5]}", gext[f]["test"], Y, rL))
        # 事後：靜態傾斜拆解
        L1 = T.ridge(D, "US30+TW50", "zscore", True)
        for lab, P in (("線性-z", L["test"]), ("線性-z + 每檔截距", L1["test"]), ("N-GBM", G), ("主 arm 集成", ens(M))):
            Zp = zrow(P)
            st = np.tile(Zp.mean(0, keepdims=True), (len(Zp), 1))
            share = float(np.mean(np.var(st, 1)) / np.mean(np.var(Zp, 1)))
            s, sd_ = summ(P, Y), summ(Zp - st, Y)
            print(f"  [事後] {lab:16s} IC {s['IC']:+.4f}｜靜態部分佔變異 {100 * share:4.1f}%｜動態部分 IC {sd_['IC']:+.4f}")
        # 事後組合（權重在 val 上選）
        combo_block(f, D, M, L, gbm[f], mlp[f])


def combo_block(f, D, M, L, g, m):
    gr = np.round(np.arange(0, 2.01, 0.1), 2)
    Y, Yv = D["y_test"], D["y_val"]
    Mv = load(MAIN[f], "val", D)

    def zr(P):
        return zrow(np.nan_to_num(P))

    def pick(pv):
        best = None
        for ws in itertools.product(gr, repeat=len(pv) - 1):
            v = summ(zr(pv[0]) + sum(w * zr(p) for w, p in zip(ws, pv[1:])), Yv)["IC"]
            if best is None or v > best[0]:
                best = (v, ws)
        return best[1]

    Me, Mve = ens(M), ens(Mv)
    rM, ref2 = ser(Me, Y), None
    for lab, pt, pv in (("主 arm + 線性-z", [Me, L["test"]], [Mve, L["val"]]),
                        ("主 arm + GBM", [Me, g["test"]], [Mve, g["val"]]),
                        ("GBM + 線性-z", [g["test"], L["test"]], [g["val"], L["val"]]),
                        ("主 arm + MLP", [Me, ens(m["test"])], [Mve, ens(m["val"])]),
                        ("主 arm + 線性-z + GBM", [Me, L["test"], g["test"]], [Mve, L["val"], g["val"]])):
        ws = pick(pv)
        P = zr(pt[0]) + sum(w * zr(p) for w, p in zip(ws, pt[1:]))
        s = summ(P, Y)
        extra = ""
        if ref2 is None:
            ref2 = s["ser"]
        else:
            extra = f"  對「主 arm + 線性-z」{np.nanmean(s['ser'] - ref2):+.4f}（HAC p {hac(s['ser'], ref2):.4f}）"
        print(line(f"[事後組合] {lab} w={tuple(float(w) for w in ws)}", P, Y, rM) + extra)


# ---------------------------------------------------------------- R：全域資料與市場狀態
STOCKS = [i for i, c in enumerate(US) if c not in ETF]


def _series(name):
    df = pd.read_csv(ROOT / f"data/processed/sentiment/{name}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    df.index = df.index.strftime("%Y-%m-%d")
    return df["Close"].astype(float)


def _last_before(s, dates, extra=0):
    idx, v = np.array(s.index), s.to_numpy()
    lv, ch = [], []
    for t in dates:
        i = np.searchsorted(idx, t) - 1 - extra
        lv.append(v[i])
        ch.append(np.log(v[i] / v[i - 1]))
    return np.log(np.array(lv)), np.array(ch)


def regime(D, sp):
    d, us = D[f"d_{sp}"], D[f"us_{sp}"]
    vix, fx = _series("VIX"), _series("USDTWD")
    vl, vc = _last_before(vix, d)
    _, fc = _last_before(fx, d)
    _, fc2 = _last_before(fx, d, extra=1)
    return dict(VIXlvl=vl, VIXchg=vc, SPYabs=np.abs(us[:, SPY]), USdisp=us[:, STOCKS].std(1), FXchg=fc, FXchg2=fc2)


def nw_slope(y, x):
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    X = np.column_stack([np.ones_like(x), (x - x.mean()) / x.std()])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e, n = y - X @ b, len(y)
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for k in range(1, lag + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / n
        S += (1 - k / (lag + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    se = np.sqrt(np.diag(Q @ S @ Q) / n)
    return b[1], float(2 * stats.t.sf(abs(b[1] / se[1]), n - 2))


def block_r():
    print("\n==================== R：全域資料與市場狀態")
    out = {}
    for f in FOLDS:
        D = fold(f)
        Y = {sp: D[f"y_{sp}"] for sp in ("train", "val", "test")}
        G = {sp: regime(D, sp) for sp in Y}
        X = {sp: X80(D, sp) for sp in Y}
        base = ridge_design(X, Y)
        rB = ser(base["test"], Y["test"])
        rM = ser(ens(load(MAIN[f], "test", D)), Y["test"])
        out[f] = {}
        print(f"\n######## {f}")
        print(line(f"線性-z（alpha {base['alpha']:.3g}）", base["test"], Y["test"]))

        def stdz(k):
            m, s = G["train"][k].mean(), G["train"][k].std()
            return {sp: (G[sp][k] - m) / s for sp in G}

        for k in ("VIXlvl", "SPYabs", "USdisp"):
            g = stdz(k)
            r = ridge_design({sp: np.hstack([X[sp], X[sp] * g[sp][:, None], g[sp][:, None]]) for sp in X}, Y)
            out[f][k] = ser(r["test"], Y["test"])
            print(line(f"[{'主' if k == 'VIXlvl' else '次'}] 交互作用 g={k}（val {r['val_ic']:+.4f}）", r["test"], Y["test"], rB))
        for keys in (("VIXlvl", "VIXchg", "FXchg"), ("VIXlvl", "VIXchg", "FXchg2"), ("VIXlvl", "VIXchg")):
            r = ridge_design({sp: np.hstack([X[sp]] + [G[sp][k][:, None] for k in keys]) for sp in X}, Y)
            print(line(f"[次] +{'+'.join(keys)}（val {r['val_ic']:+.4f}）", r["test"], Y["test"], rB))
        print("  描述性：test 日依 train 三分位分組的 IC（主 arm 集成 / 線性-z / 差），與逐日 IC 對變數的斜率")
        for k in ("VIXlvl", "SPYabs", "USdisp", "VIXchg", "FXchg"):
            q1, q2 = np.quantile(G["train"][k], [1 / 3, 2 / 3])
            gt = G["test"][k]
            cells = []
            for lo, hi in ((-np.inf, q1), (q1, q2), (q2, np.inf)):
                m = (gt > lo) & (gt <= hi)
                cells.append(f"{m.sum():3d}d {np.nanmean(rM[m]):+.3f}/{np.nanmean(rB[m]):+.3f}/{np.nanmean(rM[m] - rB[m]):+.3f}")
            (bm, pm), (bb, pb), (bd, pd_) = nw_slope(rM, gt), nw_slope(rB, gt), nw_slope(rM - rB, gt)
            print(f"  {k:7s} 低 {cells[0]} | 中 {cells[1]} | 高 {cells[2]}")
            print(f"          斜率（每 1 sd）主 arm {bm:+.4f} p {pm:.3f}；線性-z {bb:+.4f} p {pb:.3f}；差 {bd:+.4f} p {pd_:.3f}")
    pickle.dump(out, open(CACHE / "R.pkl", "wb"))


# ---------------------------------------------------------------- A：不對稱傳導
def block_a():
    print("\n==================== A：不對稱傳導")
    for f in FOLDS:
        D = fold(f)
        Y = {sp: D[f"y_{sp}"] for sp in ("train", "val", "test")}
        X = {sp: X80(D, sp) for sp in Y}
        base = ridge_design(X, Y)
        rB = ser(base["test"], Y["test"])
        print(f"\n######## {f}")
        print(line("線性-z", base["test"], Y["test"]))
        for lab, XX in (("[主] x+ / x- 分開", {sp: np.hstack([np.maximum(X[sp], 0), np.minimum(X[sp], 0)]) for sp in X}),
                        ("[次1] 只拆美股", {sp: np.hstack([np.maximum(X[sp][:, :30], 0), np.minimum(X[sp][:, :30], 0),
                                                         X[sp][:, 30:]]) for sp in X}),
                        ("[次2] x 與 x*1(SPY<0)", {sp: np.hstack([X[sp], X[sp] * (X[sp][:, [SPY]] < 0)]) for sp in X})):
            r = ridge_design(XX, Y)
            print(line(f"{lab}（val {r['val_ic']:+.4f}）", r["test"], Y["test"], rB))


# ---------------------------------------------------------------- S：供應鏈先驗圖
ELEC = {"半導體業", "電腦及週邊設備業", "其他電子業", "光電業", "通信網路業", "電子零組件業"}
MAT = {"水泥工業", "塑膠工業", "紡織纖維", "鋼鐵工業", "橡膠工業"}
FIRM = {  # (i) 同公司 / 同集團 / 同業 + (ii) 客戶與供應商（登記原文，§64.5）
    "TSM": ["2330"], "UMC": ["2303"], "ASX": ["3711"], "CHT": ["2412", "3045", "4904"], "IMOS": ["8150"],
    "AUOTY": ["2409", "3481"], "HNHPF": ["2317", "2354"],
    "AAPL": ["2317", "4938", "2382", "2330", "3008", "2474", "2354", "3711"],
    "NVDA": ["2330", "3711", "2382", "2317", "2357", "2308"],
    "AMD": ["2330", "3711", "2357"], "QCOM": ["2330", "3711", "2454", "2303"], "AVGO": ["2330", "3711"],
    "MRVL": ["2330", "3711"], "TXN": ["2303", "3711"], "ADI": ["2330", "2303"],
    "INTC": ["2330", "2357", "2382", "4938", "2301", "2395"], "MU": ["2408", "8150"],
    "ASML": ["2330", "2303", "2408"], "AMAT": ["2330", "2303", "2408"], "LRCX": ["2330", "2303", "2408"],
    "KLAC": ["2330", "2303", "2408"], "TER": ["2330", "3711", "8150"],
    "SNPS": ["2330", "2454"], "CDNS": ["2330", "2454"],
}
SECTOR = {  # (iii) ETF -> 產業（產業取 universe 的 industry 欄）
    "SPY": TW,
    "QQQ": [c for c in TW if U.industry[c] in ELEC],
    "SMH": [c for c in TW if U.industry[c] == "半導體業"],
    "XLF": [c for c in TW if U.industry[c] == "金融保險業"] + ["5871"],
    "XLB": [c for c in TW if U.industry[c] in MAT],
    "XLE": [c for c in TW if U.industry[c] in ("油電燃氣業", "塑膠工業")],
}
KAPPAS = (1, 2, 4, 8, 16, 64, np.inf)


def prior(*dicts):
    P = np.zeros((30, 50), bool)
    P[SPY, :] = True                                   # SPY 一律保留
    for d in dicts:
        for u, lst in d.items():
            for c in lst:
                P[US.index(u), TW.index(c)] = True
    return P


PRIORS = {"P（全部）": prior(FIRM, SECTOR), "公司層（i+ii）": prior(FIRM), "產業層（iii）": prior(SECTOR)}


def placebo(P, rng):
    Q = np.zeros_like(P)
    others = [i for i in range(30) if i != SPY]
    Q[SPY, :] = True
    for j in range(50):
        Q[rng.choice(others, size=int(P[others, j].sum()), replace=False), j] = True
    return Q


class PriorFitter:
    """固定一折的標準化設計，對任一先驗解 (alpha, κ) 路徑並以 val 平均 IC 選。"""

    def __init__(self, D, kappas=KAPPAS):
        Xtr, Xva, Xte = X80(D, "train"), X80(D, "val"), X80(D, "test")
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        self.Z = {"train": (Xtr - mu) / sd, "val": (Xva - mu) / sd, "test": (Xte - mu) / sd}
        Ttr = zrow(D["y_train"])
        self.ZtZ = self.Z["train"].T @ self.Z["train"]
        self.ZtT = self.Z["train"].T @ (Ttr - Ttr.mean(0))
        self.Yv, self.Yt, self.kappas = D["y_val"], D["y_test"], kappas

    def fit(self, P):
        nA, nK = len(T.ALPHAS), len(self.kappas)
        Pv = np.zeros((nK, nA) + self.Yv.shape)
        Pt = np.zeros((nK, nA) + self.Yt.shape)
        for ki, kap in enumerate(self.kappas):
            for j in range(50):
                s = np.ones(80)
                s[:30][~P[:, j]] = 0.0 if np.isinf(kap) else 1.0 / np.sqrt(kap)
                e, V = np.linalg.eigh(self.ZtZ * np.outer(s, s))
                B = V @ ((V.T @ (s * self.ZtT[:, j]))[:, None] / (e[:, None] + T.ALPHAS[None, :]))
                W = s[:, None] * B
                Pv[ki, :, :, j] = (self.Z["val"] @ W).T
                Pt[ki, :, :, j] = (self.Z["test"] @ W).T
        vic = np.array([[np.nanmean(ser(Pv[ki, a], self.Yv)) for a in range(nA)] for ki in range(nK)])
        ki, a = np.unravel_index(np.nanargmax(vic), vic.shape)
        return dict(kappa=self.kappas[ki], alpha=T.ALPHAS[a], val_ic=vic[ki, a], test=Pt[ki, a],
                    base_test=Pt[0, int(np.nanargmax(vic[0]))])


def auc_stat(A, P):
    rows = [i for i in range(30) if i != SPY]
    vals = []
    for j in range(50):
        pos = np.array([abs(A[i, j]) for i in rows if P[i, j]])
        neg = np.array([abs(A[i, j]) for i in rows if not P[i, j]])
        if len(pos) and len(neg):
            vals.append((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return float(np.mean(vals))


def block_s(n_placebo: int):
    print("\n==================== S：供應鏈先驗圖")
    rng = np.random.default_rng(42)
    for f in FOLDS:
        D = fold(f)
        F = PriorFitter(D)
        Yt = D["y_test"]
        print(f"\n######## {f}")
        for name, P in PRIORS.items():
            r = F.fit(P)
            rb, rp = ser(r["base_test"], Yt), ser(r["test"], Yt)
            d_real = float(np.nanmean(rp - rb))
            pl = np.array([float(np.nanmean(ser(F.fit(placebo(P, rng))["test"], Yt) - rb)) for _ in range(n_placebo)])
            print(f"  [主] {name:12s} 邊數（不含 SPY）{int(P.sum() - 50):3d}  val 選 κ={r['kappa']} alpha={r['alpha']:.3g}  "
                  f"test dIC {d_real:+.4f}（HAC p {hac(rp, rb):.4f}）  安慰劑第 95 百分位 {np.quantile(pl, 0.95):+.4f}  "
                  f"安慰劑選到 κ>1 而有差異的比例 {(pl != 0).mean():.2f}", flush=True)
    print("\n  -- 事後 1：只用先驗邊（κ = ∞），對 100 組同度數隨機先驗")
    rng = np.random.default_rng(7)
    for f in FOLDS:
        D = fold(f)
        F = PriorFitter(D, kappas=(np.inf,))
        Yt = D["y_test"]
        for name, P in PRIORS.items():
            r = F.fit(P)
            tr = float(np.nanmean(ser(r["test"], Yt)))
            pv, pt = [], []
            for _ in range(n_placebo):
                q = F.fit(placebo(P, rng))
                pv.append(q["val_ic"])
                pt.append(float(np.nanmean(ser(q["test"], Yt))))
            pv, pt = np.array(pv), np.array(pt)
            print(f"  {f} {name:12s} val IC {r['val_ic']:+.4f}（安慰劑中位數 {np.median(pv):+.4f}，百分位 {100 * (pv < r['val_ic']).mean():.0f}）"
                  f"  test IC {tr:+.4f}（中位數 {np.median(pt):+.4f}，百分位 {100 * (pt < tr).mean():.0f}）", flush=True)
    s_posthoc2(n_placebo)


def s_posthoc2(n_placebo: int):
    """S 的事後 2，單獨成函式，--blocks S2 可以只跑這段（隨機數序列與在 S 裡跑時相同）。"""
    import beta_industry_clustering as BIC
    print("\n  -- 事後 2：MAGNET 學到的 B 與線性-z 的係數是否偏重先驗邊（AUC，每欄先去均值，不含 SPY）")
    rng = np.random.default_rng(11)
    for f, armname in MAIN.items():
        Bs = [B - B.mean(0, keepdims=True) for B in (BIC.load_B(d) for d in BIC.find_seeds(armname).values())
              if B is not None]
        C = linz(fold(f))["coef"][:30]
        for name, P in PRIORS.items():
            pls = [placebo(P, rng) for _ in range(n_placebo)]
            real = np.mean([auc_stat(B, P) for B in Bs])
            null = np.array([np.mean([auc_stat(B, Q) for B in Bs]) for Q in pls])
            print(f"  {f} {name:12s} MAGNET B（{len(Bs)} 顆）AUC {real:.3f}（中位數 {np.median(null):.3f}，"
                  f"百分位 {100 * (null < real).mean():.0f}）", flush=True)
        for name, P in PRIORS.items():
            real = auc_stat(C, P)
            null = np.array([auc_stat(C, placebo(P, rng)) for _ in range(n_placebo)])
            print(f"  {f} {name:12s} 線性-z 係數 AUC {real:.3f}（中位數 {np.median(null):.3f}，"
                  f"百分位 {100 * (null < real).mean():.0f}）", flush=True)


# ---------------------------------------------------------------- E：early fusion
EF_OLD = {"第一折": ("tw50chk_ef", True), "第二折": ("f2_ef", False)}
EF_D3 = {"第一折": "tw50_efD3", "第二折": "f2_efD3"}


def gnorm(P, ref):
    return (P - np.nanmean(P, 1, keepdims=True)) / ref


def gsd(A):
    return np.nanmean(np.nanstd(A, 1))


def three_way(D, M, Mv, E, Ev, lab):
    """協定照 ceiling/probe_3way.py：線性權重 a、EF 權重 b 在 {0..2}² 以 val（種子集成）IC 選；兩方基準 w = 0.6。"""
    gr = np.round(np.arange(0, 2.01, 0.1), 2)
    Y, Yv = D["y_test"], D["y_val"]
    cl = linz(D)
    L, Lv = gnorm(cl["test"], gsd(cl["val"])), gnorm(cl["val"], gsd(cl["val"]))
    seeds = sorted(set(M) & set(E) & set(Mv) & set(Ev))
    Me, Ee = np.nanmean([Mv[s] for s in seeds], 0), np.nanmean([Ev[s] for s in seeds], 0)
    _, a, b = max(((summ(gnorm(Me, gsd(Me)) + a * Lv + b * gnorm(Ee, gsd(Ee)), Yv)["IC"], a, b) for a in gr for b in gr))
    r2 = np.array([summ(gnorm(M[s], gsd(Mv[s])) + 0.6 * L, Y)["ICIR"] for s in seeds])
    r3 = np.array([summ(gnorm(M[s], gsd(Mv[s])) + a * L + b * gnorm(E[s], gsd(Ev[s])), Y)["ICIR"] for s in seeds])
    MeT, EeT = np.nanmean([M[s] for s in seeds], 0), np.nanmean([E[s] for s in seeds], 0)
    s2 = summ(gnorm(MeT, gsd(Me)) + 0.6 * L, Y)
    s3 = summ(gnorm(MeT, gsd(Me)) + a * L + b * gnorm(EeT, gsd(Ee)), Y)
    print(f"  [{lab}] n={len(seeds)}  val 選 a={a:.1f} b={b:.1f}  兩方 ICIR {r2.mean():.4f} 三方 {r3.mean():.4f} "
          f"差 {np.mean(r3 - r2):+.4f}（配對 t p {stats.ttest_rel(r3, r2).pvalue:.4f}，{int((r3 > r2).sum())}/{len(seeds)}）")
    corr = np.nanmean([np.corrcoef(x, y)[0, 1] for x, y in zip(MeT, EeT) if np.nanstd(x) > 0 and np.nanstd(y) > 0])
    print(f"        集成：兩方 IC {s2['IC']:+.4f} ICIR {s2['ICIR']:.4f}；三方 IC {s3['IC']:+.4f} ICIR {s3['ICIR']:.4f}；"
          f"逐日 IC 差 {np.nanmean(s3['ser'] - s2['ser']):+.4f}（HAC p {hac(s3['ser'], s2['ser']):.4f}）；"
          f"MAGNET 與 EF 預測相關 {corr:+.3f}")


def persist(P):
    Z = zrow(np.nan_to_num(P))
    return float(np.nanmean([np.corrcoef(Z[t], Z[t - 1])[0, 1] for t in range(1, len(Z))]))


def block_e():
    print("\n==================== E：early fusion 的正交訊號與 D3")
    for f in FOLDS:
        D = fold(f)
        Y = D["y_test"]
        M, Mv = load(MAIN[f], "test", D), load(MAIN[f], "val", D)
        pat, re = EF_OLD[f]
        Eo, Eov = load(pat, "test", D, reeval=re), load(pat, "val", D)
        Ed, Edv = load(EF_D3[f], "test", D), load(EF_D3[f], "val", D)
        print(f"\n######## {f}  主 arm {len(M)} / 舊版 EF {len(Eo)} / D3 {len(Ed)}")
        three_way(D, M, Mv, Eo, Eov, "舊版 EF，" + ("主判準" if f == "第二折" else "原始發現（非確認性）"))
        three_way(D, M, Mv, Ed, Edv, "D3 EF（次要）")
        icM, irM, _ = seed_stats(M, Y)
        rM = seedmean_ser(M, Y)
        for lab, dct in (("舊版 EF", Eo), ("D3 EF", Ed)):
            ic, ir, rc = seed_stats(dct, Y)
            print(f"  {lab:8s} IC {ic.mean():+.4f}±{ic.std(ddof=1):.4f}  ICIR {ir.mean():.4f}  RankIC {rc.mean():+.4f} | "
                  f"主 arm − 它 {icM.mean() - ic.mean():+.4f}（Welch p {stats.ttest_ind(icM, ic, equal_var=False).pvalue:.4g}；"
                  f"逐日 HAC p {hac(rM, seedmean_ser(dct, Y)):.4g}）")
        ic_o, ic_d = seed_stats(Eo, Y)[0], seed_stats(Ed, Y)[0]
        print(f"  D3 − 舊版 EF {ic_d.mean() - ic_o.mean():+.4f}（Welch p {stats.ttest_ind(ic_d, ic_o, equal_var=False).pvalue:.4g}；"
              f"逐日 HAC p {hac(seedmean_ser(Ed, Y), seedmean_ser(Eo, Y)):.4g}）")
        for lab, P in (("主 arm 集成", ens(M)), ("舊版 EF 集成", ens(Eo)), ("線性-z", linz(D)["test"])):
            s = summ(P, Y)
            print(f"  [事後] {lab:10s} 逐日橫截面自相關 {persist(P):+.3f}  IC {s['IC']:+.4f}  sd {s['sd']:.4f}")


# ---------------------------------------------------------------- L / P2 / P3
def block_l():
    print("\n==================== L：IC 損失（--ic-weight 1.0）")
    for f, name in {"第一折": "tw50_icl", "第二折": "f2_icl"}.items():
        D = fold(f)
        Y = D["y_test"]
        M, A = load(MAIN[f], "test", D), load(name, "test", D)
        seeds = sorted(set(M) & set(A))
        Ms, As = {s: M[s] for s in seeds}, {s: A[s] for s in seeds}
        icM, irM, rcM = seed_stats(Ms, Y)
        icA, irA, rcA = seed_stats(As, Y)
        rM, rA, rK = seedmean_ser(Ms, Y), seedmean_ser(As, Y), ser(ktw(f, D), Y)
        d = irA - irM
        print(f"\n######## {f}  共同種子 {len(seeds)}")
        print(f"  主 arm  IC {icM.mean():+.4f}±{icM.std(ddof=1):.4f}  ICIR {irM.mean():.4f}  RankIC {rcM.mean():+.4f}")
        print(f"  IC 損失 IC {icA.mean():+.4f}±{icA.std(ddof=1):.4f}  ICIR {irA.mean():.4f}  RankIC {rcA.mean():+.4f}")
        print(f"  [主] ICIR 同種子配對差 {d.mean():+.4f}（配對 t p {stats.ttest_rel(irA, irM).pvalue:.4f}，{int((d > 0).sum())}/{len(d)}）；"
              f"IC 差 {icA.mean() - icM.mean():+.4f}（護欄 −0.0082：{'守住' if icA.mean() - icM.mean() >= -0.0082 else '破'}）")
        print(f"  [次] 種子平均逐日 IC 對主 arm {np.nanmean(rA - rM):+.4f}（HAC p {hac(rA, rM):.4f}）；"
              f"對 KTW+ {np.nanmean(rA - rK):+.4f}（HAC p {hac(rA, rK):.4f}）")
        coll = [int((np.nanstd(A[s], 1) < 1e-12).sum()) for s in seeds]
        ok = np.nanstd(Y, 1) > 1e-12                       # 排除 y 全為 0 的補值日（分母為 0）
        disp = np.mean([np.nanmean(np.nanstd(A[s][ok], 1) / np.nanstd(Y[ok], 1)) for s in seeds])
        dispM = np.mean([np.nanmean(np.nanstd(M[s][ok], 1) / np.nanstd(Y[ok], 1)) for s in seeds])
        print(f"  [次] 塌縮天數（逐種子）{coll}；離散比 {disp:.2f}（主 arm {dispM:.2f}）")


def block_p2():
    print("\n==================== P2：第二折單市場消融")
    f = "第二折"
    D = fold(f)
    Y = D["y_test"]
    M, A = load(MAIN[f], "test", D), load("f2_smktA", "test", D)
    icM, irM, rcM = seed_stats(M, Y)
    icA, irA, rcA = seed_stats(A, Y)
    print(f"  主 arm  IC {icM.mean():+.4f}±{icM.std(ddof=1):.4f}  ICIR {irM.mean():.4f}  RankIC {rcM.mean():+.4f}")
    print(f"  單市場  IC {icA.mean():+.4f}±{icA.std(ddof=1):.4f}  ICIR {irA.mean():.4f}  RankIC {rcA.mean():+.4f}（n={len(A)}）")
    print(f"  [主] 主 arm − 單市場 {icM.mean() - icA.mean():+.4f}（Welch p {stats.ttest_ind(icM, icA, equal_var=False).pvalue:.3g}；"
          f"逐日 HAC p {hac(seedmean_ser(M, Y), seedmean_ser(A, Y)):.3g}）；佔主 arm IC 的 "
          f"{100 * (icM.mean() - icA.mean()) / icM.mean():.0f}%")


def block_p3():
    print("\n==================== P3：beta_init_std 敏感度（主 arm = 0.3）")
    arms = {"第一折": {"0.1": "tw50_bis01", "1.0": "tw50_bis10"}, "第二折": {"0.1": "f2_bis01", "1.0": "f2_bis10"}}
    res = {}
    for f, a in arms.items():
        D = fold(f)
        Y = D["y_test"]
        icM, irM, _ = seed_stats(load(MAIN[f], "test", D), Y)
        print(f"\n######## {f}  0.3：IC {icM.mean():+.4f}±{icM.std(ddof=1):.4f}  ICIR {irM.mean():.4f}")
        for v, name in a.items():
            A = load(name, "test", D)
            if not A:
                print(f"  {v}：沒有 run")
                continue
            ic, ir, rc = seed_stats(A, Y)
            p = stats.ttest_ind(ic, icM, equal_var=False).pvalue
            coll = sum(int((np.nanstd(A[s], 1) < 1e-12).sum()) for s in A)
            res.setdefault(v, []).append((ic.mean() - icM.mean(), p))
            print(f"  {v}：n={len(A)} IC {ic.mean():+.4f}±{ic.std(ddof=1):.4f}  ICIR {ir.mean():.4f}  RankIC {rc.mean():+.4f}  "
                  f"dIC {ic.mean() - icM.mean():+.4f}（Welch p {p:.4f}）dICIR {ir.mean() - irM.mean():+.4f}  塌縮天數 {coll}")
    for v, lst in res.items():
        if len(lst) == 2:
            sens = np.sign(lst[0][0]) == np.sign(lst[1][0]) and all(p < 0.05 for _, p in lst)
            print(f"  判定 {v}：" + (f"對初始化敏感（兩折同向顯著，{'較好' if lst[0][0] > 0 else '較差'}）" if sens
                                    else "在 0.1–1.0 間穩健（未達兩折同向顯著）"))


# ---------------------------------------------------------------- POOL
def block_pool():
    print("\n==================== POOL：兩折 test 窗串接（描述性，非登記）")
    gbm = pickle.load(open(CACHE / "N_gbm.pkl", "rb"))
    mlp = pickle.load(open(CACHE / "N_mlp.pkl", "rb"))
    R = pickle.load(open(CACHE / "R.pkl", "rb"))
    S = {}
    for f in FOLDS:
        D = fold(f)
        Y = D["y_test"]
        M = load(MAIN[f], "test", D)
        S[f] = dict(lin=ser(linz(D)["test"], Y), ktw=ser(ktw(f, D), Y), gbm=ser(gbm[f]["test"], Y),
                    mlp=ser(ens(mlp[f]["test"]), Y), magS=seedmean_ser(M, Y), magE=ser(ens(M), Y),
                    vix=R[f]["VIXlvl"], spy=R[f]["SPYabs"])
    print(f"  {'比較':32s} {'第一折':>8s} {'第二折':>8s} {'合併':>8s} {'合併 HAC p':>10s}")
    for lab, a, b in (("GBM − 線性-z", "gbm", "lin"), ("MLP 集成 − 線性-z", "mlp", "lin"),
                      ("VIX 交互作用 − 線性-z", "vix", "lin"), ("|SPY| 交互作用 − 線性-z", "spy", "lin"),
                      ("主 arm（種子平均）− 線性-z", "magS", "lin"), ("主 arm（種子平均）− KTW+", "magS", "ktw"),
                      ("主 arm（集成）− KTW+", "magE", "ktw"), ("GBM − 主 arm（種子平均）", "gbm", "magS"),
                      ("GBM − KTW+", "gbm", "ktw")):
        d = np.concatenate([S["第二折"][a] - S["第二折"][b], S["第一折"][a] - S["第一折"][b]])
        d = d[np.isfinite(d)]
        d1 = np.nanmean(S["第一折"][a] - S["第一折"][b])
        d2 = np.nanmean(S["第二折"][a] - S["第二折"][b])
        print(f"  {lab:32s} {d1:+8.4f} {d2:+8.4f} {d.mean():+8.4f} {T.hac_p(d):10.4f}  (n={len(d)})")


# ---------------------------------------------------------------- main
def main() -> None:
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", default=["N", "R", "A", "S", "E", "L", "P2", "P3", "POOL"])
    ap.add_argument("--cache", type=str, default=str(CACHE))
    ap.add_argument("--no-ext", action="store_true", help="區塊 N 不跑事後的延伸網格")
    ap.add_argument("--placebo", type=int, default=100)
    # 2026-10-27 起主結果是 SA（tw50_rgcnSA / f2_rgcnSA，§83）；預設仍是原主結果，§64 的數字照舊重現
    ap.add_argument("--arm", default=MAIN["第一折"])
    ap.add_argument("--arm-f2", default=MAIN["第二折"])
    a = ap.parse_args()
    MAIN["第一折"], MAIN["第二折"] = a.arm, a.arm_f2
    CACHE = Path(a.cache)
    CACHE.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    for b in a.blocks:
        {"N": lambda: block_n(not a.no_ext), "R": block_r, "A": block_a, "S": lambda: block_s(a.placebo),
         "S2": lambda: s_posthoc2(a.placebo),
         "E": block_e, "L": block_l, "P2": block_p2, "P3": block_p3, "POOL": block_pool}[b]()


if __name__ == "__main__":
    main()
