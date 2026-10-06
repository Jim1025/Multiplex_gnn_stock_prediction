"""tw_lag_channel.py — 台股層內線性領先落後通道 ④（proposal §62，事前登記 P10 / P11）

§60.2a 的未解問題：主結果 arm 兩折都是「IC 小贏、ICIR 大輸」KTW+，七個候選解釋都已排除。
本腳本把 KTW+（per-target ridge，30 檔美股 + 50 檔台股昨日報酬，rank 目標）拆開，
找到缺的那一塊，並固定之後檢驗 ④ 要用的評估程式。

區塊：

  A. KTW+ 拆解        per-target ridge 逐項加輸入：台股落後區塊讓 sd(IC) 在 6 格全部下降
  B. 分散             台股落後訊號與美股訊號、與 MAGNET 的逐日 IC 相關，以及預測相關
  C. 事後疊加          MAGNET 集成 + w x 台股落後通道（w 在 val 選）：ICIR 追平 KTW+；
                      截距單獨無效、去掉截距仍保留大部分增益
  D. 正負號結構        登記版本的 C：自身反轉、同業延續、跨大類反向
  E. 自身昨日報酬負載   MAGNET 對自身昨日報酬的負載很重，但移除它 val ICIR 下降（不是原因）
  F. 登記的通道        P10 的 C、alpha 與 w（兩折各自，規則見 §62.6）
  G. P10a             逐種子事後組合（不重訓）的事前判準檢驗——**登記並 commit 之後才跑**
  H. P10b             重訓版（④ 內建、凍結）的事前判準檢驗——run 完成後才跑

通道的定義（登記版本，§62.5）：
    z_k   = (r_{k,t−1} − μ_k) / σ_k          50 檔台股昨日 log return，μ、σ 取訓練窗
    ĉ_j   = Σ_k C[k,j] · z_k                 C 由 per-target ridge 解出：目標為逐日橫截面
                                              z-score 的 y，無截距，含對角，alpha 以 val 平均 IC 選
    ŷ'_j  = z(ŷ_j) + w · z(ĉ_j)             z(·) 為逐日橫截面標準化；IC 只看這個組合

只讀 MultiplexDataset、`runs/` 與 `runs_f2/` 的預測檔，不寫入任何檔案。

用法：
    .venv/bin/python scripts/tw_lag_channel.py --blocks A B C D E F
    .venv/bin/python scripts/tw_lag_channel.py --blocks G        # 登記之後
    .venv/bin/python scripts/tw_lag_channel.py --blocks H        # P10b 的 run 完成之後
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from factor_vs_graph import ALPHAS, RET, collect, rank_transform, zscore_transform  # noqa: E402
from src.dataset.config import load_universe  # noqa: E402
from src.dataset.multiplex_dataset import MultiplexDataset  # noqa: E402

CUT = "2024-12-25"
GRID = np.round(np.arange(0, 1.01, 0.1), 2)
MDE = 0.0082            # n=10 的跨種子最小可偵測差異（主表腳註）
FOLDS = {
    "第一折": dict(cfg="configs/tw50.yaml", f2=False,
                  main="runs/**/*tw50_betaF1nA2r1_s*/predictions/{}_predictions.csv",
                  ktw="runs/linear/*_fvg_KTWp/predictions/test_predictions.csv",
                  p10b="runs/**/*tw50_p10cf_s*/predictions/{}_predictions.csv"),
    "第二折": dict(cfg="configs/tw50_fold2.yaml", f2=True,
                  main="runs/**/*f2_best_s*/predictions/{}_predictions.csv",
                  ktw="runs_f2/*fvg_KTWp/predictions/*.csv",
                  p10b="runs/**/*f2_p10cf_s*/predictions/{}_predictions.csv"),
}
PAIRED = {"2330", "2303", "3711", "2412", "8150", "2409", "2317"}
ELEC = {"半導體業", "電腦及週邊設備業", "其他電子業", "光電業", "通信網路業", "電子零組件業"}
BLOCKS = "ABCDEFGH"


# ---------------------------------------------------------------- 資料
def load_fold(cfg_path: str, fold2: bool) -> dict:
    cfg = yaml.safe_load(open(ROOT / cfg_path))
    kw = dict(snapshot_dir=cfg["data"]["snapshot_dir"], features_dir=cfg["data"]["features_dir"],
              T=cfg["model"]["lstm"]["T_history"], config_path=cfg_path)
    out = {}
    for sp in ("train", "val", "test"):
        ds = MultiplexDataset(split=sp, **kw)
        d = collect(ds)
        dates = np.array([str(x)[:10] for x in d["dates"]])
        keep = dates <= CUT if (fold2 and sp == "test") else np.ones(len(dates), bool)
        out[f"us_{sp}"] = d["X1"][keep, -1, :, RET].astype(np.float64)
        out[f"tw_{sp}"] = d["X2"][keep, -1, :, RET].astype(np.float64)
        out[f"y_{sp}"] = d["Y"][keep].astype(np.float64)
        out[f"d_{sp}"] = dates[keep]
    out["codes"] = [str(c) for c in ds.tw_codes]
    return out


def preds(pattern: str, split: str, dates, codes) -> tuple[np.ndarray, list[int]]:
    """pattern -> (逐種子預測 [S, T, n], 種子清單)，對齊資料集的日期與代號順序。"""
    Hs, seeds = [], []
    for f in sorted(glob.glob(str(ROOT / pattern.format(split)), recursive=True)):
        tail = Path(f).parents[1].name.rsplit("_s", 1)[-1]
        s = int(tail) if tail.isdigit() else 0          # 線性 baseline 沒有種子
        df = pd.read_csv(f, dtype={"ticker": str})
        df["target_date"] = df["target_date"].astype(str).str[:10]
        H = df.pivot(index="target_date", columns="ticker", values="y_hat")
        Hs.append(H.reindex(index=dates, columns=codes).to_numpy(float))
        seeds.append(s)
    order = np.argsort(seeds)
    return (np.stack([Hs[i] for i in order]) if Hs else np.empty((0,))), [seeds[i] for i in order]


# ---------------------------------------------------------------- 統計
def ic_ser(P, Y, rank=False):
    s = []
    for t in range(len(Y)):
        a, b = P[t], Y[t]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            s.append(np.nan)
            continue
        s.append(stats.spearmanr(a, b).statistic if rank else np.corrcoef(a, b)[0, 1])
    return np.array(s)


def summ(P, Y) -> dict:
    p, r = ic_ser(P, Y), ic_ser(P, Y, rank=True)
    ser = p.copy()
    p, r = p[~np.isnan(p)], r[~np.isnan(r)]
    return dict(IC=p.mean(), sd=p.std(ddof=1), ICIR=p.mean() / p.std(ddof=1),
                RIC=r.mean(), RICIR=r.mean() / r.std(ddof=1), ser=ser)


def zrow(P):
    P = P - np.nanmean(P, 1, keepdims=True)
    sd = np.nanstd(P, 1, keepdims=True)
    return P / np.where(sd < 1e-12, 1.0, sd)


def hac_p(d) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    x = d - d.mean()
    var = float(x @ x) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float(x[k:] @ x[:-k]) / n
    return float(2 * stats.t.sf(abs(d.mean() / np.sqrt(max(var, 1e-24) / n)), n - 1))


# ---------------------------------------------------------------- per-target ridge（與 factor_vs_graph 同協定）
def _std_cols(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [(X - mu) / sd for X in (Xtr, *others)]


def _path(Xtr, Ttr):
    Tm = Ttr.mean(0)
    s, V = np.linalg.eigh(Xtr.T @ Xtr)
    VtXtT = V.T @ (Xtr.T @ (Ttr - Tm))
    return [(V @ (VtXtT / (s[:, None] + a)), Tm) for a in ALPHAS]


def _design(D, sp, j, name):
    us, tw = D[f"us_{sp}"], D[f"tw_{sp}"]
    return {"US30": lambda: us, "US30+own": lambda: np.hstack([us, tw[:, [j]]]),
            "US30+TW49": lambda: np.hstack([us, np.delete(tw, j, axis=1)]),
            "US30+TW50": lambda: np.hstack([us, tw]), "TW50": lambda: tw,
            "TW49": lambda: np.delete(tw, j, axis=1), "own": lambda: tw[:, [j]],
            "TWmkt": lambda: tw.mean(1, keepdims=True)}[name]()


SHARED = {"US30", "US30+TW50", "TW50", "TWmkt"}


def ridge(D, name, mode, intercept=True) -> dict:
    """per-target ridge，50 個目標共用一個 alpha，以 val 平均 IC 選。

    mode：raw | zscore | rank（擬合目標）。intercept=False 時預測不加逐檔截距。
    回傳 alpha、val IC、val / test 預測，以及共用設計時的係數 [p, 50]。
    """
    n = D["y_train"].shape[1]
    Ttr = {"raw": lambda Y: Y, "zscore": zscore_transform, "rank": rank_transform}[mode](D["y_train"])
    P = {sp: np.zeros((len(ALPHAS),) + D[f"y_{sp}"].shape) for sp in ("val", "test")}
    coefs = None
    for j in range(1 if name in SHARED else n):
        Xtr, Xva, Xte = _std_cols(*(_design(D, sp, j, name) for sp in ("train", "val", "test")))
        T = Ttr if name in SHARED else Ttr[:, [j]]
        path = _path(Xtr, T)
        if name in SHARED:
            coefs = path
        for k, (C, b) in enumerate(path):
            b = b if intercept else 0.0
            if name in SHARED:
                P["val"][k], P["test"][k] = Xva @ C + b, Xte @ C + b
            else:
                P["val"][k][:, j], P["test"][k][:, j] = (Xva @ C + b)[:, 0], (Xte @ C + b)[:, 0]
    vic = [np.nanmean(ic_ser(P["val"][k], D["y_val"])) for k in range(len(ALPHAS))]
    k = int(np.nanargmax(vic))
    return dict(alpha=ALPHAS[k], val_IC=vic[k], val=P["val"][k], test=P["test"][k],
                coef=None if coefs is None else coefs[k][0])


def channel(D) -> dict:
    """P10 的登記版本：台股 50 檔、z-score 目標、無截距、含對角。"""
    return ridge(D, "TW50", "zscore", intercept=False)


def pick_w(mva, cva, yva) -> float:
    """w 的登記規則：z(MAGNET 集成) + w·z(ĉ)，grid 0..1 步長 0.1，取 val 平均 IC 最大者。"""
    return float(GRID[int(np.argmax([summ(zrow(mva) + g * zrow(cva), yva)["IC"] for g in GRID]))])


def shrink_own(P, L, lam):
    out = P.copy()
    for t in range(len(P)):
        l, p = L[t] - L[t].mean(), P[t] - P[t].mean()
        if l @ l > 1e-18:
            out[t] = p - lam * (p @ l) / (l @ l) * l
    return out


def xs_corr(A, B) -> float:
    v = [np.corrcoef(A[t], B[t])[0, 1] for t in range(len(A))
         if np.std(A[t]) > 1e-12 and np.std(B[t]) > 1e-12]
    return float(np.mean(v))


# ---------------------------------------------------------------- 區塊
def block_a(D):
    print("[A] KTW+ 拆解：per-target ridge 逐項加輸入（原始 y 目標；末列為 KTW+ 本身）")
    print(f"    {'輸入':24s} {'IC':>8s} {'sd(IC)':>8s} {'ICIR':>7s}")
    for name, mode in (("US30", "raw"), ("US30+own", "raw"), ("US30+TW49", "raw"),
                       ("US30+TW50", "raw"), ("US30+TW50", "rank")):
        s = summ(ridge(D, name, mode)["test"], D["y_test"])
        lab = name + ("（rank，= KTW+）" if mode == "rank" else "")
        print(f"    {lab:24s} {s['IC']:+8.4f} {s['sd']:8.4f} {s['ICIR']:7.4f}")
    print("    台股落後區塊對 sd(IC) 的效果（US30 -> US30+TW50，三種目標）：", end="")
    for mode in ("raw", "zscore", "rank"):
        a = summ(ridge(D, "US30", mode)["test"], D["y_test"])["sd"]
        b = summ(ridge(D, "US30+TW50", mode)["test"], D["y_test"])["sd"]
        print(f"  {mode} {(b / a - 1) * 100:+.1f}%", end="")
    print()


def block_b(D, M):
    print("[B] 分散：逐日 IC 的相關")
    for mode in ("raw", "rank"):
        a = summ(ridge(D, "US30", mode)["test"], D["y_test"])["ser"]
        b = summ(ridge(D, "TW50", mode)["test"], D["y_test"])["ser"]
        ok = ~(np.isnan(a) | np.isnan(b))
        print(f"    {mode:6s} corr_t(IC 美股 30 檔, IC 台股 50 檔落後) = {np.corrcoef(a[ok], b[ok])[0, 1]:+.3f}")
    c = channel(D)
    m = summ(zrow(np.nanmean(M["test"], 0)), D["y_test"])["ser"]
    s = summ(c["test"], D["y_test"])["ser"]
    ok = ~(np.isnan(m) | np.isnan(s))
    print(f"    corr_t(IC MAGNET 集成, IC 登記通道) = {np.corrcoef(m[ok], s[ok])[0, 1]:+.3f}；"
          f"逐日預測相關 {xs_corr(zrow(np.nanmean(M['test'], 0)), zrow(c['test'])):+.3f}")


def block_c(D, M, K):
    print("[C] 事後疊加：z(MAGNET 集成) + w·z(通道)，w 在 val 上以平均 IC 選（grid 0..1）")
    mte, mva = np.nanmean(M["test"], 0), np.nanmean(M["val"], 0)
    base = summ(zrow(mte), D["y_test"])
    k = summ(K, D["y_test"])
    print(f"    {'':30s} {'w':>4s} {'IC':>8s} {'sd':>7s} {'ICIR':>7s} {'RICIR':>7s}")
    print(f"    {'MAGNET 集成（基準）':30s} {'':>4s} {base['IC']:+8.4f} {base['sd']:7.4f} "
          f"{base['ICIR']:7.4f} {base['RICIR']:7.4f}")
    icp = {sp: np.tile(np.nanmean(D["y_train"], 0), (len(D[f"y_{sp}"]), 1)) for sp in ("val", "test")}
    rows = [("台股 50 檔，無截距（登記版本）", channel(D)),
            ("台股 50 檔，含截距", ridge(D, "TW50", "zscore", True)),
            ("只有逐檔截距", dict(val=icp["val"], test=icp["test"])),
            ("其他 49 檔（不含自身），無截距", ridge(D, "TW49", "zscore", False)),
            ("只有自身昨日報酬，無截距", ridge(D, "own", "zscore", False)),
            ("昨日台股大盤（單一純量），無截距", ridge(D, "TWmkt", "zscore", False))]
    for lab, r in rows:
        w = pick_w(mva, r["val"], D["y_val"])
        s = summ(zrow(mte) + w * zrow(r["test"]), D["y_test"])
        print(f"    {lab:30s} {w:4.1f} {s['IC']:+8.4f} {s['sd']:7.4f} {s['ICIR']:7.4f} {s['RICIR']:7.4f}")
    print(f"    {'KTW+（參照）':30s} {'':>4s} {k['IC']:+8.4f} {k['sd']:7.4f} {k['ICIR']:7.4f} {k['RICIR']:7.4f}")


def block_d(D):
    u = load_universe("tw50")
    codes = D["codes"]
    ind = np.array([u.industry[c] for c in codes])
    C = channel(D)["coef"]                      # [50 昨日來源 k, 50 目標 j]
    same, off = ind[:, None] == ind[None, :], ~np.eye(50, dtype=bool)
    g = np.where(np.isin(ind, list(ELEC)), "E", np.where(ind == "金融保險業", "F", "O"))
    cs = g[:, None] == g[None, :]
    dg = np.diag(C)
    nets = [(a, b, C[np.ix_(g == b, g == a)].sum(0).mean()) for a in "EFO" for b in "EFO" if a != b]
    print(f"[D] 正負號結構（登記版本的 C）：自身 {int((dg < 0).sum())}/50 為負；"
          f"同產業 {(C[same & off] > 0).mean() * 100:.0f}% 為正（n={int((same & off).sum())}）；"
          f"跨大類淨負載為負的方向 {sum(v < 0 for *_, v in nets)}/6")
    print("    跨大類淨負載（目標 <- 來源，x1e3）：" + "  ".join(f"{a}<-{b} {v * 1e3:+.1f}" for a, b, v in nets))


def block_e(D, M, K):
    print("[E] 自身昨日報酬：逐日橫截面 corr(預測, r_{j,t−1})")
    lag = D["tw_test"]
    print(f"    實際報酬 {xs_corr(D['y_test'], lag):+.4f}   MAGNET 集成 {xs_corr(np.nanmean(M['test'], 0), lag):+.4f}"
          f"   KTW+ {xs_corr(K, lag):+.4f}")
    for lam in (0.0, 0.5, 1.0):
        v = np.mean([summ(shrink_own(M["val"][s], D["tw_val"], lam), D["y_val"])["ICIR"] for s in range(len(M["val"]))])
        print(f"    移除比例 {lam:.1f}：val ICIR（逐種子平均）{v:.4f}")


def block_f(D, M):
    c = channel(D)
    w = pick_w(np.nanmean(M["val"], 0), c["val"], D["y_val"])
    print(f"[F] 登記的通道：alpha {c['alpha']:.4g}，通道自身 val IC {c['val_IC']:+.4f}，"
          f"test IC {summ(c['test'], D['y_test'])['IC']:+.4f}；登記的 w = {w:.1f}")
    return w


def evaluate(D, A, B, seeds_a, seeds_b, K, label):
    """逐種子配對：A = 主 arm，B = 處理組（同一組種子）。印事前判準要的數字。"""
    common = sorted(set(seeds_a) & set(seeds_b))
    ia, ib = [seeds_a.index(s) for s in common], [seeds_b.index(s) for s in common]
    ra = [summ(A[i], D["y_test"]) for i in ia]
    rb = [summ(B[i], D["y_test"]) for i in ib]
    k = summ(K, D["y_test"])
    print(f"    [{label}] 共同種子 {len(common)} 顆：{common}")
    for key, lab in (("ICIR", "ICIR"), ("IC", "IC"), ("RIC", "RankIC"), ("RICIR", "RankICIR"), ("sd", "sd(IC)")):
        a, b = np.array([r[key] for r in ra]), np.array([r[key] for r in rb])
        d = b - a
        p = stats.ttest_rel(b, a).pvalue if len(d) > 1 else np.nan
        print(f"      {lab:9s} 主 arm {a.mean():.4f} -> {b.mean():.4f}  差 {d.mean():+.4f}"
              f"（配對 p {p:.4f}，{int((d > 0).sum())}/{len(d)} 為正）")
    icir_b = np.array([r["ICIR"] for r in rb])
    print(f"      對 KTW+ 的 ICIR（{k['ICIR']:.4f}）：單樣本差 {icir_b.mean() - k['ICIR']:+.4f}，"
          f"p {stats.ttest_1samp(icir_b, k['ICIR']).pvalue:.4f}，{int((icir_b > k['ICIR']).sum())}/{len(icir_b)} 高於")
    db = np.nanmean([r["ser"] for r in rb], 0) - k["ser"]
    print(f"      對 KTW+ 的逐日 IC（種子平均）：差 {np.nanmean(db):+.4f}，逐日 HAC p {hac_p(db):.4f}")
    dd = np.array([r["ICIR"] for r in rb]) - np.array([r["ICIR"] for r in ra])
    di = np.array([r["IC"] for r in rb]) - np.array([r["IC"] for r in ra])
    return dict(dICIR=dd.mean(), p=stats.ttest_rel([r["ICIR"] for r in rb], [r["ICIR"] for r in ra]).pvalue,
                dIC=di.mean())


def block_g(D, M, seeds, K, w):
    c = channel(D)
    B = np.stack([zrow(M["test"][s]) + w * zrow(c["test"]) for s in range(len(M["test"]))])
    print(f"[G] P10a：逐種子事後組合，w = {w:.1f}（不重訓）")
    r = evaluate(D, M["test"], B, seeds, seeds, K, "P10a 登記版本")
    for lab, alt in (("不含對角（其他 49 檔）", ridge(D, "TW49", "zscore", False)),
                     ("含截距", ridge(D, "TW50", "zscore", True))):
        wa = pick_w(np.nanmean(M["val"], 0), alt["val"], D["y_val"])
        Ba = np.stack([zrow(M["test"][s]) + wa * zrow(alt["test"]) for s in range(len(M["test"]))])
        print(f"    次要（探索，不計入判準）：{lab}，w = {wa:.1f}")
        evaluate(D, M["test"], Ba, seeds, seeds, K, lab)
    return r


def block_h(D, M, seeds, K, pattern):
    P, s_b = preds(pattern, "test", D["d_test"], D["codes"])
    print(f"[H] P10b：重訓版（④ 內建、凍結），找到 {len(s_b)} 顆種子")
    if len(s_b) == 0:
        print("    尚無 run")
        return None
    return evaluate(D, M["test"], P, seeds, s_b, K, "P10b")


def verdict(res: dict, label: str):
    """事前判準：兩折 ICIR 配對 p < 0.05 且同為正；兩折 IC 平均差不低於 −MDE。"""
    if len(res) < 2 or any(v is None for v in res.values()):
        return
    ok_icir = all(v["dICIR"] > 0 and v["p"] < 0.05 for v in res.values())
    ok_ic = all(v["dIC"] >= -MDE for v in res.values())
    print(f"\n== {label} 判定：ICIR 條件 {'通過' if ok_icir else '未通過'}；"
          f"IC 護欄 {'通過' if ok_ic else '未通過'} -> {'通過' if ok_icir and ok_ic else '未通過'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="台股層內線性領先落後通道 ④（proposal §62）")
    ap.add_argument("--blocks", nargs="*", default=list("ABCDEF"), choices=list(BLOCKS))
    args = ap.parse_args()
    res_g, res_h = {}, {}
    for fold, f in FOLDS.items():
        D = load_fold(f["cfg"], f["f2"])
        Mte, seeds = preds(f["main"], "test", D["d_test"], D["codes"])
        Mva, _ = preds(f["main"], "val", D["d_val"], D["codes"])
        M = {"test": Mte, "val": Mva}
        K = preds(f["ktw"], "", D["d_test"], D["codes"])[0][0]
        print(f"\n######## {fold}：測試窗 {D['d_test'][0]} ~ {D['d_test'][-1]}（{len(D['d_test'])} 天），"
              f"主 arm {len(seeds)} 顆種子")
        if "A" in args.blocks:
            block_a(D)
        if "B" in args.blocks:
            block_b(D, M)
        if "C" in args.blocks:
            block_c(D, M, K)
        if "D" in args.blocks:
            block_d(D)
        if "E" in args.blocks:
            block_e(D, M, K)
        w = block_f(D, M) if any(b in args.blocks for b in "FG") else None
        if "G" in args.blocks:
            res_g[fold] = block_g(D, M, seeds, K, w)
        if "H" in args.blocks:
            res_h[fold] = block_h(D, M, seeds, K, f["p10b"])
    if res_g:
        verdict(res_g, "P10a")
    if res_h:
        verdict(res_h, "P10b")


if __name__ == "__main__":
    main()
