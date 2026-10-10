"""p12_explore.py — P12 之前的探索（proposal §63.1）：怎麼走到「MAGNET + 凍結的完整線性通道」

**事後、在已經用過的兩折 test 窗上做的，p 值沒有校正多重比較。** 這支程式只負責重現 §63.1 的
數字；P12 的事前判準由 scripts/hybrid_channel.py 檢驗。

區塊：

  L. 資訊面   報酬類的線性特徵（美股拆隔夜／日盤、台股拆跳空／盤中、週與月的落後、以跳空為訓練
              標籤）。原始 OHLC 取自 data/raw，對齊方式與資料集相同（US 取嚴格早於目標日的最後一個
              交易日；缺漏日以前值補，報酬記為 0）
  M. 組合面   主 arm、④、P10b 疊上完整線性通道（逐日 z 與全域尺度兩種組合）
  K. 對照     別的模型疊上同一個通道是否有增量；線性 + 線性；三方組合（第一折）
  N. 檔數     50 檔的抽樣雜訊佔逐日 IC 變異的比例，與 ICIR 隨檔數的變化

兩種組合的定義：
  逐日 z     z_t(A) + w·z_t(B)，每天各自標準化（④ 與 P12 登記版用的式子）
  全域尺度   (A − 當日均值) / 整段平均的橫截面 sd + w·(同上的 B)，保留每天預測拉多開的資訊

只讀 data/raw、MultiplexDataset 與 runs/、runs_f2/ 的預測檔，不寫入任何檔案。

用法：
    .venv/bin/python scripts/p12_explore.py --blocks L M K N
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from factor_vs_graph import ALPHAS, zscore_transform  # noqa: E402
from src.dataset.config import load_universe  # noqa: E402
from tw_lag_channel import FOLDS, GRID, channel, hac_p, load_fold, preds, ridge, summ, zrow  # noqa: E402

GRID2 = np.round(np.arange(0, 2.01, 0.1), 2)
BLOCKS = "LMKN"


# ---------------------------------------------------------------- 共用
def gnorm(P, ref_sd):
    """全域尺度：扣逐日均值後除以一個整段期間的常數（逐日的相對大小保留）。"""
    return (P - np.nanmean(P, 1, keepdims=True)) / ref_sd


def msd(P) -> float:
    return float(np.nanmean(np.nanstd(P, 1)))


def best_w(score, grid) -> float:
    return float(grid[int(np.nanargmax([score(g) for g in grid]))])


def mean_icir(Ps, Y) -> float:
    return float(np.mean([summ(p, Y)["ICIR"] for p in Ps]))


def dcorr(A, B) -> float:
    return float(np.nanmean([np.corrcoef(a, b)[0, 1] for a, b in zip(A, B)
                             if np.nanstd(a) > 0 and np.nanstd(b) > 0]))


# ---------------------------------------------------------------- L：資訊面
def _raw(market, code):
    df = pd.read_csv(ROOT / f"data/raw/{market}/{code}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    df.index = df.index.strftime("%Y-%m-%d")
    return df


def _panel(frames, cal):
    """逐檔 reindex 到日曆 cal；缺的日子 Close 前值補、Open 設成補過的 Close（報酬記為 0，同資料集）。"""
    C = pd.DataFrame({c: f["Close"] for c, f in frames.items()}).reindex(cal).ffill()
    O = pd.DataFrame({c: f["Open"] for c, f in frames.items()}).reindex(cal)
    return C.to_numpy(float), O.where(O.notna(), C).to_numpy(float)


def split_features(D) -> dict:
    """{sp: {us_cc, us_on, us_id, tw_cc, tw_gap, tw_id, tw_w, tw_m, y_cc, y_gap}}，對齊資料集的目標日。

    us_on = 收盤 -> 隔日開盤，us_id = 開盤 -> 收盤；tw_w = t−5..t−2、tw_m = t−20..t−6（不含昨日）。
    y_cc / y_gap 是從原始價格算的兩種訓練標籤（同一種算法，比較才公平）；評估一律用資料集的 y。
    """
    us_codes = list(load_universe("tw50").us_nodes)                  # 與資料集的美股節點順序相同
    us = {c: _raw("adr", c) for c in us_codes}
    tw = {c: _raw("tw", c) for c in D["codes"]}
    alld = sorted(set(D["d_train"]) | set(D["d_val"]) | set(D["d_test"]))
    tw_cal = sorted(set(alld) | set().union(*[set(v.index) for v in tw.values()]))
    us_cal = np.array(sorted(set().union(*[set(v.index) for v in us.values()])))
    TC, TO = _panel(tw, tw_cal)
    UC, UO = _panel(us, us_cal)
    pos = {d: i for i, d in enumerate(tw_cal)}
    out = {}
    for sp in ("train", "val", "test"):
        R = {k: [] for k in ("us_cc", "us_on", "us_id", "tw_cc", "tw_gap", "tw_id", "tw_w", "tw_m", "y_cc", "y_gap")}
        for t in D[f"d_{sp}"]:
            i = np.searchsorted(us_cal, t) - 1                         # 嚴格早於 t 的最後一個 US 日
            R["us_cc"].append(np.log(UC[i] / UC[i - 1]))
            R["us_on"].append(np.log(UO[i] / UC[i - 1]))
            R["us_id"].append(np.log(UC[i] / UO[i]))
            j = pos[t]
            R["tw_cc"].append(np.log(TC[j - 1] / TC[j - 2]))
            R["tw_gap"].append(np.log(TO[j - 1] / TC[j - 2]))
            R["tw_id"].append(np.log(TC[j - 1] / TO[j - 1]))
            R["tw_w"].append(np.log(TC[j - 2] / TC[j - 6]))
            R["tw_m"].append(np.log(TC[j - 6] / TC[j - 21]))
            R["y_cc"].append(np.log(TC[j] / TC[j - 1]))
            R["y_gap"].append(np.log(TO[j] / TC[j - 1]))
        out[sp] = {k: np.asarray(v, float) for k, v in R.items()}
    return out


def ridge_xy(X: dict, Ytr, Y: dict) -> dict:
    """共用設計矩陣的 per-target ridge（z-score 目標、無截距），alpha 以 val 平均 IC 選。"""
    mu, sd = X["train"].mean(0), X["train"].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = {sp: (X[sp] - mu) / sd for sp in X}
    T = zscore_transform(Ytr)
    s, V = np.linalg.eigh(Z["train"].T @ Z["train"])
    VtXtT = V.T @ (Z["train"].T @ (T - T.mean(0)))
    best = None
    for a in ALPHAS:
        C = V @ (VtXtT / (s[:, None] + a))
        v = summ(Z["val"] @ C, Y["val"])["IC"]
        if best is None or v > best[0]:
            best = (v, C)
    v, C = best
    return dict(val_IC=v, test=Z["test"] @ C)


def block_l(D):
    F = split_features(D)
    Y = {sp: D[f"y_{sp}"] for sp in ("train", "val", "test")}
    X = lambda *ks: {sp: np.hstack([F[sp][k] for k in ks]) for sp in ("train", "val", "test")}
    print("[L] 資訊面（z-score 目標、無截距、val 選 alpha；評估在 c2c 的 y）")
    print(f"    {'設計':30s} {'維度':>4s} {'val IC':>8s} {'test IC':>8s} {'test ICIR':>9s}")
    for lab, ks in (("美股 c2c", ("us_cc",)), ("美股隔夜段", ("us_on",)), ("美股日盤段", ("us_id",)),
                    ("美股隔夜 + 日盤分開", ("us_on", "us_id")), ("台股昨日 c2c（= ④）", ("tw_cc",)),
                    ("台股昨日跳空 + 盤中分開", ("tw_gap", "tw_id")), ("台股昨日 + 週 + 月", ("tw_cc", "tw_w", "tw_m"))):
        r = ridge_xy(X(*ks), Y["train"], Y)
        s = summ(r["test"], Y["test"])
        print(f"    {lab:30s} {X(*ks)['train'].shape[1]:4d} {r['val_IC']:+8.4f} {s['IC']:+8.4f} {s['ICIR']:9.4f}")
    for lab, tgt in (("訓練標籤 = c2c", F["train"]["y_cc"]), ("訓練標籤 = 跳空", F["train"]["y_gap"])):
        r = ridge_xy(X("us_cc"), tgt, Y)
        s = summ(r["test"], Y["test"])
        print(f"    輸入美股 c2c，{lab:14s} val IC {r['val_IC']:+.4f}  test IC {s['IC']:+.4f}  ICIR {s['ICIR']:.4f}")


# ---------------------------------------------------------------- M：組合面
def block_m(D, f):
    Y, Yv = D["y_test"], D["y_val"]
    M, _ = preds(f["main"], "test", D["d_test"], D["codes"])
    Mv, _ = preds(f["main"], "val", D["d_val"], D["codes"])
    P, _ = preds(f["p10b"], "test", D["d_test"], D["codes"])
    Pv, _ = preds(f["p10b"], "val", D["d_val"], D["codes"])
    K = preds(f["ktw"], "", D["d_test"], D["codes"])[0][0]
    c4, cl = channel(D), ridge(D, "US30+TW50", "zscore", False)
    Mve, Pve = np.nanmean(Mv, 0), np.nanmean(Pv, 0)
    rules = {}
    w = best_w(lambda g: summ(zrow(Mve) + g * zrow(c4["val"]), Yv)["IC"], GRID)
    rules["主 arm + ④（逐日 z，= P10a）"] = ([zrow(m) + w * zrow(c4["val"]) for m in Mv],
                                      [zrow(m) + w * zrow(c4["test"]) for m in M])
    w = best_w(lambda g: summ(gnorm(Mve, msd(Mve)) + g * gnorm(c4["val"], msd(c4["val"])), Yv)["IC"], GRID2)
    rules["主 arm + ④（全域尺度）"] = ([gnorm(m, msd(m)) + w * gnorm(c4["val"], msd(c4["val"])) for m in Mv],
                                [gnorm(m, msd(mv)) + w * gnorm(c4["test"], msd(c4["val"])) for m, mv in zip(M, Mv)])
    rules["P10b（④ 內建）"] = (list(Pv), list(P))
    w = best_w(lambda g: summ(zrow(Pve) + g * zrow(cl["val"]), Yv)["IC"], GRID2)
    rules["P10b + 完整線性通道（逐日 z）"] = ([zrow(p) + w * zrow(cl["val"]) for p in Pv],
                                     [zrow(p) + w * zrow(cl["test"]) for p in P])
    w = best_w(lambda g: summ(gnorm(Mve, msd(Mve)) + g * gnorm(cl["val"], msd(cl["val"])), Yv)["IC"], GRID2)
    rules["主 arm + 完整線性通道（全域尺度）"] = ([gnorm(m, msd(m)) + w * gnorm(cl["val"], msd(cl["val"])) for m in Mv],
                                      [gnorm(m, msd(mv)) + w * gnorm(cl["test"], msd(cl["val"])) for m, mv in zip(M, Mv)])
    print("[M] 組合面（逐種子平均的 ICIR）")
    print(f"    {'組合':34s} {'val ICIR':>9s} {'test ICIR':>10s}")
    print(f"    {'主 arm（無 ④）':34s} {mean_icir(Mv, Yv):9.4f} {mean_icir(M, Y):10.4f}")
    for lab, (V, Te) in rules.items():
        print(f"    {lab:34s} {mean_icir(V, Yv):9.4f} {mean_icir(Te, Y):10.4f}")
    kv = ridge(D, "US30+TW50", "rank", True)
    print(f"    {'KTW+（參照）':34s} {summ(kv['val'], Yv)['ICIR']:9.4f} {summ(K, Y)['ICIR']:10.4f}")
    H = rules["主 arm + 完整線性通道（全域尺度）"][1]
    hser = np.nanmean([summ(h, Y)["ser"] for h in H], 0)
    print(f"    全域尺度版對 KTW+ 的逐日 IC 差 {np.nanmean(hser - summ(K, Y)['ser']):+.4f}，"
          f"HAC p {hac_p(hser - summ(K, Y)['ser']):.4f}")
    return M, Mv


# ---------------------------------------------------------------- K：對照
def _arm(arm, D):
    """{種子: test 預測}、{種子: val 預測}；test 優先用 CPU 重評版（較舊的 run 有）。"""
    out, outv = {}, {}
    for d in sorted(glob.glob(str(ROOT / f"runs/**/*{arm}_s*"), recursive=True)):
        p = Path(d) / "predictions"
        tf = p / "test_predictions_reeval.csv"
        tf = tf if tf.exists() else p / "test_predictions.csv"
        vf = p / "val_predictions.csv"
        tail = Path(d).name.rsplit("_s", 1)[-1]
        if not (tf.exists() and vf.exists() and tail.isdigit()):
            continue
        for f, dates, dst in ((tf, D["d_test"], out), (vf, D["d_val"], outv)):
            df = pd.read_csv(f, dtype={"ticker": str})
            df["target_date"] = df["target_date"].astype(str).str[:10]
            dst[int(tail)] = (df.pivot(index="target_date", columns="ticker", values="y_hat")
                              .reindex(index=dates, columns=D["codes"]).to_numpy(float))
    return out, outv


ARMS = {"第一折": [("MAGNET 主 arm", "tw50_betaF1nA2r1"), ("MAGNET 前版（beta 層）", "tw50_beta"),
                  ("MAGNET 單市場消融", "tw50_smktA"), ("Early fusion", "tw50chk_ef"), ("LSTM only", "tw50chk_lstm"),
                  ("HGT", "tw50_bl_hgt"), ("MEIG", "tw50_bl_meig"), ("Adv-ALSTM", "tw50_bl_adv_alstm"),
                  ("HATS", "tw50_bl_hats"), ("MAN-SF", "tw50_bl_man_sf")],
        "第二折": [("MAGNET 主 arm", "f2_best"), ("MAGNET 前版", "f2_base")]}


def block_k(D, fold):
    Y, Yv = D["y_test"], D["y_val"]
    cl = ridge(D, "US30+TW50", "zscore", False)
    kr = ridge(D, "US30+TW50", "rank", True)
    L, Lv = gnorm(cl["test"], msd(cl["val"])), gnorm(cl["val"], msd(cl["val"]))
    print(f"[K] 對照：疊上同一個線性通道（全域尺度，w 在 val 上以 0..2 選）；線性通道單獨 ICIR {summ(cl['test'], Y)['ICIR']:.4f}")
    w = best_w(lambda g: summ(gnorm(kr["val"], msd(kr["val"])) + g * Lv, Yv)["IC"], GRID2)
    LL = gnorm(kr["test"], msd(kr["val"])) + w * L
    print(f"    {'線性 + 線性（KTW+ 疊通道）':26s} ICIR {summ(LL, Y)['ICIR']:.4f}（w={w:.1f}）；"
          f"逐日預測相關 通道 vs KTW+ {dcorr(cl['test'], kr['test']):+.3f}")
    cache = {}
    for lab, arm in ARMS[fold]:
        A, Av = _arm(arm, D)
        s = sorted(A)
        if not s:
            continue
        Ave = np.nanmean([Av[k] for k in s], 0)
        w = best_w(lambda g: summ(gnorm(Ave, msd(Ave)) + g * Lv, Yv)["IC"], GRID2)
        i0 = np.mean([summ(A[k], Y)["ICIR"] for k in s])
        i1 = np.mean([summ(gnorm(A[k], msd(Av[k])) + w * L, Y)["ICIR"] for k in s])
        cache[arm] = (A, Av)
        print(f"    {lab:26s} n={len(s):2d} 與通道相關 {dcorr(np.nanmean([A[k] for k in s], 0), cl['test']):+.3f}  "
              f"單獨 ICIR {i0:.4f} -> 混合 {i1:.4f}（w={w:.1f}）")
    if fold == "第一折" and {"tw50_betaF1nA2r1", "tw50chk_ef"} <= set(cache):
        (M, Mv), (E, Ev) = cache["tw50_betaF1nA2r1"], cache["tw50chk_ef"]
        seeds = sorted(set(M) & set(E))
        Me, Ee = np.nanmean([Mv[k] for k in seeds], 0), np.nanmean([Ev[k] for k in seeds], 0)
        _, a, b = max((summ(gnorm(Me, msd(Me)) + a * Lv + b * gnorm(Ee, msd(Ee)), Yv)["IC"], a, b)
                      for a in GRID2 for b in GRID2)
        wm = best_w(lambda g: summ(gnorm(Me, msd(Me)) + g * Lv, Yv)["IC"], GRID2)
        r2 = np.array([summ(gnorm(M[k], msd(Mv[k])) + wm * L, Y)["ICIR"] for k in seeds])
        r3 = np.array([summ(gnorm(M[k], msd(Mv[k])) + a * L + b * gnorm(E[k], msd(Ev[k])), Y)["ICIR"] for k in seeds])
        print(f"    三方（線性 {a:.1f}、early fusion {b:.1f}）ICIR {r3.mean():.4f} vs 線性 + MAGNET {r2.mean():.4f}："
              f"差 {np.mean(r3 - r2):+.4f}（配對 p {stats.ttest_rel(r3, r2).pvalue:.4f}，{int((r3 > r2).sum())}/{len(seeds)}）；"
              f"MAGNET vs early fusion 逐日相關 {dcorr(np.nanmean([M[k] for k in seeds], 0), np.nanmean([E[k] for k in seeds], 0)):+.3f}")


# ---------------------------------------------------------------- N：檔數
def block_n(D, f):
    P, s = preds(f["p10b"], "test", D["d_test"], D["codes"])
    ser = summ(P[0], D["y_test"])["ser"]
    ser = ser[np.isfinite(ser)]
    tot = np.var(ser, ddof=1)
    noise = lambda n: np.mean((1 - ser ** 2) ** 2) / (n - 3)          # var(r) ≈ (1−ρ²)²/(n−3)
    true_v = max(tot - noise(50), 1e-9)
    print(f"[N] 檔數（P10b 種子 {s[0]}）：n=50 的抽樣雜訊佔 var(IC_t) {noise(50) / tot:.0%}；ICIR 隨檔數："
          + "，".join(f"n={'∞' if n > 9_999 else n} {ser.mean() / np.sqrt(true_v + noise(n)):.3f}"
                     for n in (50, 100, 150, 10_000)))


def main() -> None:
    ap = argparse.ArgumentParser(description="P12 之前的探索（proposal §63.1）")
    ap.add_argument("--blocks", nargs="*", default=list(BLOCKS), choices=list(BLOCKS))
    args = ap.parse_args()
    for fold, f in FOLDS.items():
        D = load_fold(f["cfg"], f["f2"])
        print(f"\n######## {fold}：測試窗 {D['d_test'][0]} ~ {D['d_test'][-1]}（{len(D['d_test'])} 天）")
        if "L" in args.blocks:
            block_l(D)
        if "M" in args.blocks:
            block_m(D, f)
        if "K" in args.blocks:
            block_k(D, fold)
        if "N" in args.blocks:
            block_n(D, f)


if __name__ == "__main__":
    main()
