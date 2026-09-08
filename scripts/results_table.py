"""
results_table.py — 產出跨方法的總結果表（docs/results_table.md）

收三類方法，全部在同一個 tw50 walk-forward test（246 天）上：
  1. 神經模型：本專案的 MAGNET 各版本 + 六個文獻 baseline，逐 seed 平均
  2. 線性/樹模型：ridge 階梯、factor_vs_graph、[24] 二部圖重現，單一預測檔
  3. 空模型：RC 常數對照

統計一律以目前最佳的 MAGNET（tw50_T1F3bnl1）為基準：
  - 神經 arm：取共同 seed，逐日 IC 序列先跨 seed 平均再配對檢定，
    另報跨 seed 的 Welch 與 Mann-Whitney（後者在 n<5 時無法達到 p<0.05）
  - 非神經：只有一條預測序列，僅報逐日配對

用法：
    .venv/bin/python scripts/results_table.py            # 印出並寫檔
    .venv/bin/python scripts/results_table.py --no-write
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 主結果 arm。2026-08-30 由 tw50_beta 換成 tw50_betaF1nA2r1
# （= beta 層 + F1 只留 log_return + 關掉台股層圖 A₂ + rank 損失權重 1.0）：
#   - 第一折 IC +0.1011 -> +0.1090、RankIC +0.0967 -> +0.1098，10/10 種子
#   - **第二折（train 0-902 / 評估窗 2023-12-20 ~ 2024-12-25，與第一折不重疊）
#     ΔIC +0.0308、ΔRankIC +0.0262，10/10 種子，且逐日檢定兩個指標都過
#     （p 0.0004 / 0.0036）** —— 本專案第一次有架構改動在逐日層級對自家基準顯著
#   - 換基準的依據是 proposal §36 事先登記的判準（§38 判定通過最高一檔）
#   - **不可寫「超越線性 baseline」**：第二折對 KTW+ 的 RankIC 是 −0.0115，
#     對 R2 ridge 恰好打平（+0.0000），IC 雖名目領先但無一顯著（§40）
#   - 須揭露的退步：過度離散再惡化，std 比 1.69 -> 2.59、MSE 0.00106 -> 0.00204
#   - 三個改動單獨都無效甚至有害（F1 +0.0030 ns、無A₂ IC −0.0050、
#     rank 1.0 單獨 −0.0003），是超可加的交互作用（§35）
BEST = "tw50_betaF1nA2r1"       # 主結果，所有比較的基準（見上）

# (顯示名稱, arm 或 glob, 類別, 備註)
NEURAL = [
    ("MAGNET 本版（F1 + 無A₂ + rank 1.0）", "tw50_betaF1nA2r1", "本專案",
     "三者缺一不可，見 §35"),
    ("　└ 同上，但保留台股層圖與 F3", "tw50_betaR1", "本專案", "只調 rank 權重"),
    ("　└ F1 + 無A₂（rank 0.5）", "tw50_betaF1nA2", "本專案", "未加 rank 權重"),
    ("　└ F1（單獨）", "tw50_betaF1", "本專案", "只換特徵"),
    ("MAGNET + beta 層（前版）", "tw50_beta", "本專案",
     "每檔一個 alpha_j / gamma_j"),
    ("　└ 同上，B 降秩 r=3", "tw50_betaLR3", "本專案", "B 參數 1500 -> 240，null"),
    ("　└ 同上，per-target 讀出", "tw50_betaPT", "本專案", "已證偽，−0.0093"),
    ("　└ 同上，多因子 k=3", "tw50_betaK3", "本專案", "已證偽，null"),
    ("MAGNET + 輸入正規化 + AdamW（前版）", "tw50_inbnw_noskip", "本專案",
     "F3, T=1, L=1, 無跳接"),
    ("　└ 同上，Adam", "tw50_inbn_noskip", "本專案", "分離 AdamW 的貢獻"),
    ("　└ 同上，再加拼接跳接", "tw50_inbn_skip", "本專案", "跳接是否仍有增量"),
    ("　└ 同上，F9", "tw50_inbnw_f9_noskip", "本專案", "特徵數是否仍無差異"),
    ("MAGNET + BN 拼接跳接（前版）", "tw50_T1F3bnl1", "本專案", "F3, T=1, L=1"),
    ("MAGNET 原始（9 特徵 T=20 L=2）", "tw50chk_wl0",  "本專案", "F9, T=20, L=2"),
    ("MAGNET F9 T=1 L=1",            "tw50_T1F9",     "本專案", "F9, T=1, L=1"),
    ("MAGNET F3（無跳接）",            "tw50_T1F3rb",   "本專案", "F3, T=1, L=1"),
    ("MAGNET + 稠密耦合 A",            "tw50_T1F9dense","本專案", "A[30,50] 可學"),
    ("Early fusion 對照",             "tw50chk_ef",    "本專案", "拼接後單一編碼器"),
    ("單市場消融（主結果 arm）", "tw50_smktA", "本專案",
     "disable_a12，跨市場全關"),
    ("單市場消融（前一版 arm）", "tw50_smktB", "本專案",
     "同上，台股側 F3 + 有圖"),
    ("LSTM only（無圖）",              "tw50chk_lstm",  "本專案",
     "另一架構，非對等對照"),
    ("HGT [14]",                      "tw50_bl_hgt",       "文獻", "未調參"),
    ("DeltaLag [13]",                 "tw50_bl_delta_lag", "文獻", "未調參，預測退化"),
    ("MEIG [1]",                      "tw50_bl_meig",      "文獻", "未調參"),
    ("Adv-ALSTM [10]",                "tw50_bl_adv_alstm", "文獻", "未調參"),
    ("HATS [12]",                     "tw50_bl_hats",      "文獻", "未調參"),
    ("MAN-SF [11]",                   "tw50_bl_man_sf",    "文獻", "未調參"),
]
NONNEURAL = [
    ("[24] 二部圖 ens-avg",  "*_bipartite_walkforward_t2_ens-avg", "文獻", "t 檢定篩邊 + 集成"),
    ("[24] 二部圖 LASSO",    "*_bipartite_walkforward_t2_LASSO",   "文獻", ""),
    ("KTW+（rank 目標）",     "*_fvg_KTWp",                          "線性", "美股+台股 80 維"),
    ("R2（30 檔美股報酬）",    "*_fvg_R2",                            "線性", "per-target ridge"),
    ("RC 常數對照",           "*_ridge_RC",                          "空模型", "0 特徵"),
]


# 預測檔來源。reeval = CPU 重評版（scripts/reeval_checkpoints.py 產生），
# recorded = run 當下寫的 CSV。含 GAT 的 arm 在 MPS 上的 scatter-add 不可重現，
# 同一份 best.pt 連評 5 次 test IC 全距 0.0169，所以本表預設走 reeval；
# 無重評檔的 arm（無 GAT，或本來就在 CPU 上評的新 run）自動回退到 recorded。
PREDICTIONS = "reeval"


def daily_series(run_dir: str):
    f = os.path.join(run_dir, "predictions", "test_predictions_reeval.csv")
    if PREDICTIONS != "reeval" or not os.path.exists(f):
        f = os.path.join(run_dir, "predictions", "test_predictions.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
    Y = df.pivot(index="target_date", columns="ticker", values="y").sort_index()
    Hn, Yn = H.to_numpy(), Y.to_numpy()
    ic = np.full(len(Yn), np.nan)
    ric = np.full(len(Yn), np.nan)
    for t in range(len(Yn)):
        a, b = Hn[t], Yn[t]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        ic[t] = np.corrcoef(a, b)[0, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = stats.spearmanr(a, b).statistic
        ric[t] = np.nan if np.isnan(v) else v
    return H.index.to_numpy(), ic, ric


def find_seeds(arm: str) -> dict[str, str]:
    out = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / "**" / f"*{arm}_s*"), recursive=True)):
        name = re.sub(r"^\d{8}_\d{4}_", "", os.path.basename(d))
        mm = re.match(rf"^{re.escape(arm)}_s(\d+)$", name)
        if mm and os.path.exists(os.path.join(d, "meta.json")):
            out[mm.group(1)] = d
    return out


def colmean(stack: np.ndarray) -> np.ndarray:
    out = np.full(stack.shape[1], np.nan)
    ok = ~np.isnan(stack).all(axis=0)
    out[ok] = np.nanmean(stack[:, ok], axis=0)
    return out


def neural_stats(arm: str):
    runs = find_seeds(arm)
    if not runs:
        return None
    ics, rics, per_ic, per_ric, params = [], [], [], [], None
    for s, d in sorted(runs.items(), key=lambda kv: int(kv[0])):
        tm = (json.load(open(os.path.join(d, "meta.json"))).get("test_metrics") or {})
        rv = os.path.join(d, "meta_reeval.json")
        if PREDICTIONS == "reeval" and os.path.exists(rv):
            tm = {**tm, **json.load(open(rv))["reevaluated"]}
        if tm.get("IC") is None:
            continue
        ser = daily_series(d)
        # RankIC 改由預測檔重算，不讀 meta.json。
        #
        # 2026-09-01：meta.json 的 RankIC 是用舊的 `argsort(argsort())` 取名次算的，
        # 對平手值給相異名次。實測 y（次日報酬）有 208/246 天存在平手——多半是
        # 恰好 0% 的股票——所以那個欄位一直帶著系統性偏差（proposal §44.7、D1）。
        # `daily_series` 用的是 scipy 的 spearmanr，平手取平均名次，是正確的。
        #
        # 影響：IC 逐位元不變（實測差 0.000000），RankIC 上移約 +0.0013 ~ +0.0017，
        # 各 arm 同向，差值幾乎不動（第二折對 KTW+ −0.0114 -> −0.0115），
        # 沒有任何結論翻轉。改完之後本表與 proposal 內文的 RankIC 才一致。
        per_ic.append(tm["IC"])
        per_ric.append(float(np.nanmean(ser[2])) if ser is not None else tm["RankIC"])
        if ser is not None:
            ics.append(ser[1]); rics.append(ser[2]); dates = ser[0]
        if params is None:
            params = tm.get("n_params")
    if not per_ic:
        return None
    return dict(seeds=sorted(runs, key=int), n=len(per_ic),
                ic=float(np.mean(per_ic)), ic_sd=float(np.std(per_ic, ddof=1)) if len(per_ic) > 1 else np.nan,
                ric=float(np.mean(per_ric)), ric_sd=float(np.std(per_ric, ddof=1)) if len(per_ric) > 1 else np.nan,
                per_ic=dict(zip(sorted(runs, key=int), per_ic)),
                per_ric=dict(zip(sorted(runs, key=int), per_ric)),
                daily_ic=colmean(np.vstack(ics)) if ics else None,
                daily_ric=colmean(np.vstack(rics)) if rics else None,
                dates=dates if ics else None)


def nonneural_stats(pat: str):
    ds = sorted(glob.glob(str(ROOT / "runs" / "**" / pat), recursive=True))
    if not ds:
        return None
    ser = daily_series(ds[-1])
    if ser is None:
        return None
    dates, ic, ric = ser
    return dict(n=1, ic=float(np.nanmean(ic)), ic_sd=np.nan,
                ric=float(np.nanmean(ric)), ric_sd=np.nan,
                daily_ic=ic, daily_ric=ric, dates=dates, seeds=None)


def paired_daily(base: dict, other: dict, key: str):
    """回傳 (差, HAC p)。差為正代表 base 較優。"""
    a, b = base["daily_" + key], other["daily_" + key]
    if a is None or b is None or len(a) != len(b):
        return np.nan, np.nan
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    n = len(d)
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    x = d - d.mean()
    var = float((x @ x) / n)
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float((x[k:] @ x[:-k]) / n)
    se = np.sqrt(max(var, 1e-24) / n)
    t = d.mean() / se
    return float(d.mean()), float(2 * stats.t.sf(abs(t), n - 1))


def across_seed(base: dict, other: dict, key: str):
    """同種子集合的跨 seed 檢定。回傳 (Welch p, Mann-Whitney p, n_common)。"""
    if base.get("seeds") is None or other.get("seeds") is None:
        return np.nan, np.nan, 0
    common = sorted(set(base["per_" + key]) & set(other["per_" + key]), key=int)
    if len(common) < 2:
        return np.nan, np.nan, len(common)
    a = np.array([base["per_" + key][s] for s in common])
    b = np.array([other["per_" + key][s] for s in common])
    _, pw = stats.ttest_ind(a, b, equal_var=False)
    _, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(pw), float(pu), len(common)


def fmt(v, nd=4, signed=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"



def _fold2_rows():
    """第二折的對照列。找不到必要的 run 時回傳 None。"""
    import glob as _g
    CUT = "2024-12-25"

    def _load(pat):
        ds = []
        for f in sorted(_g.glob(pat, recursive=True)):
            df = pd.read_csv(f)
            o = {}
            for d, g in df.groupby("target_date"):
                d = str(d)[:10]
                if d > CUT or g.y.std() == 0 or g.y_hat.std() == 0:
                    continue
                o[d] = (np.corrcoef(g.y_hat, g.y)[0, 1],
                        stats.spearmanr(g.y_hat, g.y).statistic)
            if o:
                ds.append(o)
        if not ds:
            return None
        k = sorted(set.intersection(*[set(x) for x in ds]))
        return k, np.array([[np.mean([x[d][i] for x in ds]) for d in k]
                            for i in (0, 1)])

    def _hac(d):
        n = len(d); L = int(4 * (n / 100) ** (2 / 9)); s2 = np.var(d, ddof=1)
        for l in range(1, L + 1):
            s2 += 2 * (1 - l / (L + 1)) * np.cov(d[l:], d[:-l], ddof=1)[0, 1]
        tt = d.mean() / (s2 / n) ** 0.5
        return 2 * (1 - stats.norm.cdf(abs(tt)))

    b = _load(str(ROOT / "runs/**/*f2_best_s*/predictions/test_predictions.csv"))
    if b is None:
        return None
    kb, BESTV = b
    rows = [("**MAGNET 本版（F1 + 無A₂ + rank 1.0）**", None)]
    for lab, pat in (("KTW+（最高標）", "runs_f2/*fvg_KTWp/predictions/*.csv"),
                     ("[24] 二部圖 LASSO", "runs_f2/*bipartite*t2_LASSO/predictions/*.csv"),
                     ("R2 per-target ridge", "runs_f2/*ridge_R2/predictions/*.csv"),
                     ("[24] 二部圖 ens-avg", "runs_f2/*bipartite*t2_ens-avg/predictions/*.csv"),
                     ("TW+（美股+台股 80 維）", "runs_f2/*fvg_TWp/predictions/*.csv"),
                     ("MAGNET 基準（自家前版）", "runs/**/*f2_base_s*/predictions/test_predictions.csv"),
                     ("RC 常數對照", "runs_f2/*ridge_RC/predictions/*.csv")):
        rows.append((lab, _load(str(ROOT / pat))))
    out = []
    for lab, r in rows:
        if r is None and lab.startswith("**"):
            out.append(f"| {lab} | {BESTV[0].mean():+.4f} | {BESTV[1].mean():+.4f} "
                       f"| — | — | — | — |")
            continue
        if r is None:
            continue
        kk, V = r
        ii = [kk.index(d) for d in kk if d in kb]
        jj = [kb.index(d) for d in kk if d in kb]
        cells = []
        for i in (0, 1):
            dd = BESTV[i][jj] - V[i][ii]
            cells += [f"{dd.mean():+.4f}", f"{_hac(dd):.4f}"]
        out.append(f"| {lab} | {V[0][ii].mean():+.4f} | {V[1][ii].mean():+.4f} "
                   f"| {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    return out


def _ens_block():
    """§43 的種子集成與等權混合。缺 run 時回傳 None。"""
    import glob as _g

    def _pred(d, cut=None):
        f = os.path.join(d, "predictions", "test_predictions_reeval.csv")
        if PREDICTIONS != "reeval" or not os.path.exists(f):
            f = os.path.join(d, "predictions", "test_predictions.csv")
        if not os.path.exists(f):
            return None
        df = pd.read_csv(f)
        df["target_date"] = df.target_date.astype(str).str[:10]
        return df[df.target_date <= cut] if cut else df

    def _metrics(df):
        """回傳 {date: (IC, RankIC, 離散比)}。"""
        o = {}
        for d, g in df.groupby("target_date"):
            if g.y.std() == 0 or g.y_hat.std() == 0:
                continue
            o[d] = (np.corrcoef(g.y_hat, g.y)[0, 1],
                    stats.spearmanr(g.y_hat, g.y).statistic,
                    g.y_hat.std() / g.y.std())
        return o

    def _per_seed(pat, cut=None):
        """(1) 先算每顆種子的逐日指標，再跨種子平均。"""
        ds = [x for x in (_pred(d, cut)
                          for d in sorted(_g.glob(str(ROOT / pat), recursive=True)))
              if x is not None]
        if not ds:
            return None
        ms = [_metrics(x) for x in ds]
        k = sorted(set.intersection(*[set(m) for m in ms]))
        return k, np.array([[np.mean([m[d][i] for m in ms]) for d in k]
                            for i in (0, 1, 2)]), len(ds)

    def _ens(pat, cut=None):
        """(2) 先平均全部種子的預測，再算逐日指標。"""
        ds = [x for x in (_pred(d, cut)
                          for d in sorted(_g.glob(str(ROOT / pat), recursive=True)))
              if x is not None]
        if not ds:
            return None
        df = (pd.concat(ds).groupby(["target_date", "ticker"])
              .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
        m = _metrics(df)
        k = sorted(m)
        return k, np.array([[m[d][i] for d in k] for i in (0, 1, 2)]), df

    def _hac(d):
        n = len(d)
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        x = d - d.mean()
        var = float((x @ x) / n)
        for j in range(1, lag + 1):
            var += 2.0 * (1.0 - j / (lag + 1.0)) * float((x[j:] @ x[:-j]) / n)
        return float(2 * stats.t.sf(abs(d.mean() / np.sqrt(max(var, 1e-24) / n)), n - 1))

    def _cmp(mine, other):
        km, M = mine[0], mine[1]
        ko, O = other[0], other[1]
        k = sorted(set(km) & set(ko))
        ii = [km.index(d) for d in k]
        jj = [ko.index(d) for d in k]
        return [(M[i][ii] - O[i][jj]).mean() for i in (0, 1)], \
               [_hac(M[i][ii] - O[i][jj]) for i in (0, 1)]

    FOLDS = (("第一折", f"runs/**/*{BEST}_s*", None,
              "runs/linear/*_fvg_KTWp/predictions/*.csv"),
             ("第二折", "runs/**/*f2_best_s*", "2024-12-25",
              "runs_f2/*_fvg_KTWp/predictions/*.csv"))

    out, ok = [], False
    out.append("### (A) 種子集成：先平均預測，再算 IC")
    out.append("")
    out.append("| 折 | n 種子 | 聚合 | IC | RankIC | 離散比 std(y_hat)/std(y) |")
    out.append("|---|---:|---|---:|---:|---:|")
    ens_cache = {}
    for lab, pat, cut, _ in FOLDS:
        ps = _per_seed(pat, cut)
        en = _ens(pat, cut)
        if ps is None or en is None:
            continue
        ok = True
        ens_cache[lab] = (en, cut)
        out.append(f"| {lab} | {ps[2]} | 先算 IC 再平均 | {ps[1][0].mean():+.4f} "
                   f"| {ps[1][1].mean():+.4f} | {ps[1][2].mean():.2f} |")
        out.append(f"| {lab} | {ps[2]} | **先平均預測再算 IC** "
                   f"| **{en[1][0].mean():+.4f}** | **{en[1][1].mean():+.4f}** "
                   f"| **{en[1][2].mean():.2f}** |")
        out.append(f"| {lab} | | 增益 | {en[1][0].mean() - ps[1][0].mean():+.4f} "
                   f"| {en[1][1].mean() - ps[1][1].mean():+.4f} "
                   f"| {en[1][2].mean() - ps[1][2].mean():+.2f} |")
    if not ok:
        return None

    BLS = (("KTW+（最高標）", "*_fvg_KTWp"), ("[24] 二部圖 LASSO", "*bipartite*t2_LASSO"),
           ("R2 per-target ridge", "*_ridge_R2"), ("[24] 二部圖 ens-avg", "*bipartite*t2_ens-avg"),
           ("RC 常數對照", "*_ridge_RC"))
    out += ["", "### (B) 集成後對線性 baseline 的逐日檢定", "",
            "| 折 | 對照 | 其 IC | 其 RankIC | dIC | 逐日 p | dRankIC | 逐日 p |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for lab, _, cut, _ in FOLDS:
        if lab not in ens_cache:
            continue
        en, cut = ens_cache[lab]
        root = "runs/linear" if lab == "第一折" else "runs_f2"
        for bl, bp in BLS:
            b = _per_seed(f"{root}/{bp}", cut)
            if b is None:
                continue
            d, pv = _cmp(en, b)
            out.append(f"| {lab} | {bl} | {b[1][0].mean():+.4f} | {b[1][1].mean():+.4f} "
                       f"| {d[0]:+.4f} | {pv[0]:.4f} | {d[1]:+.4f} | {pv[1]:.4f} |")

    out += ["", "### (C) 與 KTW+ 等權混合（w=0.5，逐日橫截面 z 分數，未調參）", "",
            "| 折 | 方法 | IC | RankIC | dIC vs KTW+ | 逐日 p | dRankIC | 逐日 p |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for lab, _, cut, kp in FOLDS:
        if lab not in ens_cache:
            continue
        en, cut = ens_cache[lab]
        kf = sorted(_g.glob(str(ROOT / kp)))
        if not kf:
            continue
        K = pd.read_csv(kf[-1])
        K["target_date"] = K.target_date.astype(str).str[:10]
        if cut:
            K = K[K.target_date <= cut]
        j = en[2].merge(K[["target_date", "ticker", "y_hat"]],
                        on=["target_date", "ticker"], suffixes=("_m", "_b"))

        def _z(g, c):
            v = g[c].to_numpy()
            sd = v.std()
            return (v - v.mean()) / sd if sd > 1e-12 else v * 0.0

        for c, src in (("zm", "y_hat_m"), ("zb", "y_hat_b")):
            j[c] = j.groupby("target_date", group_keys=False).apply(
                lambda g: pd.Series(_z(g, src), index=g.index), include_groups=False)
        rows = [("純 KTW+", 0.0), ("**等權混合 w=0.5**", 0.5), ("純 MAGNET（集成）", 1.0)]
        base = None
        for nm, w in rows:
            jj = j.assign(y_hat=w * j.zm + (1 - w) * j.zb)
            m = _metrics(jj)
            k = sorted(m)
            V = np.array([[m[d][i] for d in k] for i in (0, 1)])
            if w == 0.0:
                base = V
                out.append(f"| {lab} | {nm} | {V[0].mean():+.4f} | {V[1].mean():+.4f} "
                           f"| — | — | — | — |")
                continue
            d0, d1 = V[0] - base[0], V[1] - base[1]
            out.append(f"| {lab} | {nm} | {V[0].mean():+.4f} | {V[1].mean():+.4f} "
                       f"| {d0.mean():+.4f} | {_hac(d0):.4f} "
                       f"| {d1.mean():+.4f} | {_hac(d1):.4f} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="產出跨方法總結果表")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--predictions", choices=["reeval", "recorded"],
                    default="reeval",
                    help="reeval=CPU 重評（預設，可重現）；"
                         "recorded=run 當下寫的 CSV（含 GAT 的 arm 走 MPS，不可重現）")
    args = ap.parse_args()
    global PREDICTIONS
    PREDICTIONS = args.predictions

    rows = []
    for label, arm, cat, note in NEURAL:
        st = neural_stats(arm)
        if st is None:
            print(f"[skip] {label}（找不到 {arm}）"); continue
        rows.append((label, cat, note, st, arm))
    for label, pat, cat, note in NONNEURAL:
        st = nonneural_stats(pat)
        if st is None:
            print(f"[skip] {label}（找不到 {pat}）"); continue
        rows.append((label, cat, note, st, pat))

    base = next(st for _, _, _, st, arm in rows if arm == BEST)
    out = []
    out.append("# 跨方法結果表")
    out.append("")
    out.append("由 `scripts/results_table.py` 產生。universe = tw50（US 30 / TW 50，"
               "配對 7 檔），**第一折** walk-forward test 246 天（2024-12-26 ~ 2025-12-30）。第二折的獨立驗證見下方專節——**單折排名會翻轉，兩節必須一起看**。")
    out.append("")
    out.append(f"統計基準 = **{BEST}**（本專案目前最佳）。差為正代表基準較優。")
    out.append("")
    out.append("| 方法 | 類別 | 設定 | n | test IC | sd | RankIC | sd | dIC | 逐日 p | 跨種子 Welch p | MW p |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, cat, note, st, arm in sorted(rows, key=lambda r: -r[3]["ic"]):
        d_ic, p_pd = paired_daily(base, st, "ic")
        pw, pu, _ = across_seed(base, st, "ic")
        is_base = (arm == BEST)
        star = " **" if is_base else " "
        out.append(
            f"|{star}{label}{star.strip()} | {cat} | {note} | {st['n']} | "
            f"{fmt(st['ic'])} | {fmt(st['ic_sd'], signed=False)} | "
            f"{fmt(st['ric'])} | {fmt(st['ric_sd'], signed=False)} | "
            f"{'—' if is_base else fmt(d_ic)} | "
            f"{'—' if is_base else fmt(p_pd, signed=False)} | "
            f"{fmt(pw, signed=False)} | {fmt(pu, signed=False)} |")
    # ── 第二折驗證（proposal §36 事先登記、§38/§40 判定）──────────────
    out.append("")
    out.append("## 第二折驗證（獨立測試期）")
    out.append("")
    out.append("切法：train 0-902 / val 903-1149，評估窗 **2023-12-20 ~ 2024-12-25**"
               "（246 天），與第一折的測試期完全不重疊，且模型從未在其上做過任何選擇。"
               "所有線性 baseline 皆以第二折切法重新擬合，非沿用第一折的版本。")
    out.append("")
    f2 = _fold2_rows()
    if f2 is None:
        out.append("_（第二折的 run 或 baseline 尚未齊備）_")
    else:
        out.append("| 方法 | test IC | RankIC | dIC | 逐日 p | dRankIC | 逐日 p |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in f2:
            out.append(r)
    out.append("")
    out.append("**判定（§38）**：對自家基準 ΔIC +0.0308 / ΔRankIC +0.0262，"
               "10/10 種子，**逐日檢定兩個指標都過**（0.0004 / 0.0036）——"
               "本專案第一次有架構改動在逐日層級對自家基準顯著。")
    out.append("")
    out.append("**但（§40）**：對線性 baseline **不成立**。IC 名目領先每一個"
               "（+0.0055 ~ +0.0203）卻無一顯著；RankIC 與 R2 ridge 恰好打平"
               "（+0.0000），並輸給 KTW+（−0.0115）。"
               "**兩折合看，相對最佳線性 baseline 是打平，不是超越。**")
    out.append("")
    out.append("## 種子集成與混合（proposal §43）")
    out.append("")
    out.append("以下三塊**不需要重訓任何模型**，全部由既有預測檔重算。"
               "(A) 改變聚合方式，(B) 是它對 baseline 的後果，"
               "(C) 是與最強線性 baseline 的等權混合。")
    out.append("")
    eb = _ens_block()
    if eb is None:
        out.append("_（集成所需的預測檔尚未齊備）_")
    else:
        out += eb
    out.append("")
    out.append("## 讀表注意")
    out.append("")
    out.append("- **n 是種子數。** sigma_seed = 0.0062，n=3 的最小可偵測差異為 0.0188、"
               "n=10 為 0.0082。n=3 的比較若差異小於 0.019，「不顯著」不構成證據。")
    out.append("- **Mann-Whitney 在 n=3 的雙尾 p 下限是 0.10**，永遠無法達到 0.05；"
               "n=1（非神經方法）則無法計算。跨種子欄為「—」代表該方法只有單一預測序列。")
    out.append("- **逐日 p 用 Newey-West HAC 修正自相關**，以「日」為重複單位、種子視為固定，"
               "敏感但不外推到新種子。兩欄應一起看。")
    out.append("- **六個文獻 baseline 各只跑一組預設超參，MAGNET 跑了約 50 組設定。**"
               "這是目前最大的公平性缺口，比較結果須據此保留。")
    out.append("- **本表只報 IC / RankIC，兩者都是相關係數、對預測的尺度不敏感。**"
               "主結果 arm 的預測是**過度離散**的——逐日橫截面 "
               "std(y_hat)/std(y) = **2.59**（beta 層前版 1.69、更早的版本 0.47 是收縮）。"
               "MSE 隨之由 0.00106 升到 **0.00204**，是最早版本的 4.5 倍。"
               "排序能力的提升是真的，但任何報 MSE / R² 或做組合回測的地方"
               "都必須先處理這一點。")
    out.append("- **不可寫「超越線性 baseline」。** 第一折對 KTW+ 名目領先"
               "（IC +0.0013 / RankIC +0.0021）但逐日檢定全部不顯著；"
               "**第二折 RankIC 反而輸給 KTW+（−0.0115）、與 R2 ridge 恰好打平"
               "（+0.0000）**。兩折合看是打平。可以宣稱的是"
               "**對自家前一版的架構改善**（第二折逐日 p 0.0004 / 0.0036）。"
               "**此條指的是純 MAGNET；種子集成後仍成立**（對 KTW+ 兩折兩指標皆不顯著）。"
               "唯一在兩折上都顯著超越 KTW+ 的是 §43 (C) 的**等權混合**，"
               "但那是混合模型的主張，不是本架構單獨的主張。")
    out.append("- **單折排名會翻轉。** 第一折的優勢集中在測試期前半"
               "（dIC 前半 +0.0300、後半 −0.0141），第二折則相反（後半更好）。"
               "任何只根據單一測試期的排名都不可靠——這是本表最重要的保留條款。")
    out.append("- **本版的三個改動單獨都無效甚至有害**"
               "（F1 +0.0030 ns、關 A₂ 的 IC −0.0050、rank 1.0 單獨 −0.0003），"
               "合起來才是 +0.0079 / +0.0131（第一折）。這是超可加的交互作用，"
               "機制未確認，列為 open observation。")
    out.append("- **單市場消融是本表效果量最大的一格，也是「多層圖值不值得」的直接答案。**"
               "`disable_a12` 把 h_L1 在進 fusion 前零化，**參數量 48,548 與完整版完全相同**"
               "（美股側與耦合的 12,256 個參數梯度實測恰為 0），"
               "所以差異可以完全歸因到跨市場資訊。"
               "淨值：主結果 arm **ΔIC +0.1004**（10/10 種子、逐日 HAC p 3.3e-06）、"
               "前一版 arm +0.0961（10/10、3.2e-07）——"
               "**佔模型表現的 92%，切掉後低於常數對照。** 詳見 proposal §41。")
    out.append("- **`LSTM only` 不能當單市場對照。** 它是另一個架構（無 GAT、無融合閘門、"
               "無 beta 層），差異裡混了「沒有跨市場」與「少三個模組」。"
               "要回答教授建議 1，用的是上面兩個 `disable_a12` 的消融 arm。")
    out.append("- **兩個單市場 arm 的方向是反的**：台股側資訊較「完整」的 B"
               "（3 特徵 + 2,200 條邊的台股圖）RankIC 反而比 A（1 特徵、圖只剩 self-loop）"
               "低 0.0292，10 顆種子無一例外。理由見 §37——"
               "RSI/BB 的逐日自相關 0.85–0.91（只能產生近乎不變的排序）、"
               "台股圖對相關股票做平滑（抹掉排序唯一需要的橫截面差異）。"
               "有跨市場訊號時這兩項的傷害被蓋過去，切掉後才顯現。")
    out.append("- 單市場的數字**不可解讀成「台股資料沒有預測力」**：T_history=1 是在"
               "「有跨市場資訊」的前提下選的，沒有它時只看前一天本來就極難預測。"
               "佐證：六個文獻的單市場 baseline 也全部落在 −0.0053 ~ +0.0113。")
    out.append("- **§43 (A) 的兩個數字是不同的估計對象，不是同一個東西的兩種算法。**"
               "「先算 IC 再平均」= 隨機抽一個訓練好的模型的期望表現；"
               "「先平均預測再算 IC」= 實際部署那套系統（跑 10 個模型取平均）的表現。"
               "後者較高是因為種子雜訊互相抵消——實測種子兩兩預測相關 0.819，"
               "即每顆種子的橫截面預測有 **18.1%** 是種子特異雜訊"
               "（變異數分解獨立給出 17.7%）。理論 IC(S) = IC(1)·sqrt(S·SNR/(1+S·SNR))，"
               "SNR=4.54，預測 S=1->10 增益 +0.0101，實測 +0.0088。"
               "**論文兩個都要報。**")
    out.append("- **集成不是挑最好的種子。** 事後每天挑最佳種子可得 RankIC +0.2161，"
               "那是 cherry-picking、不可實現；集成用的是全部 10 顆、事先固定的規則，"
               "得到 +0.1202。另：邊際報酬在 **k=5 就飽和**"
               "（k=1 +0.1081、k=5 +0.1154、k=10 +0.1163、k=inf 的理論上限 +0.1204），"
               "種子加到 30 顆不值得。")
    out.append("- **集成必須聲明它用了 10 倍訓練算力，且不是對等的算力比較**——"
               "線性/樹 baseline 是凸問題的確定性解，沒有種子雜訊可平均。"
               "正當性有二：集成是隨機方法部署時的標準作法；"
               "我們自己的 baseline 集裡 [24] 就有 ens-avg 變體。")
    out.append("- **(B) 集成後仍然不能寫「超越所有 baseline」。** 對 KTW+ 兩折兩指標"
               "全部不顯著；表中 p 0.032 / 0.044 / 0.046 **通不過 Holm**"
               "（5 個對照時門檻 0.010）。集成改變的是：第二折 RankIC 對 KTW+ "
               "由 **−0.0115（輸）變成 +0.0006（平）**，且 20 個比較"
               "（2 折 x 5 對照 x 2 指標）**第一次全部同號為正**。")
    out.append("- **(C) 等權混合是目前唯一在兩折上都顯著超越最高標的設定**"
               "（四個檢定 p 0.0233 / 0.0041 / 0.0002 / 0.0177 全部 < 0.05，"
               "且兩折 ΔIC 都在逐日 MDE +0.0156 之上）。"
               "**w=0.5 是事先可指定的等權，沒有在測試集上調參**"
               "（附帶事實：兩折最佳 w 落在 0.55 與 0.6，等權接近最優）。"
               "機制：MAGNET 與 KTW+ 的逐日預測相關只有 **+0.472 / +0.490**，"
               "一半以上的橫截面資訊不重疊——這與 §37.2「優勢集中在最安靜的 20% 日子」一致。")
    out.append("- **(C) 的代價：主張形式改變。** 由「MAGNET 比線性準」變成"
               "**「MAGNET 提供線性模型抓不到的增量資訊」**。"
               "後者是可量測的科學陳述（相關 0.48），但**不是**「我們的架構單獨最好」。"
               "純 MAGNET（集成）對 KTW+ 仍然不顯著。")
    out.append("- **`rank_normalize`（訓練目標尺度不變化）已證偽，未列為表列 arm。**"
               "它的 IC 在多數種子上**無定義**——第一折 5/10、第二折 3/10 顆種子的預測"
               "塌縮成常數橫截面（83~100% 的測試日），對照 arm 是 0/10、0/10。"
               "只計未塌縮的種子，ΔIC 仍是 **−0.0724 / −0.1099**（同號 0/5、0/7）。"
               "機制是正規化後的 pairwise 損失在完全塌縮時恰為 ln(2)=0.6931，"
               "而排序不夠好時攤開來的損失是 0.9344——**塌縮是更低的損失狀態**。"
               "完整分析見 proposal §44，論文化素材見 §45。")
    out.append("- **RankIC 的平手處理已於 2026-09-01 修正（D1）。** 舊的 "
               "`_spearman_corr` 用 `argsort(argsort(·))` 取名次，對平手值給相異名次；"
               "實測 y 有 **208/246 天存在平手**（多半是恰好 0% 的股票），"
               "所以 `meta.json` 的 RankIC 一直帶系統性偏差，常數輸入更會回傳"
               "「ticker 順序 vs 真實報酬」的相關而非 NaN。"
               "現已改用平均名次，且本表的 RankIC 一律由預測檔重算。"
               "**影響：IC 逐位元不變，RankIC 上移約 +0.0013 ~ +0.0017、各 arm 同向，"
               "差值與所有 p 值幾乎不動（第二折對 KTW+ −0.0114 -> −0.0115），"
               "沒有結論翻轉。** 詳見 proposal §44.7 與 §54。"
               "掃過全部 631 個預測檔：**主結果 arm、beta 各版本、兩折的 f2_best / f2_base、"
               "全部線性 baseline 的塌縮天數皆為 0**，本表主要數字不受影響。"
               "兩個例外：`tw50_smktA` 每顆種子 1/246 天（對平均影響 < 0.0005）、"
               "`tw50_bl_delta_lag` 27/246 天（**其 RankIC 有 11% 的天數是 ticker 順序捏造的**）。")
    out.append("- DeltaLag 的預測退化（多日全 50 檔近乎同值），其數字不可信，待修。")
    out.append("- 線性/樹模型（[24]、KTW+、R2、RC）無隨機種子，sd 欄為「—」。")
    txt = "\n".join(out)
    print(txt)
    if not args.no_write:
        p = ROOT / "docs" / "results_table.md"
        p.write_text(txt + "\n")
        print(f"\n-> 已寫入 {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
