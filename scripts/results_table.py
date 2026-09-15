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
#   - 三個改動是**兩兩協同，沒有三階交互作用**（§35.3，2026-09-15 補完 2^3）。
#     主效應 F1 +0.0050 p=0.026、rank1.0 +0.0037 p=0.016、無A₂ −0.0006 ns
BEST = "tw50_betaF1nA2r1"       # 主結果，所有比較的基準（見上）

# (顯示名稱, arm 或 glob, 類別, 備註)
NEURAL = [
    # 以下六格 + tw50_beta + tw50_beta_g_empty_l2 構成 §35.3 的 2^3 完整設計
    # （A=F1 / B=無A₂ / C=rank1.0）。原本的備註「三者缺一不可」已被 2026-09-15
    # 的分解證偽：三階項是零，F1 與 rank1.0 各有顯著主效應。
    ("MAGNET 本版（F1 + 無A₂ + rank 1.0）", "tw50_betaF1nA2r1", "本專案",
     "2^3 的 (1,1,1)，見 §35.3"),
    ("　└ F1 + rank 1.0（保留 A₂）", "tw50_betaF1r1", "本專案",
     "(1,0,1)，2^3 補格"),
    ("　└ 同上，但保留台股層圖與 F3", "tw50_betaR1", "本專案",
     "(0,0,1)，只調 rank 權重"),
    ("　└ 無A₂ + rank 1.0（F3）", "tw50_betanA2r1", "本專案",
     "(0,1,1)，2^3 補格"),
    ("　└ F1 + 無A₂（rank 0.5）", "tw50_betaF1nA2", "本專案",
     "(1,1,0)，未加 rank 權重"),
    ("　└ F1（單獨）", "tw50_betaF1", "本專案", "(1,0,0)，只換特徵"),
    ("MAGNET + beta 層（前版）", "tw50_beta", "本專案",
     "(0,0,0) 基準，每檔一個 alpha_j / gamma_j"),
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


TOPK = 10       # 與 scripts/portfolio_readout.py 的 TOPK_DEFAULT 同值。
                # 這是「讀法」的選擇不是模型超參，故不從 base.yaml 讀。
ANN = float(np.sqrt(252.0))

# 砍掉的兩欄是純換算，留著只是同一個數字換單位：
#   年化(簡單) = 日均報酬 x 252      年化波動 = 日 sd x sqrt(252)
# 兩者都能從留下的欄位還原，Sharpe 則是新資訊（兩者之比）。
PF_HEAD = ("| 日均報酬 ↑ | 日 sd ↓ | 年化(幾何) ↑ "
           "| Sharpe ↑ | MDD(複利) ↓ | MDD(加總) ↓ ")
PF_SEP = "|---:|---:|---:|---:|---:|---:"
PF_BLANK = "| — | — | — | — | — | — "


def topk_ret(p_hat, y, k: int = TOPK) -> float:
    """Top-k 多空、等權、金額中性的當日報酬。

    權重是 ±1/(2k)，所以 sum|w| = 1（總曝險 1、淨曝險 0）；
    因此 (多腿均值 − 空腿均值) 要再除以 2 才是「單位本金」的報酬。
    不除等於偷偷假設 2 倍槓桿。
    """
    a = np.asarray(p_hat, float)
    b = np.asarray(y, float)
    o = np.argsort(-a)
    return float((b[o[:k]].mean() - b[o[-k:]].mean()) / 2.0)


def _mdd(curve) -> float:
    peak = np.maximum.accumulate(curve)
    return float(((peak - curve) / peak).max())


def portfolio_stats(series):
    """一批（逐 seed 的）Top-K 日報酬序列 -> 八個組合指標。

    逐 seed 算完再平均，與 ICIR 同一個慣例。

    **y 是 log return**（`src/dataset/pipeline.py:659` 的
    `log(Close/Close.shift(1))`），所以複利不能用 (1+r) 連乘：

        年化(幾何) = exp(mean(r) x 252) − 1
        MDD(複利)  的權益曲線 = exp(cumsum(r))

    MDD(加總) 用 1 + cumsum(r)，即獲利不滾入的固定名目本金——
    金額中性多空每天重設回同樣的名目曝險，這個讀法才對應實際操作。

    已知近似：對 log return 取橫截面平均**不等於**等權組合的報酬
    （後者是簡單報酬的算術平均）。實測差 +0.14 bp/日 = +0.36 pp/年，
    不影響 Sharpe，也不影響 §60.4 的打平成本（兩種算法都是 33 bp）。

    **這不是回測。** 無成本、無衝擊、無流動性限制、每日全額換手、
    可完全放空。用途是把 IC 翻譯成組合單位，見 proposal §60.6。
    """
    rows = []
    for r in series:
        r = np.asarray(r, float)
        r = r[~np.isnan(r)]
        if r.size < 2:
            continue
        mu, sd = float(r.mean()), float(r.std(ddof=1))
        rows.append((mu, sd, mu * 252.0, float(np.exp(mu * 252.0) - 1.0),
                     sd * ANN, (mu / sd * ANN) if sd > 1e-15 else np.nan,
                     _mdd(np.exp(np.cumsum(r))), _mdd(1.0 + np.cumsum(r))))
    if not rows:
        return None
    m = np.array(rows, float).mean(axis=0)
    return dict(zip(("r_mu", "r_sd", "ann_s", "ann_g", "vol",
                     "sharpe", "mdd_c", "mdd_a"), m.tolist()))


def pf_cells(pf) -> str:
    if pf is None:
        return PF_BLANK
    return (f"| {pf['r_mu'] * 100:+.4f}% | {pf['r_sd'] * 100:.4f}% "
            f"| {pf['ann_g'] * 100:+.1f}% | {pf['sharpe']:.2f} "
            f"| {pf['mdd_c'] * 100:.2f}% | {pf['mdd_a'] * 100:.2f}% ")


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
    rk = np.full(len(Yn), np.nan)          # Top-K 多空的日報酬（組合欄用）
    for t in range(len(Yn)):
        a, b = Hn[t], Yn[t]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        ic[t] = np.corrcoef(a, b)[0, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = stats.spearmanr(a, b).statistic
        ric[t] = np.nan if np.isnan(v) else v
        rk[t] = topk_ret(a, b)
    return H.index.to_numpy(), ic, ric, rk


def icir(arr) -> float:
    """ICIR = mean(逐日 IC) / sd(逐日 IC)。

    與 `src/train/metrics.py:176` 同一個定義（ddof=1），只是多了 nan-aware，
    因為 `daily_series` 會把 std(y)=0 的日子標成 NaN（第一折有 1 天，
    50 檔報酬完全相同）。

    為什麼要報這個：§57.8 的恆等式是 r_t = IC_t x sigma_t，所以
    **IC 只決定組合報酬的分子**。若 sigma_t 為常數，
    Sharpe = ICIR x sqrt(252)——與 Sharpe 對應的是 ICIR，不是 IC。
    實測（proposal §60）：本專案最佳 arm 的 IC 高於 KTW+，ICIR 卻較低
    （0.4368 vs 0.4810），Top-10 多空的 Sharpe 也較低（5.86 vs 6.53）。
    """
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size < 2:
        return float("nan")
    sd = a.std(ddof=1)
    return float("nan") if sd < 1e-12 else float(a.mean() / sd)


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
    ics, rics, rks = [], [], []
    per_ic, per_ric, per_icir, params = [], [], [], None
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
        # ICIR 逐 seed 算完再平均，不是拿跨 seed 平均後的日序列去算——
        # 後者是「集成」的 ICIR，系統性偏高（§43 的 A 區塊在講同一件事）。
        per_icir.append(icir(ser[1]) if ser is not None
                        else (tm.get("ICIR") if tm.get("ICIR") is not None else np.nan))
        if ser is not None:
            ics.append(ser[1]); rics.append(ser[2]); rks.append(ser[3])
            dates = ser[0]
        if params is None:
            params = tm.get("n_params")
    if not per_ic:
        return None
    return dict(seeds=sorted(runs, key=int), n=len(per_ic),
                ic=float(np.mean(per_ic)), ic_sd=float(np.std(per_ic, ddof=1)) if len(per_ic) > 1 else np.nan,
                ric=float(np.mean(per_ric)), ric_sd=float(np.std(per_ric, ddof=1)) if len(per_ric) > 1 else np.nan,
                icir=float(np.nanmean(per_icir)) if per_icir else np.nan,
                icir_sd=(float(np.nanstd(per_icir, ddof=1))
                         if sum(~np.isnan(per_icir)) > 1 else np.nan),
                per_ic=dict(zip(sorted(runs, key=int), per_ic)),
                per_ric=dict(zip(sorted(runs, key=int), per_ric)),
                daily_ic=colmean(np.vstack(ics)) if ics else None,
                daily_ric=colmean(np.vstack(rics)) if rics else None,
                pf=portfolio_stats(rks) if rks else None,
                dates=dates if ics else None)


def nonneural_stats(pat: str):
    ds = sorted(glob.glob(str(ROOT / "runs" / "**" / pat), recursive=True))
    if not ds:
        return None
    ser = daily_series(ds[-1])
    if ser is None:
        return None
    dates, ic, ric, rk = ser
    return dict(n=1, ic=float(np.nanmean(ic)), ic_sd=np.nan,
                ric=float(np.nanmean(ric)), ric_sd=np.nan,
                icir=icir(ic), icir_sd=np.nan,
                daily_ic=ic, daily_ric=ric, pf=portfolio_stats([rk]),
                dates=dates, seeds=None)


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


def pm(v, sd, nd=4, signed=True) -> str:
    """把「值」與「跨 seed sd」併成一格 `+0.1090 ± 0.0051`。

    n=1 的確定性 baseline（Ridge/LASSO 等）沒有 seed 變異，sd 為 NaN，
    這時只印值——注意那**不代表它沒有不確定性**，只代表它的不確定性
    來自資料而不是種子，那一側由逐日 HAC 檢定負責（見「讀表注意」）。
    """
    a = fmt(v, nd, signed)
    if sd is None or (isinstance(sd, float) and np.isnan(sd)):
        return a
    return f"{a} ± {fmt(sd, nd, False)}"


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
                        stats.spearmanr(g.y_hat, g.y).statistic,
                        topk_ret(g.y_hat.to_numpy(), g.y.to_numpy()))
            if o:
                ds.append(o)
        if not ds:
            return None
        k = sorted(set.intersection(*[set(x) for x in ds]))
        # P[run, 指標(0=IC, 1=RankIC, 2=Top-K 日報酬), 日]——保留逐 run 的
        # 日序列才算得出跨 run 的 sd。V 是它對 run 取平均，與加 sd 欄之前
        # 逐位元相同（第 2 個指標只給組合欄用，不進 V 的既有欄位）。
        P = np.array([[[x[d][i] for d in k] for i in (0, 1, 2)] for x in ds])
        V = P.mean(axis=0)
        # ICIR 逐 run 算完再平均。用 V[0]（跨 run 平均後的日序列）算會得到
        # 「集成」的 ICIR，系統性偏高，與主表的慣例不一致。
        per_ir = np.array([icir(P[r, 0]) for r in range(P.shape[0])])
        ir = float(np.nanmean(per_ir))
        return k, V, ir, P, per_ir, portfolio_stats(list(P[:, 2, :]))

    def _sd(a):
        """跨 run 的樣本 sd（ddof=1）。n<2 時回 NaN，`fmt` 會印成「—」。"""
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return float(a.std(ddof=1)) if a.size >= 2 else float("nan")

    def _hac(d):
        n = len(d); L = int(4 * (n / 100) ** (2 / 9)); s2 = np.var(d, ddof=1)
        for l in range(1, L + 1):
            s2 += 2 * (1 - l / (L + 1)) * np.cov(d[l:], d[:-l], ddof=1)[0, 1]
        tt = d.mean() / (s2 / n) ** 0.5
        return 2 * (1 - stats.norm.cdf(abs(tt)))

    b = _load(str(ROOT / "runs/**/*f2_best_s*/predictions/test_predictions.csv"))
    if b is None:
        return None
    kb, BESTV, BESTIR, BESTP, BESTIRS, BESTPF = b
    rows = [("**MAGNET 本版（F1 + 無A₂ + rank 1.0）**", None)]
    for lab, pat in (("KTW+（最高標）", "runs_f2/*fvg_KTWp/predictions/*.csv"),
                     ("[24] 二部圖 LASSO", "runs_f2/*bipartite*t2_LASSO/predictions/*.csv"),
                     ("R2 per-target ridge", "runs_f2/*ridge_R2/predictions/*.csv"),
                     ("[24] 二部圖 ens-avg", "runs_f2/*bipartite*t2_ens-avg/predictions/*.csv"),
                     ("TW+（美股+台股 80 維）", "runs_f2/*fvg_TWp/predictions/*.csv"),
                     ("MAGNET 基準（自家前版）", "runs/**/*f2_base_s*/predictions/test_predictions.csv"),
                     ("RC 常數對照", "runs_f2/*ridge_RC/predictions/*.csv")):
        rows.append((lab, _load(str(ROOT / pat))))
    sig, pf_rows = [], []                 # 拆成訊號層與組合讀法兩張表
    for lab, r in rows:
        if r is None and lab.startswith("**"):
            sig.append(
                f"| {lab} | {BESTP.shape[0]} "
                f"| {pm(BESTV[0].mean(), _sd(BESTP[:, 0].mean(axis=1)))} "
                f"| {pm(BESTIR, _sd(BESTIRS), signed=False)} "
                f"| {pm(BESTV[1].mean(), _sd(BESTP[:, 1].mean(axis=1)))} | — | — |")
            pf_rows.append(f"| {lab} | {BESTP.shape[0]} " + pf_cells(BESTPF) + "|")
            continue
        if r is None:
            continue
        kk, V, IR, P, IRS, PF = r
        ii = [kk.index(d) for d in kk if d in kb]
        jj = [kb.index(d) for d in kk if d in kb]
        cells = []
        for i in (0, 1):
            dd = BESTV[i][jj] - V[i][ii]
            cells += [f"{dd.mean():+.4f}", f"{_hac(dd):.4f}"]
        # IC / RankIC 的 sd 與其點估計取同一組日子（ii）。ICIR 的點估計 IR 是在
        # 完整 kk 上算的，其 sd 因此也用完整 kk，兩者口徑一致。
        sig.append(
            f"| {lab} | {P.shape[0]} "
            f"| {pm(V[0][ii].mean(), _sd(P[:, 0][:, ii].mean(axis=1)))} "
            f"| {pm(IR, _sd(IRS), signed=False)} "
            f"| {pm(V[1][ii].mean(), _sd(P[:, 1][:, ii].mean(axis=1)))} "
            f"| {cells[1]} | {cells[3]} |")
        pf_rows.append(f"| {lab} | {P.shape[0]} " + pf_cells(PF) + "|")
    return sig, pf_rows


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
        """回傳 {date: (IC, RankIC, 離散比, Top-K 日報酬)}。"""
        o = {}
        for d, g in df.groupby("target_date"):
            if g.y.std() == 0 or g.y_hat.std() == 0:
                continue
            o[d] = (np.corrcoef(g.y_hat, g.y)[0, 1],
                    stats.spearmanr(g.y_hat, g.y).statistic,
                    g.y_hat.std() / g.y.std(),
                    topk_ret(g.y_hat.to_numpy(), g.y.to_numpy()))
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
        # 第 4 個回傳值是**逐 seed**的 Top-K 日報酬矩陣 [n_seed, T]；
        # 組合指標要逐 seed 算完再平均，不能拿跨 seed 平均後的序列去算。
        return (k, np.array([[np.mean([m[d][i] for m in ms]) for d in k]
                             for i in (0, 1, 2, 3)]), len(ds),
                [np.array([m[d][3] for d in k]) for m in ms])

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
        return k, np.array([[m[d][i] for d in k] for i in (0, 1, 2, 3)]), df

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
    out.append("| 折 | n 種子 | 聚合 | IC ↑ | RankIC ↑ | 離散比 std(y_hat)/std(y) "
               + PF_HEAD + "|")
    out.append("|---|---:|---|---:|---:|---:" + PF_SEP + "|")
    ens_cache = {}
    for lab, pat, cut, _ in FOLDS:
        ps = _per_seed(pat, cut)
        en = _ens(pat, cut)
        if ps is None or en is None:
            continue
        ok = True
        ens_cache[lab] = (en, cut)
        out.append(f"| {lab} | {ps[2]} | 先算 IC 再平均 | {ps[1][0].mean():+.4f} "
                   f"| {ps[1][1].mean():+.4f} | {ps[1][2].mean():.2f} "
                   + pf_cells(portfolio_stats(ps[3])) + "|")
        out.append(f"| {lab} | {ps[2]} | **先平均預測再算 IC** "
                   f"| **{en[1][0].mean():+.4f}** | **{en[1][1].mean():+.4f}** "
                   f"| **{en[1][2].mean():.2f}** "
                   + pf_cells(portfolio_stats([en[1][3]])) + "|")
        # 「增益」是兩列相減，沒有對應的日報酬序列，組合欄留空
        out.append(f"| {lab} | | 增益 | {en[1][0].mean() - ps[1][0].mean():+.4f} "
                   f"| {en[1][1].mean() - ps[1][1].mean():+.4f} "
                   f"| {en[1][2].mean() - ps[1][2].mean():+.2f} "
                   + PF_BLANK + "|")
    if not ok:
        return None

    BLS = (("KTW+（最高標）", "*_fvg_KTWp"), ("[24] 二部圖 LASSO", "*bipartite*t2_LASSO"),
           ("R2 per-target ridge", "*_ridge_R2"), ("[24] 二部圖 ens-avg", "*bipartite*t2_ens-avg"),
           ("RC 常數對照", "*_ridge_RC"))
    out += ["", "### (B) 集成後對線性 baseline 的逐日檢定", "",
            "| 折 | 對照 | 其 IC ↑ | 其 RankIC ↑ | 逐日 p (IC) | 逐日 p (RankIC) "
            + PF_HEAD + "|",
            "|---|---|---:|---:|---:|---:" + PF_SEP + "|"]
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
                       f"| {pv[0]:.4f} | {pv[1]:.4f} "
                       + pf_cells(portfolio_stats(b[3])) + "|")

    out += ["", "### (C) 與 KTW+ 等權混合（w=0.5，逐日橫截面 z 分數，未調參）", "",
            "| 折 | 方法 | IC ↑ | RankIC ↑ | 逐日 p (IC) | 逐日 p (RankIC) "
            + PF_HEAD + "|",
            "|---|---|---:|---:|---:|---:" + PF_SEP + "|"]
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
            V = np.array([[m[d][i] for d in k] for i in (0, 1, 3)])
            pf = pf_cells(portfolio_stats([V[2]]))
            if w == 0.0:
                base = V
                out.append(f"| {lab} | {nm} | {V[0].mean():+.4f} | {V[1].mean():+.4f} "
                           f"| — | — " + pf + "|")
                continue
            d0, d1 = V[0] - base[0], V[1] - base[1]
            out.append(f"| {lab} | {nm} | {V[0].mean():+.4f} | {V[1].mean():+.4f} "
                       f"| {_hac(d0):.4f} | {_hac(d1):.4f} " + pf + "|")
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
    out.append(f"統計基準 = **{BEST}**（本專案目前最佳）。"
               "**已移除「類別」、`dIC`、`dRankIC` 三種欄位**"
               "（類別看得出來：`[n]` 是文獻、`KTW+`/`R2`/`[24]` 是線性、"
               "`RC` 是空模型；兩個差值欄自己減得出來），改放組合讀法的八欄。"
               "差值的**顯著性**仍在——就是 `逐日 p (IC)` 與 `逐日 p (RankIC)` 兩欄。")
    out.append("")
    ordered = sorted(rows, key=lambda r: -r[3]["ic"])

    out.append("## 第一折（主結果）")
    out.append("")
    out.append("### 訊號層")
    out.append("")
    out.append("| 方法 | 設定 | n | test IC ↑ | **ICIR ↑** | RankIC ↑ "
               "| 逐日 p | 跨種子 Welch p | MW p |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, cat, note, st, arm in ordered:
        _, p_pd = paired_daily(base, st, "ic")
        pw, pu, _ = across_seed(base, st, "ic")
        is_base = (arm == BEST)
        star = " **" if is_base else " "
        out.append(
            f"|{star}{label}{star.strip()} | {note} | {st['n']} | "
            f"{pm(st['ic'], st['ic_sd'])} | "
            f"{pm(st.get('icir'), st.get('icir_sd'), signed=False)} | "
            f"{pm(st['ric'], st['ric_sd'])} | "
            f"{'—' if is_base else fmt(p_pd, signed=False)} | "
            f"{fmt(pw, signed=False)} | {fmt(pu, signed=False)} |")

    out.append("")
    out.append("### 組合讀法")
    out.append("")
    out.append(f"**這不是回測。** Top-{TOPK} 多空、等權、金額中性、逐日再平衡，"
               "**無交易成本、無市場衝擊、無流動性限制、可完全放空**。"
               "用途是把 IC 翻譯成組合單位，不是策略績效（proposal §60.6 / §60.8）。"
               "本專案最佳 arm 的**打平來回成本是 33 bp，而台灣證交稅單項就 30 bp**"
               "（§60.4、`scripts/portfolio_readout.py`）。列序與上表相同。")
    out.append("")
    out.append("| 方法 | n " + PF_HEAD + "|")
    out.append("|---|---:" + PF_SEP + "|")
    for label, cat, note, st, arm in ordered:
        star = " **" if arm == BEST else " "
        out.append(f"|{star}{label}{star.strip()} | {st['n']} "
                   + pf_cells(st.get("pf")) + "|")
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
        f2_sig, f2_pf = f2
        out.append("### 訊號層（第二折）")
        out.append("")
        out.append("| 方法 | n | test IC ↑ | **ICIR ↑** | RankIC ↑ "
                   "| 逐日 p (IC) | 逐日 p (RankIC) |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        out += f2_sig
        out.append("")
        out.append("### 組合讀法（第二折）")
        out.append("")
        out.append("建構與警語同第一折的組合讀法表。列序與上表相同。")
        out.append("")
        out.append("| 方法 | n " + PF_HEAD + "|")
        out.append("|---|---:" + PF_SEP + "|")
        out += f2_pf
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
    out.append("## 每一欄怎麼算")
    out.append("")
    out.append("記號：某一天的橫截面有 `n` 檔台股，`y_hat` 是預測、`y` 是**實際次日 "
               "log return**（`log(Close_t / Close_{t-1})`，"
               "`src/dataset/pipeline.py:659`，由 `graph_builder.py` 的 "
               "`_extract_target_returns` 取出）。`T` 是**有效交易日數**——"
               "`std(y)=0` 或 `std(y_hat)=0` 的日子一律跳過，"
               "第一折 246 天扣掉 1 天（該日 50 檔報酬完全相同）得 **T=245**。")
    out.append("")
    out.append("### 訊號層的欄")
    out.append("")
    out.append("| 欄 | 算式 | 出處 |")
    out.append("|---|---|---|")
    out.append("| `n` | 種子數。線性／樹模型是確定性擬合（固定設計矩陣上的 "
               "`Ridge.fit`），**沒有 seed 可補**，一律 n=1 | `find_seeds` |")
    out.append("| `test IC` | 逐日 `corr(y_hat_t, y_t)`（Pearson）對 T 天取平均，"
               "再跨 seed 取平均。**逐 seed 的值讀自 `meta.json`**"
               "（`--predictions reeval` 時改讀 `meta_reeval.json`） | `neural_stats` |")
    out.append("| `ICIR` | `mean_t(IC_t) / sd_t(IC_t)`，`ddof=1`。"
               "**逐 seed 算完再跨 seed 平均**——拿跨 seed 平均後的日序列去算"
               "會得到「集成」的 ICIR，系統性偏高 | `icir()` |")
    out.append("| `RankIC` | 同 `IC` 但用 Spearman（平手取**平均名次**，`scipy.spearmanr`）。"
               "**一律由預測檔重算，不讀 `meta.json`** | `daily_series` |")
    out.append("| `± sd` | 跨 seed 的樣本標準差，`ddof=1`。n=1 時不印——"
               "**那不代表沒有不確定性**，只代表它來自資料而非種子 | `pm()` |")
    out.append("| `逐日 p` | 取「統計基準與該列」的**逐日 IC 差值序列** `d_t`，"
               "做 Newey-West HAC 修正的雙尾 t 檢定；"
               "`lag = floor(4 x (T/100)^(2/9))`，T=245 時 **lag=4**，`df = T−1` "
               "| `paired_daily` |")
    out.append("| `跨種子 Welch p` | 在**共同種子集合**上，對兩組逐 seed 的 IC 做 "
               "Welch t 檢定（不假設等變異） | `across_seed` |")
    out.append("| `MW p` | 同上，改用 Mann-Whitney U，雙尾。"
               "**n=3 時下限是 `2/C(6,3) = 0.1000`**，達不到 0.05 | `across_seed` |")
    out.append("")
    out.append("### 組合讀法的欄")
    out.append("")
    out.append(f"**建構**：每天把當日 `y_hat` 由高到低排序，"
               f"**前 {TOPK} 檔做多、後 {TOPK} 檔做空、等權**。"
               f"權重是 `±1/(2K)`，所以 `sum|w| = 1`（總曝險 1、淨曝險 0），"
               "因此當日報酬是")
    out.append("")
    out.append("```")
    out.append(f"r_t = ( mean(y[多腿 {TOPK} 檔]) − mean(y[空腿 {TOPK} 檔]) ) / 2")
    out.append("```")
    out.append("")
    out.append("那個 `/2` 不可省——不除等於偷偷假設 2 倍槓桿。"
               "八個指標**一律逐 seed 算完再平均**，與 `ICIR` 同慣例。")
    out.append("")
    out.append("| 欄 | 算式 | 為什麼是這個形式 |")
    out.append("|---|---|---|")
    out.append("| `日均報酬` | `mean(r)` | — |")
    out.append("| `日 sd` | `sd(r, ddof=1)` | — |")
    out.append("| `年化(幾何)` | `exp(mean(r) x 252) − 1` | **`y` 是 log return，"
               "所以複利用 `exp` 不是 `(1+r)` 連乘。** 用錯公式在本專案的量級上"
               "差約 1.1 個百分點（82.0% vs 83.1%） |")
    out.append("| `Sharpe` | `mean(r) / sd(r) x sqrt(252)`，`rf = 0` | "
               "金額中性多空是**自融資**的，多空相抵不佔用本金，"
               "所以無風險利率為 0。淨多頭策略就必須扣 |")
    out.append("| `MDD(複利)` | 權益曲線 `exp(cumsum(r))`，"
               "`max_t (peak_t − v_t)/peak_t` | 獲利滾入的讀法 |")
    out.append("| `MDD(加總)` | 權益曲線 `1 + cumsum(r)`，同上取最大回撤 | "
               "**獲利不滾入的固定名目本金**。金額中性多空每天重設回同樣曝險，"
               "這個讀法才對應實際操作 |")
    out.append("")
    out.append("**已移除的兩欄可以自己還原**，它們是純換算、沒有新資訊：")
    out.append("")
    out.append("```")
    out.append("年化(簡單) = 日均報酬 x 252          年化波動 = 日 sd x sqrt(252)")
    out.append("Sharpe     = 年化(簡單) / 年化波動    （所以 Sharpe 才是新資訊）")
    out.append("```")
    out.append("")
    out.append("**已知近似**：對 log return 取橫截面平均**不等於**等權組合的報酬"
               "（後者是簡單報酬的算術平均）。實測差 **+0.14 bp/日 = +0.36 pp/年**，"
               "不影響 `Sharpe`，也不影響 §60.4 的打平成本（兩種算法都是 33 bp）。")
    out.append("")
    out.append("### §43 專屬的欄")
    out.append("")
    out.append("| 欄 | 算式 |")
    out.append("|---|---|")
    out.append("| `離散比 std(y_hat)/std(y)` | 逐日的橫截面標準差之比，對 T 天取平均。"
               "**在 ŷ 宣告為「基數分數、尺度未校正」下這是規範選擇不是缺陷**"
               "（IC 與 RankIC 都看不到它，proposal §57.11 / §60.7） |")
    out.append("| `其 IC` / `其 RankIC`（B 表） | 該 baseline 自己的值，"
               "算法同訊號層 |")
    out.append("| `增益`（A 表） | 「先平均預測再算 IC」減「先算 IC 再平均」。"
               "它是兩列相減，**沒有對應的日報酬序列**，故組合欄留空 |")
    out.append("")
    out.append("## 讀表注意")
    out.append("")
    out.append("- **欄名的 ↑ / ↓ 是「其他條件相同下哪個方向較好」**，"
               "**不是說該欄可以單獨拿來排名**。有三類欄位**刻意沒有箭頭**：<br>"
               "(1) **`sd` 各欄**（跨 seed 離散度）——離散度小只代表結果比較不挑種子，"
               "一個爛模型的 sd 小並不好，它不是績效；<br>"
               "(2) **`逐日 p (IC)` / `逐日 p (RankIC)` / `Welch p` / `MW p`**——這些是"
               "**「統計基準對該列」**的檢定與差值，不是該列自己的績效。"
               "p 小代表基準顯著贏過該列，站在該列的立場方向是相反的；<br>"
               "(3) **`離散比 std(y_hat)/std(y)`**（§43 A 表）——在 ŷ 宣告為"
               "「基數分數、尺度未校正」之下，IC 與 RankIC 對仿射變換不變、"
               "**兩者都看不到這個比值**，所以它是規範選擇不是缺陷"
               "（proposal §57.11、§60.7）。")
    out.append("- **`MDD ↓` 的箭頭尤其不等於可以拿來排名**（見下方 MDD 那條）："
               "它兩折的名次會翻轉。箭頭只說明方向，不背書該欄的可靠度。")
    out.append("- **n 是種子數。** sigma_seed = 0.0062，n=3 的最小可偵測差異為 0.0188、"
               "n=10 為 0.0082。n=3 的比較若差異小於 0.019，「不顯著」不構成證據。")
    out.append("- **Mann-Whitney 在 n=3 的雙尾 p 下限是 0.10**（= 2/C(6,3)，"
               "`across_seed` 取共同種子集合，故 n=3 的列是 3 對 3），永遠無法達到 0.05。"
               "**本表九列 n=3 的 MW p 全部恰為 0.1000——那是下界，不是量測值**，"
               "不可讀成「接近顯著」；同樣九列的 Welch p 是 0.0000 ~ 0.0016，"
               "IC 差距約 0.10 而 sd 約 0.01。"
               "n=1（非神經方法）則無法計算，跨種子欄為「—」。")
    out.append("- **第二折表也有 n 與 sd 欄了**（2026-09-14 補）。"
               "點估計逐位元未變，只是原本沒把離散度印出來。"
               "**但 ICIR 沒有逐日檢定可做**——它是整段期間兩個動差的比值，"
               "不存在逐日序列，所以它只有跨種子這一個檢定，"
               "而跨種子檢定**不包含日層級的不確定性**（那才是主要來源）。"
               "因此 ICIR 的差只能說「兩折同向」，不可升級成「顯著落後」。")
    out.append(f"- **右邊八欄是組合讀法，不是回測。** 建構：每天按 `y_hat` 取"
               f"**Top-{TOPK} 做多 / Bottom-{TOPK} 做空、等權、金額中性**"
               f"（權重 ±1/(2K)，`sum|w|=1`，所以多空腿的報酬差要除以 2），"
               "逐日再平衡。**無交易成本、無市場衝擊、無流動性限制、可完全放空**——"
               "所以 `Sharpe` 5.86 / `年化` 60% 這種數字**不可以寫成策略績效**"
               "（proposal §60.8）。打平成本與成本敏感度見 §60.4 與 "
               "`scripts/portfolio_readout.py`：本專案最佳 arm 的打平來回成本是 "
               "**33 bp**，而台灣證交稅單項就 30 bp。")
    out.append("- **組合欄的複利用 `exp`，因為 `y` 是 log return**"
               "（`src/dataset/pipeline.py:659`）。`年化(幾何)` = "
               "`exp(mean(r) x 252) − 1`，`MDD(複利)` 的權益曲線 = `exp(cumsum(r))`；"
               "`MDD(加總)` 用 `1 + cumsum(r)`，即獲利不滾入的固定名目本金"
               "（金額中性多空每天重設回同樣曝險，這個讀法才對應實際操作）。"
               "`年化(簡單)` = `mean(r) x 252`。八欄一律**逐 seed 算完再平均**，與 ICIR 同慣例。"
               "**已知近似**：對 log return 取橫截面平均不等於等權組合的報酬"
               "（後者是簡單報酬的算術平均），實測差 **+0.14 bp/日 = +0.36 pp/年**，"
               "不影響 Sharpe，也不影響打平成本（兩種算法都是 33 bp）。")
    out.append("- **`MDD` 是單路徑極值，一折只有一個觀測值，是八欄裡最不可靠的，"
               "而且它兩折的名次是反的**：第一折 MAGNET 4.51% 勝過 KTW+ 5.05%，"
               "第二折 1.70% 卻輸給 1.32%。**不要拿 MDD 做任何排名主張。**"
               "會翻轉正是它不可靠的直接證據——與 `ICIR` 兩折同向落後恰成對比。")
    out.append("- **逐日 p 用 Newey-West HAC 修正自相關**，以「日」為重複單位、種子視為固定，"
               "敏感但不外推到新種子。兩欄應一起看。")
    out.append("- **六個文獻 baseline 各只跑一組預設超參，MAGNET 跑了約 50 組設定。**"
               "這是目前最大的公平性缺口，比較結果須據此保留。")
    out.append("- **ICIR = mean(逐日 IC) / sd(逐日 IC)**，逐 seed 算完再跨 seed 平均"
               "（不是拿跨 seed 平均後的日序列去算——那是集成的 ICIR，系統性偏高）。"
               "**為什麼要看它**：§57.8 的恆等式是 `r_t = IC_t x sigma_t`，"
               "所以 IC 只決定組合報酬的**分子**；若 sigma_t 為常數，"
               "`Sharpe = ICIR x sqrt(252)`。**與回測 Sharpe 對應的是 ICIR，不是 IC。**"
               "實測（§60）本專案最佳 arm 的 IC 高於 KTW+（+0.1090 vs +0.1077）"
               "但 ICIR 較低（0.4368 vs 0.4810），Top-10 多空的 Sharpe 也較低"
               "（5.86 vs 6.53）——**IC 的排名不保證回測的排名**。")
    out.append("- **沒有報 RankICIR**：組合報酬的恆等式建立在 Pearson IC 上，"
               "名次相關沒有對應的組合讀法，報了會被誤用。")
    out.append("- **本表只報 IC / RankIC / ICIR，三者都是相關係數的函數、對預測的尺度不敏感。**"
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
               "（IC 差前半 +0.0300、後半 −0.0141），第二折則相反（後半更好）。"
               "任何只根據單一測試期的排名都不可靠——這是本表最重要的保留條款。")
    out.append("- **`F1 + rank 1.0（保留 A₂）` 的 ICIR（0.4439）看起來比本版（0.4368）高，"
               "但那是噪音**：配對 p=0.2016，本版只在 4/10 顆種子落後。"
               "同一組配對下**本版的 RankIC 高 +0.0103，p<0.0001，10/10 種子**，"
               "IC 高 +0.0033（p=0.055）。組合層面也一致——Sharpe 5.92 vs 5.86 打平。"
               "**差別是 `無A₂` 買到的排序品質**（§35.3：`無A₂` 對 RankIC 的主效應 "
               "+0.0077，10/10）。這一格與 §60.2 那個已被 P6 證偽的「F1 單獨 ICIR 較高」"
               "是同一種讀法陷阱，**不要拿單一指標的名次下結論**。")
    out.append("- **本版的三個改動是兩兩協同，不是三階交互作用**"
               "（§35.3，2026-09-15 補完 2^3 的八格）。"
               "**三階項在 IC / RankIC / ICIR 上都是零**（−0.0002 / −0.0001 / +0.0008，"
               "p 全部 > 0.73）；三個兩兩交互項在 IC 上都是正且顯著。"
               "主效應是 **F1 +0.0050（p=0.026）**、**rank1.0 +0.0037（p=0.016）**、"
               "無A₂ −0.0006（ns）——"
               "舊版寫的「三個改動單獨都無效」描述的是**基準角落的條件效應**，"
               "不是主效應，那是只看一個角落造成的假象。"
               "`無A₂` 的真實作用是**用 ICIR 換 RankIC**"
               "（+0.0077 / −0.0258，兩者皆 10/10 種子），在 IC 上恰好抵消。")
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
