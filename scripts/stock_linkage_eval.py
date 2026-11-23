"""
stock_linkage_eval.py — 評估議題 5：個股與美股的對應。把逐日 IC 拆成每檔股票的貢獻，依 ADR 配對與產業分四組（第一折）

登記：docs/proposal_1209_revision.md §6 附錄（計算前寫下）。只做第一折，描述性分析，只跑一次。

定義：
    z        當天 50 檔的橫截面標準化（減平均，除以母體標準差 ddof=0）
    s_jt     z(ŷ)_jt · z(y)_jt；當天 IC_t = (1/50) Σ_j s_jt（Pearson 相關的恆等式）
    c_gt     組 g 的每檔平均貢獻 (1/n_g) Σ_{j∈g} s_jt；IC_t = Σ_g (n_g/50) c_gt
    RankIC   同樣的拆解用在當天的平均名次上
    神經方法逐種子算 s_jt 再對種子平均；當天 50 檔的 ŷ 或 y 全部相同時不計（與主表相同）。

分組（configs/universe/universe_tw_2019.json 的 tw_nodes，2018Q4 產業類別）：
    ADR 配對 7 / 未配對電子 15 / 金融 14 / 傳產 14；「未配對」= 後三組 43 檔。

方法（讀法與主表 results_table.py 相同）：
    MAGNET                 tw50_rgcnSA（10 顆）
    MAGNET-single market   tw50_smktSA（10 顆）
    LASSO [24]             *_bipartite_rolling_t2_LASSO 的最新一批
    MEIG [1]               tw50_bl_meig（3 顆）

主要對比（逐日序列，HAC 雙尾 p，與 results_table.paired_daily 同法；IC 兩個 p 一起 Holm，RankIC 另一族）：
    A  [c_ADR − c_未配對]_MAGNET − [c_ADR − c_未配對]_單市場     跨市場增益是否集中在配對股
    B  c_ADR,MAGNET − c_ADR,LASSO                               配對股上恆等邊是否勝過統計篩邊

用法：
    .venv/bin/python -I scripts/stock_linkage_eval.py --check    # 只做登記的檢查，不印分組結果
    .venv/bin/python -I scripts/stock_linkage_eval.py --write    # 只跑一次；結果檔已存在就拒絕
    .venv/bin/python -I scripts/stock_linkage_eval.py --report   # 之後只讀結果檔重印

不需要 torch：只讀各 run 的 predictions/test_predictions*.csv。
"""

from __future__ import annotations

import argparse
import json
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
import results_table as R  # noqa: E402

UNIVERSE = ROOT / "configs" / "universe" / "universe_tw_2019.json"
OUT_JSON = ROOT / "docs" / "figures" / "data" / "eval_stock_linkage.json"

METHODS = [("MAGNET", "neural", "tw50_rgcnSA"),
           ("MAGNET-single market", "neural", "tw50_smktSA"),
           ("LASSO [24]", "nonneural", R.BIP.format("LASSO")),
           ("MEIG [1]", "neural", "tw50_bl_meig")]
EXPECT_IC = {"MAGNET": 0.1127, "MAGNET-single market": 0.0250, "LASSO [24]": 0.1042, "MEIG [1]": 0.0061}

ELECTRONICS = ("半導體業", "電腦及週邊設備業", "光電業", "通信網路業", "電子零組件業", "其他電子業")
GROUPS = ("ADR-paired", "Unpaired electronics", "Financials", "Traditional")
GROUP_ZH = {"ADR-paired": "ADR 配對", "Unpaired electronics": "未配對電子", "Financials": "金融",
            "Traditional": "傳產", "Unpaired (all 43)": "未配對合計"}
# 登記第 2 節的表（計算前固定）；程式依規則重建後必須與此相同。
EXPECT_GROUPS = {
    "ADR-paired": {"2303", "2317", "2330", "2409", "2412", "3711", "8150"},
    "Unpaired electronics": {"2408", "2454", "2301", "2357", "2382", "2395", "4938", "3008", "3481",
                             "3045", "4904", "2308", "2327", "2354", "2474"},
    "Financials": {"2801", "2880", "2881", "2882", "2883", "2884", "2885", "2886", "2887", "2890",
                   "2891", "2892", "5876", "5880"},
    "Traditional": {"1101", "1102", "1216", "1301", "1303", "1326", "1402", "2002", "2105", "2633",
                    "2912", "5871", "6505", "9904"},
}


def build_groups() -> dict[str, set[str]]:
    nodes = json.loads(UNIVERSE.read_text())["tw_nodes"]
    g = {k: set() for k in GROUPS}
    for n in nodes:
        if n["adr"]:
            g["ADR-paired"].add(n["code"])
        elif n["industry"] in ELECTRONICS:
            g["Unpaired electronics"].add(n["code"])
        elif n["industry"] == "金融保險業":
            g["Financials"].add(n["code"])
        else:
            g["Traditional"].add(n["code"])
    if g != EXPECT_GROUPS or sum(map(len, g.values())) != 50:
        raise SystemExit("分組與登記第 2 節不符，停止")
    return g


def pred_files(kind: str, key: str) -> list[str]:
    if kind == "neural":
        runs = R.find_seeds(key)
        fs = [R._pred_file(d) for _, d in sorted(runs.items(), key=lambda kv: int(kv[0]))]
    else:
        d = R._latest("runs/**/" + key)
        fs = [R._pred_file(str(ROOT / d))] if d else []
    fs = [f for f in fs if f is not None]
    if not fs:
        raise FileNotFoundError(f"找不到 {key} 的預測檔")
    return fs


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std(ddof=0)


def contributions(f: str) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """單一 run -> (日期, 股票, s_jt [天, 檔], 名次版 s_jt)；略過的日子整列為 NaN。"""
    df = pd.read_csv(f, dtype={"ticker": str})
    H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
    Y = df.pivot(index="target_date", columns="ticker", values="y").sort_index()
    if list(H.columns) != list(Y.columns) or H.isna().any().any() or Y.isna().any().any():
        raise ValueError(f"{f} 的預測表不完整")
    Hn, Yn = H.to_numpy(float), Y.to_numpy(float)
    S = np.full(Hn.shape, np.nan)
    SR = np.full(Hn.shape, np.nan)
    for t in range(len(Hn)):
        a, b = Hn[t], Yn[t]
        if np.ptp(a) == 0 or np.ptp(b) == 0:         # 與 results_table._frame_series 相同
            continue
        S[t] = _z(a) * _z(b)
        SR[t] = _z(stats.rankdata(a)) * _z(stats.rankdata(b))
    return H.index.to_numpy(), list(H.columns), S, SR


def seed_mean(stack: np.ndarray) -> np.ndarray:
    """[種子, 天, 檔] -> [天, 檔]；某天全部種子都略過時為 NaN（與 results_table.colmean 相同）。"""
    out = np.full(stack.shape[1:], np.nan)
    ok = ~np.isnan(stack).all(axis=(0, 2))
    out[ok] = np.nanmean(stack[:, ok], axis=0)
    return out


def hac(d: np.ndarray) -> tuple[float, float, float, int]:
    """(平均, HAC 標準誤, 雙尾 p, 天數)。與 results_table.paired_daily 同一個公式（Bartlett 核，t 分配）。"""
    d = d[~np.isnan(d)]
    n = len(d)
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    x = d - d.mean()
    var = float((x @ x) / n)
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float((x[k:] @ x[:-k]) / n)
    se = np.sqrt(max(var, 1e-24) / n)
    t = d.mean() / se
    return float(d.mean()), float(se), float(2 * stats.t.sf(abs(t), n - 1)), n


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


def load() -> tuple[np.ndarray, dict]:
    """回傳 (日期, {方法: {"ic": [天, 檔], "ric": [天, 檔], "tickers": [...], "n": 顆數}})，並做登記第 3 節的檢查。"""
    dates, out = None, {}
    print("檢查（登記第 3 節）：")
    for name, kind, key in METHODS:
        parts = [contributions(f) for f in pred_files(kind, key)]
        d0, tk0 = parts[0][0], parts[0][1]
        if any(not np.array_equal(p[0], d0) or p[1] != tk0 for p in parts):
            raise ValueError(f"{name} 各種子的日期或股票不同")
        S = seed_mean(np.stack([p[2] for p in parts]))
        SR = seed_mean(np.stack([p[3] for p in parts]))
        d = np.asarray([str(x)[:10] for x in d0])
        if dates is None:
            dates = d
        elif not np.array_equal(dates, d):
            raise ValueError(f"{name} 的日期與其他方法不同")

        st = R.neural_stats(key) if kind == "neural" else R.nonneural_stats(key)
        ic_t, ric_t = S.mean(axis=1), SR.mean(axis=1)
        ref_ic, ref_ric = np.asarray(st["daily_ic"], float), np.asarray(st["daily_ric"], float)
        same_nan = np.array_equal(np.isnan(ic_t), np.isnan(ref_ic)) and np.array_equal(np.isnan(ric_t), np.isnan(ref_ric))
        ok = ~np.isnan(ref_ic)
        dmax = max(np.abs(ic_t[ok] - ref_ic[ok]).max(), np.nanmax(np.abs(ric_t - ref_ric)))
        ic = float(np.nanmean(ic_t))
        good = same_nan and dmax <= 1e-9 and abs(ic - EXPECT_IC[name]) < 5e-5
        print(f"  {name:22s} 四組加權和 IC {ic:+.4f}（主表 {EXPECT_IC[name]:+.4f}）  RankIC {np.nanmean(ric_t):+.4f}  "
              f"逐日最大差 {dmax:.1e}  略過 {int(np.isnan(ic_t).sum())} 天  n={len(parts)}"
              f"{'' if good else '  <-- 不符'}")
        if not good:
            raise SystemExit("檢查未通過，停止（不印分組結果）")
        out[name] = dict(ic=S, ric=SR, tickers=tk0, n=len(parts))
    return dates, out


def compute(dates: np.ndarray, M: dict, groups: dict[str, set[str]]) -> dict:
    gnames = list(GROUPS) + ["Unpaired (all 43)"]
    members = dict(groups)
    members["Unpaired (all 43)"] = set().union(*(groups[g] for g in GROUPS[1:]))

    # c[方法][指標][組] = 逐日序列
    c = {}
    for name, d in M.items():
        idx = {g: [d["tickers"].index(t) for t in sorted(members[g])] for g in gnames}
        c[name] = {k: {g: d[k][:, idx[g]].mean(axis=1) for g in gnames} for k in ("ic", "ric")}

    res = dict(fold=1, n_days=int(len(dates)), start=str(dates[0]), end=str(dates[-1]),
               n_skipped=int(np.isnan(c["MAGNET"]["ic"]["ADR-paired"]).sum()),
               groups={g: dict(n=len(members[g]), tickers=sorted(members[g])) for g in gnames},
               definition=("s_jt = z(yhat)_jt * z(y)_jt with cross-sectional z over all 50 stocks (ddof=0); "
                           "c_g = mean of s_jt over stocks in group g; IC_t = sum_g (n_g/50) c_g,t. "
                           "Neural methods: seed mean of s_jt. RankIC: same on average ranks."),
               levels={}, contrasts={}, descriptive={})

    for name in M:
        res["levels"][name] = {}
        for k in ("ic", "ric"):
            res["levels"][name][k] = {}
            for g in gnames:
                m, se, _, n = hac(c[name][k][g])
                res["levels"][name][k][g] = dict(mean=m, se=se, total=m * len(members[g]) / 50, n=n)

    MAG, SM, LA = "MAGNET", "MAGNET-single market", "LASSO [24]"
    adr, un = "ADR-paired", "Unpaired (all 43)"
    for k in ("ic", "ric"):
        A = (c[MAG][k][adr] - c[MAG][k][un]) - (c[SM][k][adr] - c[SM][k][un])
        B = c[MAG][k][adr] - c[LA][k][adr]
        rows = {}
        for lab, ser in (("A", A), ("B", B)):
            m, se, p, n = hac(ser)
            rows[lab] = dict(mean=m, se=se, p=p, n=n)
        for lab, ph in zip(("A", "B"), holm([rows["A"]["p"], rows["B"]["p"]])):
            rows[lab]["p_holm"] = ph
        res["contrasts"][k] = rows

        desc = {}
        for other in (SM, LA):
            for g in gnames:
                m, se, p, n = hac(c[MAG][k][g] - c[other][k][g])
                desc[f"MAGNET - {other} | {g}"] = dict(mean=m, se=se, p=p, n=n)
        m, se, p, n = hac((c[MAG][k][adr] - c[MAG][k][un]) - (c[LA][k][adr] - c[LA][k][un]))
        desc["[ADR - unpaired]_MAGNET - [ADR - unpaired]_LASSO"] = dict(mean=m, se=se, p=p, n=n)
        res["descriptive"][k] = desc

    # 自我檢查：hac() 與 results_table.paired_daily 的 p 相同
    _, p_ref = R.paired_daily({"daily_x": c[MAG]["ic"][adr]}, {"daily_x": c[LA]["ic"][adr]}, "x")
    if abs(p_ref - res["contrasts"]["ic"]["B"]["p"]) > 1e-12:
        raise SystemExit("hac() 與 results_table.paired_daily 不一致")
    return res


def report(res: dict) -> None:
    gnames = list(res["groups"])
    print(f"\n第一折 test：{res['start']} ~ {res['end']}，{res['n_days']} 天（略過 {res['n_skipped']} 天）")
    for k, lab in (("ic", "IC"), ("ric", "RankIC")):
        print(f"\n每檔平均貢獻 c_g（{lab}，± HAC 標準誤）；括號內為組的總貢獻 (n_g/50)·c_g：")
        print(f"  {'':22s}" + "".join(f"{GROUP_ZH[g] + '（' + str(res['groups'][g]['n']) + '）':>22s}" for g in gnames))
        for name, L in res["levels"].items():
            print(f"  {name:22s}" + "".join(
                f"{L[k][g]['mean']:+8.4f}±{L[k][g]['se']:.4f}({L[k][g]['total']:+.4f})" for g in gnames))
    for k, lab in (("ic", "IC"), ("ric", "RankIC")):
        print(f"\n主要對比（{lab}；HAC 雙尾 p，兩個一起 Holm）：")
        for key, txt in (("A", "A [c_ADR − c_未配對]_MAGNET − [同]_單市場"), ("B", "B c_ADR,MAGNET − c_ADR,LASSO")):
            r = res["contrasts"][k][key]
            print(f"  {txt:42s} {r['mean']:+.4f} ± {r['se']:.4f}  p {r['p']:.4f}  Holm {r['p_holm']:.4f}")
        print(f"描述（{lab}；未校正）：")
        for key, r in res["descriptive"][k].items():
            print(f"  {key:62s} {r['mean']:+.4f} ± {r['se']:.4f}  p {r['p']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="只做登記的檢查，不印分組結果")
    g.add_argument("--write", action="store_true", help=f"計算並寫 {OUT_JSON.relative_to(ROOT)}（只跑一次）")
    g.add_argument("--report", action="store_true", help="只讀結果檔重印")
    args = ap.parse_args()

    if args.report:
        report(json.loads(OUT_JSON.read_text()))
        return
    if args.write and OUT_JSON.exists():
        raise SystemExit(f"{OUT_JSON.relative_to(ROOT)} 已存在：登記規定只跑一次，請改用 --report")
    groups = build_groups()
    print("分組與登記第 2 節相同：" + "、".join(f"{GROUP_ZH[k]} {len(v)}" for k, v in groups.items()) + "\n")
    dates, M = load()
    if args.check:
        return
    res = compute(dates, M, groups)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    report(res)
    print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
