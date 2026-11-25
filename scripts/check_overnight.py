"""
check_overnight.py — 教授 11-25 意見的檢驗 3：隔夜與盤中拆解（第一折）

登記：docs/proposal_1209_revision.md §9 附錄（計算前寫下）。描述性分析，--write 只跑一次。

問題：跨市場訊號在台股 09:00 開盤前反映了多少？（議題 3 除了「落後一天就沒了」之外的資訊）

報酬拆解（data/processed/tw 的 OHLC，與 overnight_decomposition.py 相同）：
    r_tot = log(C_t / C_{t−1})    = 預測檔的 y
    r_gap = log(O_t / C_{t−1})    前一天 13:30 收盤到 09:00 開盤（美股盤在這一段裡）
    r_int = log(C_t / O_t)        盤中
量（每天、每方法；神經方法逐種子算再對種子平均，與主表相同）：
    IC_leg(t)   corr_j(ŷ, r_leg)；RankIC 用平均名次
    a_leg(t)    (1/50) Σ_j z(ŷ)_j · r_leg,j（z 為 ddof=0），a_tot = a_gap + a_int（恆等）
    隔夜占比     Σ_t a_gap / Σ_t a_tot
    跨市場增益   MAGNET − 單市場，gap / int 兩段的逐日差

用法：
    .venv/bin/python -I scripts/check_overnight.py --smoke    # 隨機數字，不讀任何預測
    .venv/bin/python -I scripts/check_overnight.py --check    # 只做閘門與重現檢查
    .venv/bin/python -I scripts/check_overnight.py --write    # 只跑一次；結果檔已存在就拒絕
    .venv/bin/python -I scripts/check_overnight.py --report   # 之後只讀結果檔重印

不需要 torch。
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
import stock_linkage_eval as SL  # noqa: E402

OUT_JSON = ROOT / "docs" / "figures" / "data" / "eval_check_overnight.json"
MEIG_SEEDS = ("7", "42", "123", "11", "99", "314", "555", "777", "888", "2026")   # §8 登記的 10 顆
METHODS = [("MAGNET", "neural", "tw50_rgcnSA", None),
           ("MAGNET-single market", "neural", "tw50_smktSA", None),
           ("LASSO [24]", "nonneural", R.BIP.format("LASSO"), None),
           ("MEIG [1]", "neural", "tw50_bl_meig", MEIG_SEEDS)]
EXPECT_IC = {"MAGNET": 0.1127, "MAGNET-single market": 0.0250, "LASSO [24]": 0.1042, "MEIG [1]": 0.0099}
LEGS = ("tot", "gap", "int")


def files(kind: str, key: str, seeds) -> list[str]:
    if kind == "neural":
        runs = R.find_seeds(key)
        if seeds is not None:
            if not set(seeds) <= set(runs):
                raise FileNotFoundError(f"{key} 缺種子 {sorted(set(seeds) - set(runs))}")
            runs = {s: runs[s] for s in seeds}
        fs = [R._pred_file(d) for _, d in sorted(runs.items(), key=lambda kv: int(kv[0]))]
    else:
        d = R._latest("runs/**/" + key)
        fs = [R._pred_file(str(ROOT / d))] if d else []
    if not fs or any(f is None for f in fs):
        raise FileNotFoundError(f"找不到 {key} 的預測檔")
    return fs


def tw_legs(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """{leg: DataFrame[日期, 股票]}，與 overnight_decomposition.tw_panel 同一個公式。"""
    out = {k: {} for k in LEGS}
    for t in tickers:
        d = (pd.read_csv(ROOT / f"data/processed/tw/{t}.csv", parse_dates=["Date"])
             .sort_values("Date").set_index("Date"))
        out["tot"][t] = np.log(d.Close / d.Close.shift(1))
        out["gap"][t] = np.log(d.Open / d.Close.shift(1))
        out["int"][t] = np.log(d.Close / d.Open)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std(ddof=0)


def per_run(f: str, L: dict) -> tuple[np.ndarray, dict, float, float]:
    """單一預測檔 -> (日期, {量: 逐日序列}, 閘門的 max|y − r_tot|, 恆等式誤差)。"""
    df = pd.read_csv(f, dtype={"ticker": str})
    H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
    Y = df.pivot(index="target_date", columns="ticker", values="y").sort_index()
    idx = pd.to_datetime(H.index)
    legs = {k: L[k].reindex(index=idx, columns=H.columns).to_numpy(float) for k in LEGS}
    gate = float(np.nanmax(np.abs(Y.to_numpy(float) - legs["tot"])))
    if np.isnan(legs["tot"]).any() or np.isnan(legs["gap"]).any():
        gate = float("inf")
    add = float(np.abs(legs["tot"] - (legs["gap"] + legs["int"])).max())
    Hn, Yn = H.to_numpy(float), Y.to_numpy(float)
    n = len(Hn)
    S = {f"{m}_{k}": np.full(n, np.nan) for m in ("ic", "ric", "a") for k in LEGS}
    for t in range(n):
        a, b = Hn[t], Yn[t]
        if np.ptp(a) == 0 or np.ptp(b) == 0:          # 與主表相同的略過規則
            continue
        za, ra = _z(a), _z(stats.rankdata(a))
        for k in LEGS:
            r = legs[k][t]
            S[f"a_{k}"][t] = float(za @ r / len(r))
            if np.ptp(r) > 0:
                S[f"ic_{k}"][t] = float(np.corrcoef(a, r)[0, 1])
                S[f"ric_{k}"][t] = float(ra @ _z(stats.rankdata(r)) / len(r))
    return np.asarray([str(x)[:10] for x in H.index]), S, gate, add


def load(L: dict | None = None) -> tuple[np.ndarray, dict, dict]:
    dates, out, gates = None, {}, {}
    for name, kind, key, seeds in METHODS:
        fs = files(kind, key, seeds)
        parts = [per_run(f, L) for f in fs]
        d0 = parts[0][0]
        if any(not np.array_equal(p[0], d0) for p in parts):
            raise ValueError(f"{name} 各種子的日期不同")
        if dates is None:
            dates = d0
        elif not np.array_equal(dates, d0):
            raise ValueError(f"{name} 的日期與其他方法不同")
        out[name] = {k: R.colmean(np.stack([p[1][k] for p in parts])) for k in parts[0][1]}
        out[name]["n"] = len(parts)
        gates[name] = dict(y_vs_rtot=max(p[2] for p in parts), additive=max(p[3] for p in parts))
    return dates, out, gates


def verify(M: dict, gates: dict) -> None:
    print("檢查（登記第 5 節）：")
    ok_all = True
    for name, d in M.items():
        g = gates[name]
        ic = float(np.nanmean(d["ic_tot"]))
        good = g["y_vs_rtot"] <= 1e-6 and g["additive"] <= 1e-12 and abs(ic - EXPECT_IC[name]) < 5e-5
        ok_all &= good
        print(f"  {name:22s} max|y − r_tot| {g['y_vs_rtot']:.1e}  恆等式 {g['additive']:.1e}  "
              f"IC(r_tot) {ic:+.4f}（主表 {EXPECT_IC[name]:+.4f}）  n={d['n']}{'' if good else '  <-- 不符'}")
    if not ok_all:
        raise SystemExit("檢查未通過，停止（不印結果）")
    print("  通過\n")


def _h(x: np.ndarray) -> dict:
    m, se, p, n = SL.hac(np.asarray(x, float))
    return dict(mean=m, se=se, p=p, n=n)


def _share(gap: np.ndarray, tot: np.ndarray) -> float:
    ok = ~np.isnan(gap) & ~np.isnan(tot)
    return float(np.sum(gap[ok]) / np.sum(tot[ok]))


def _label_share(s: float) -> str:
    return ("幾乎全部在台股開盤前就反映" if s >= 0.8 else
            "大部分在開盤前反映" if s >= 0.5 else "主要在盤中")


def compute(dates: np.ndarray, M: dict) -> dict:
    res = dict(fold=1, n_days=int(len(dates)), start=str(dates[0]), end=str(dates[-1]),
               definition=("r_gap = log(O_t/C_{t-1}), r_int = log(C_t/O_t) from data/processed/tw; "
                           "IC per leg = daily Pearson of yhat with r_leg; a_leg = mean_j z(yhat)_j r_leg,j; "
                           "overnight share = sum a_gap / sum a_tot; neural methods: per-seed daily series averaged over seeds"),
               methods={}, gain={}, judgement={})
    for name, d in M.items():
        r = {f"{m}_{k}": _h(d[f"{m}_{k}"]) for m in ("ic", "ric", "a") for k in LEGS}
        r["share_overnight"] = _share(d["a_gap"], d["a_tot"])
        r["n_seeds"] = d["n"]
        res["methods"][name] = r
    mg, sm = M["MAGNET"], M["MAGNET-single market"]
    res["gain"] = {f"{m}_{k}": _h(mg[f"{m}_{k}"] - sm[f"{m}_{k}"]) for m in ("ic", "ric", "a") for k in LEGS}
    res["gain"]["share_overnight"] = _share(mg["a_gap"] - sm["a_gap"], mg["a_tot"] - sm["a_tot"])
    for name in ("MAGNET", "LASSO [24]"):
        r = res["methods"][name]
        ii = r["ic_int"]
        res["judgement"][name] = dict(
            overnight=_label_share(r["share_overnight"]),
            intraday=(f"開盤後仍有少量可預測性（IC {ii['mean']:+.4f}）" if ii["mean"] > 0 and ii["p"] < 0.05
                      else "盤中與 0 分不出差別"))
    res["judgement"]["cross-market gain"] = dict(overnight=_label_share(res["gain"]["share_overnight"]))
    return res


def report(res: dict) -> None:
    print(f"第一折 {res['start']} ~ {res['end']}，{res['n_days']} 天\n")
    for m, lab in (("ic", "IC"), ("ric", "RankIC"), ("a", "a（組合報酬，%/日）")):
        sc = 100 if m == "a" else 1
        print(f"{lab}：總報酬 / 隔夜 / 盤中（對 0 的 HAC p）")
        for name, r in res["methods"].items():
            print(f"  {name:22s}" + "".join(
                f"  {r[f'{m}_{k}']['mean'] * sc:+.4f}({r[f'{m}_{k}']['p']:.3f})" for k in LEGS))
        g = res["gain"]
        print(f"  {'跨市場增益（− 單市場）':18s}" + "".join(
            f"  {g[f'{m}_{k}']['mean'] * sc:+.4f}({g[f'{m}_{k}']['p']:.3f})" for k in LEGS))
        print()
    print("隔夜占比（Σ a_gap / Σ a_tot）：" + "；".join(
        f"{n} {r['share_overnight']:.1%}" for n, r in res["methods"].items())
        + f"；跨市場增益 {res['gain']['share_overnight']:.1%}")
    print("判讀：")
    for k, v in res["judgement"].items():
        print(f"  {k}：" + "；".join(v.values()))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        report(json.loads(OUT_JSON.read_text()))
        return
    if args.smoke:
        rng = np.random.default_rng(0)
        M = {}
        for name, *_ in METHODS:
            d = {f"{m}_{k}": rng.normal(0.05, 0.2, 246) for m in ("ic", "ric") for k in LEGS}
            d["a_gap"], d["a_int"] = rng.normal(1e-3, 3e-3, 246), rng.normal(2e-4, 3e-3, 246)
            d["a_tot"] = d["a_gap"] + d["a_int"]
            d["n"] = 1
            M[name] = d
        report(compute(np.array(["2000-01-01"] * 246), M))
        print("\n冒煙測試完成（隨機數字，沒有讀任何預測，沒有寫檔）")
        return
    if args.write and OUT_JSON.exists():
        raise SystemExit(f"{OUT_JSON.relative_to(ROOT)} 已存在，--write 只能跑一次；請用 --report")

    tickers = sorted(set().union(*SL.build_groups().values()))
    dates, M, gates = load(tw_legs(tickers))
    verify(M, gates)
    if args.check:
        return
    res = compute(dates, M)
    OUT_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    report(res)
    print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
