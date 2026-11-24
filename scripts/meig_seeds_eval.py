"""
meig_seeds_eval.py — MEIG [1] 補到 10 顆種子之後的彙整（proposal_1209 §8）

登記：docs/proposal_1209_revision.md §8 附錄（訓練前寫下）。run 由 scripts/run_meig_seeds_queue.py 產生，
MEIG 的方法不動，每個設定只換種子。舊種子 {7, 42, 123}，新種子 {11, 99, 314, 555, 777, 888, 2026}。

    第一折主設定        tw50_bl_meig          （主表）
    第二折主設定        f2_bl_meig            （主表第二折）
    第一折美股落後 l 天  tw50_bl_meigusl{l}    （l = 2、3、4，§2）

輸出（登記第 4 節）：
    每個設定 10 顆的 IC、RankIC（種子平均 ± 種子標準差），逐日 IC 對 0 的 HAC 雙尾 p
        第一折用 results_table.paired_daily；第二折照 results_table._fold2_rows 的規則（只取 ≤ 2024-12-25、
        50 檔 y 或 y_hat 完全相同的日子略過、各 run 取共同日子、與 MAGNET 對齊日子、常態近似的 HAC）
    描述：舊 3 顆與新 7 顆分開的平均
    主表兩折：MAGNET − MEIG 的逐日 IC、RankIC 差與 HAC p
    落後版：落後 2 / 3 / 4 天 IC 對 0 的 HAC p，三個一起 Holm
檢查：只用舊 3 顆時要重現登記第 0 節的已知值；新 run 的種子與天數要齊全。
--write：寫 docs/figures/data/eval_meig_10seeds.json，供 gen_discussion_figures.py 畫圖。只執行一次。

    .venv/bin/python scripts/meig_seeds_eval.py [--write]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import results_table as R  # noqa: E402
from input_lag_eval import holm, vs_zero  # noqa: E402

OLD = ("7", "42", "123")
NEW = ("11", "99", "314", "555", "777", "888", "2026")
ALL = tuple(sorted(OLD + NEW, key=int))
LAGS = (2, 3, 4)
N_DAYS_F1 = 246
F2_CUT = "2024-12-25"
OUT_JSON = ROOT / "docs" / "figures" / "data" / "eval_meig_10seeds.json"

# 登記第 0 節：只用舊 3 顆時的已知值（4 位小數）
KNOWN = {
    "fold1": dict(ic=0.0061, ic_sd=0.0130, ric=0.0160, ric_sd=0.0045, p0=0.40),
    "fold2": dict(ic=-0.0077, ic_sd=0.0102, ric=-0.0039, ric_sd=0.0112),
    "lag2": dict(ic=0.0085, ric=0.0148, p0=0.19),
    "lag3": dict(ic=-0.0086, ric=-0.0024, p0=0.33),
    "lag4": dict(ic=0.0007, ric=0.0043, p0=0.92),
}
# 登記第 5 節：新 7 顆與舊 3 顆的平均相差超過舊 3 顆種子標準差的 2 倍時加註（只對兩折主設定）
GAP = {"fold1": 0.026, "fold2": 0.020}


# ---------------------------------------------------------------- 第一折

def f1_stats(arm: str, seeds=ALL):
    fs = R.find_seeds
    R.find_seeds = lambda a, _f=fs: {s: d for s, d in _f(a).items() if s in seeds}
    try:
        return R.neural_stats(arm)
    finally:
        R.find_seeds = fs


# ---------------------------------------------------------------- 第二折（規則同 results_table._fold2_rows）

def _seed_of(path: str) -> str:
    return re.search(r"_s(\d+)/predictions/", path).group(1)


def f2_load(pattern: str, seeds=None):
    """回傳 (日期, 種子列, P[run, 指標(0=IC, 1=RankIC), 日])。"""
    per, ss = [], []
    for f in sorted(glob.glob(str(ROOT / pattern), recursive=True)):
        s = _seed_of(f)
        if seeds is not None and s not in seeds:
            continue
        df = pd.read_csv(f)
        o = {}
        for d, g in df.groupby("target_date"):
            d = str(d)[:10]
            if d > F2_CUT or g.y.nunique() == 1 or g.y_hat.nunique() == 1:
                continue
            o[d] = (np.corrcoef(g.y_hat, g.y)[0, 1], stats.spearmanr(g.y_hat, g.y).statistic)
        if o:
            per.append(o)
            ss.append(s)
    k = sorted(set.intersection(*[set(x) for x in per]))
    return k, ss, np.array([[[x[d][i] for d in k] for i in (0, 1)] for x in per])


def f2_hac(d) -> float:
    """與 results_table._fold2_rows 內的 _hac 相同。"""
    n = len(d)
    L = int(4 * (n / 100) ** (2 / 9))
    s2 = np.var(d, ddof=1)
    for l in range(1, L + 1):
        s2 += 2 * (1 - l / (L + 1)) * np.cov(d[l:], d[:-l], ddof=1)[0, 1]
    return float(2 * (1 - stats.norm.cdf(abs(d.mean() / (s2 / n) ** 0.5))))


def f2_stats(seeds=ALL):
    kb, _, PB = f2_load(f"runs/**/*{R.BEST_F2}_s*/predictions/test_predictions.csv")
    k, ss, P = f2_load("runs/**/*f2_bl_meig_s*/predictions/test_predictions.csv", seeds)
    VB, V = np.nanmean(PB, axis=0), np.nanmean(P, axis=0)
    ii = [k.index(d) for d in k if d in set(kb)]
    jj = [kb.index(d) for d in k if d in set(kb)]
    out = dict(n=P.shape[0], seeds=sorted(ss, key=int), n_days=len(ii), n_days_magnet=len(kb))
    for idx, key in ((0, "ic"), (1, "ric")):
        per = np.nanmean(P[:, idx][:, ii], axis=1)
        out[key] = float(np.nanmean(V[idx][ii]))
        out[key + "_sd"] = float(np.std(per, ddof=1))
        out["per_" + key] = {s: float(v) for s, v in zip(ss, per)}
        z = V[idx][ii]
        out[key + "_p0"] = f2_hac(z[np.isfinite(z)])
        dd = VB[idx][jj] - V[idx][ii]
        dd = dd[np.isfinite(dd)]
        out["magnet_minus_" + key] = float(dd.mean())
        out["magnet_minus_" + key + "_p"] = f2_hac(dd)
    return out


# ---------------------------------------------------------------- 彙整

def summary_f1(st: dict) -> dict:
    o = dict(n=st["n"], seeds=list(st["seeds"]), ic=st["ic"], ic_sd=st["ic_sd"], ric=st["ric"], ric_sd=st["ric_sd"],
             per_ic={s: float(v) for s, v in st["per_ic"].items()},
             per_ric={s: float(v) for s, v in st["per_ric"].items()})
    o["ic_p0"] = vs_zero(st, "ic")[1]
    o["ric_p0"] = vs_zero(st, "ric")[1]
    return o


def split(per: dict) -> dict:
    a = [per[s] for s in OLD]
    b = [per[s] for s in NEW]
    return dict(old3=float(np.mean(a)), old3_sd=float(np.std(a, ddof=1)), new7=float(np.mean(b)),
                new7_sd=float(np.std(b, ddof=1)), gap=float(np.mean(b) - np.mean(a)))


def check_known(name: str, got: dict, problems: list) -> None:
    for k, v in KNOWN[name].items():
        g = got[k]
        nd = 2 if k == "p0" else 4
        if round(g, nd) != round(v, nd):
            problems.append(f"{name} 舊 3 顆 {k} = {g:.4f}，已知 {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"寫 {OUT_JSON.relative_to(ROOT)}")
    args = ap.parse_args()
    problems: list[str] = []

    # 1) 只用舊 3 顆：重現已知值
    old = {"fold1": f1_stats("tw50_bl_meig", OLD), **{f"lag{l}": f1_stats(f"tw50_bl_meigusl{l}", OLD) for l in LAGS}}
    for name, st in old.items():
        got = dict(ic=st["ic"], ic_sd=st["ic_sd"], ric=st["ric"], ric_sd=st["ric_sd"], p0=vs_zero(st, "ic")[1])
        check_known(name, got, problems)
    f2o = f2_stats(OLD)
    check_known("fold2", f2o, problems)

    # 2) 10 顆
    f1 = {"fold1": f1_stats("tw50_bl_meig"), **{f"lag{l}": f1_stats(f"tw50_bl_meigusl{l}") for l in LAGS}}
    for name, st in f1.items():
        if tuple(sorted(st["seeds"], key=int)) != ALL or st["n"] != 10:
            problems.append(f"{name}: 種子 {st['seeds']}（n {st['n']}）")
        if len(st["daily_ic"]) != N_DAYS_F1:
            problems.append(f"{name}: {len(st['daily_ic'])} 天")
    f2 = f2_stats()
    if tuple(f2["seeds"]) != ALL:
        problems.append(f"fold2: 種子 {f2['seeds']}")
    if f2["n_days"] != f2["n_days_magnet"]:
        problems.append(f"fold2: MEIG 與 MAGNET 對齊的日子 {f2['n_days']}，MAGNET {f2['n_days_magnet']}")
    if problems:
        raise SystemExit("檢查不通過：\n  " + "\n  ".join(problems))
    print("檢查通過：舊 3 顆重現登記第 0 節的已知值；五個設定都是 10 顆種子，"
          f"第一折 {N_DAYS_F1} 天、第二折 {f2['n_days']} 天。\n")

    base = R.neural_stats(R.BEST)
    res = dict(seeds_old=list(OLD), seeds_new=list(NEW), settings={}, tests={})
    S = {name: summary_f1(st) for name, st in f1.items()}
    S["fold2"] = {k: f2[k] for k in ("n", "seeds", "ic", "ic_sd", "ric", "ric_sd", "per_ic", "per_ric",
                                      "ic_p0", "ric_p0", "n_days")}
    for name in S:
        S[name]["split_ic"] = split(S[name]["per_ic"])
        S[name]["split_ric"] = split(S[name]["per_ric"])
    res["settings"] = S

    print("10 顆種子。IC / RankIC = 種子平均 ± 種子標準差；p0 = 逐日序列對 0 的 HAC 雙尾 p")
    lab = {"fold1": "第一折主設定", "fold2": "第二折主設定", **{f"lag{l}": f"第一折落後 {l} 天" for l in LAGS}}
    for name in ("fold1", "fold2", *[f"lag{l}" for l in LAGS]):
        s = S[name]
        print(f"  {lab[name]:12s} IC {s['ic']:+.4f} ± {s['ic_sd']:.4f}（p0 {s['ic_p0']:.4f}）   "
              f"RankIC {s['ric']:+.4f} ± {s['ric_sd']:.4f}（p0 {s['ric_p0']:.4f}）")
    print("\n逐種子 IC")
    for name in S:
        print(f"  {lab[name]:12s} " + "  ".join(f"s{k} {v:+.4f}" for k, v in
                                               sorted(S[name]["per_ic"].items(), key=lambda kv: int(kv[0]))))

    print("\n描述：舊 3 顆 {7, 42, 123} 與新 7 顆的 IC 平均（± 種子標準差）")
    for name in S:
        sp = S[name]["split_ic"]
        flag = ""
        if name in GAP:
            hit = abs(sp["gap"]) > GAP[name]
            res["tests"][f"{name}_gap_note"] = dict(gap=sp["gap"], threshold=GAP[name], note=hit)
            flag = f"   |差| {'>' if hit else '≤'} {GAP[name]} → {'加註' if hit else '不加註'}"
        print(f"  {lab[name]:12s} 舊 {sp['old3']:+.4f} ± {sp['old3_sd']:.4f}   新 {sp['new7']:+.4f} ± "
              f"{sp['new7_sd']:.4f}   新 − 舊 {sp['gap']:+.4f}{flag}")

    print("\n主表兩折：MAGNET − MEIG（逐日差的 HAC p）")
    for key, nm in (("ic", "IC"), ("ric", "RankIC")):
        d1, p1 = R.paired_daily(base, f1["fold1"], key)
        res["tests"][f"fold1_magnet_minus_meig_{key}"] = dict(diff=d1, p=p1)
        res["tests"][f"fold2_magnet_minus_meig_{key}"] = dict(diff=f2["magnet_minus_" + key],
                                                              p=f2["magnet_minus_" + key + "_p"])
        print(f"  {nm:7s} 第一折 {d1:+.4f}（p {p1:.4f}）   第二折 {f2['magnet_minus_' + key]:+.4f}"
              f"（p {f2['magnet_minus_' + key + '_p']:.4f}）")

    print("\n落後版：逐日 IC 對 0（HAC；落後 2–4 天三個一起 Holm）")
    ps = [S[f"lag{l}"]["ic_p0"] for l in LAGS]
    for l, p, ph in zip(LAGS, ps, holm(ps)):
        res["tests"][f"meig_lag{l}_vs_0"] = dict(mean=S[f"lag{l}"]["ic"], p=p, p_holm=ph)
        print(f"  l = {l}: IC {S[f'lag{l}']['ic']:+.4f}（p {p:.4f}，Holm {ph:.4f}）")
    res["tests"]["meig_lag1_vs_0"] = dict(mean=S["fold1"]["ic"], p=S["fold1"]["ic_p0"])

    if args.write:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
        print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
