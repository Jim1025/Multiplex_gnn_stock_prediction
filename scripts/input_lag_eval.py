"""
input_lag_eval.py — 評估議題 3：MAGNET 與 MEIG 的美股輸入延遲版結果彙整（第一折）

登記：docs/proposal_1209_revision.md §2.8 附錄（訓練前寫下）。只做第一折，描述性分析。
run 由 scripts/run_input_lag_queue.py 產生（模型不動，只多 --us-lag l）：

    MAGNET      落後 1 天 = 主結果 tw50_rgcnSA（10 顆）；落後 l 天 = tw50_rgcnSAusl{l}（10 顆）
    MEIG [1]    落後 1 天 = 主表 tw50_bl_meig（3 顆）；落後 l 天 = tw50_bl_meigusl{l}（3 顆）
    對照        單市場消融 tw50_smktSA（不用美股，不受延遲影響）；LASSO [24] 的落後 1–4 天（proposal_1209 §1.4）

輸出（判讀規則見登記的第 3 節）：
    每個 arm 的 IC / RankIC（種子平均 ± 種子標準差）
    MAGNET (a) 落後 1 − 落後 l 的逐日 IC 差、HAC p、Holm
           (b) 落後 l − 單市場 的逐日 IC 差、HAC p、Holm
           (c) 同一個 l 下 MAGNET − LASSO [24] 的逐日 IC / RankIC 差、HAC p（只作描述）
           (d) 10 顆種子裡有幾顆的 IC 低於同種子的落後 1 天
    MEIG   各落後天數的逐日 IC 對 0 的 HAC p、Holm
--write：寫 docs/figures/data/eval_input_lag.json，供 gen_discussion_figures.py 畫圖。
--lag1-magnet / --lag1-meig：落後 1 天改用別的 arm（登記的備用方案：目前環境重訓的 tw50_rgcnSAusl1 / tw50_bl_meigusl1）。

HAC 與資料讀法都用 results_table.py 的（paired_daily、neural_stats、nonneural_stats）。不需要 torch。

用法：
    .venv/bin/python scripts/input_lag_eval.py [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import results_table as R  # noqa: E402

LAGS = (1, 2, 3, 4)
LASSO = {1: "*_bipartite_rolling_t2_LASSO", 2: "*_bipartite_lag2_rolling_t2_LASSO",
         3: "*_bipartite_lag3_rolling_t2_LASSO", 4: "*_bipartite_lag4_rolling_t2_LASSO"}
N_SEEDS = {"MAGNET": 10, "MEIG [1]": 3}
N_DAYS = 246
OUT_JSON = ROOT / "docs" / "figures" / "data" / "eval_input_lag.json"


def holm(ps: list[float]) -> list[float]:
    """Holm 校正後的 p（順序與輸入相同）。"""
    order = np.argsort(ps)
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def vs_zero(st: dict, key: str = "ic"):
    """逐日序列對 0 的 HAC 檢定，回傳 (平均, p)。"""
    return R.paired_daily(st, {"daily_" + key: np.zeros_like(st["daily_" + key])}, key)


def summary(st: dict) -> dict:
    return dict(ic=st["ic"], ic_sd=st["ic_sd"], ric=st["ric"], ric_sd=st["ric_sd"], n=st["n"],
                per_ic={s: float(v) for s, v in st["per_ic"].items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"寫 {OUT_JSON.relative_to(ROOT)}")
    ap.add_argument("--lag1-magnet", default="tw50_rgcnSA", help="MAGNET 落後 1 天用的 arm（預設主結果）")
    ap.add_argument("--lag1-meig", default="tw50_bl_meig", help="MEIG 落後 1 天用的 arm（預設主表）")
    ap.add_argument("--allow-partial", action="store_true", help="種子數不足時仍輸出（只供進度檢視，不可 --write）")
    args = ap.parse_args()
    if args.allow_partial and args.write:
        raise SystemExit("--allow-partial 不可與 --write 併用")

    arms = {"MAGNET": {1: args.lag1_magnet, **{l: f"tw50_rgcnSAusl{l}" for l in LAGS[1:]}},
            "MEIG [1]": {1: args.lag1_meig, **{l: f"tw50_bl_meigusl{l}" for l in LAGS[1:]}}}
    S: dict = {m: {} for m in arms}
    problems = []
    for m, by_lag in arms.items():
        for l, arm in by_lag.items():
            st = R.neural_stats(arm)
            if st is None:
                problems.append(f"{arm}: 沒有 run")
                continue
            if st["n"] != N_SEEDS[m]:
                problems.append(f"{arm}: {st['n']} 顆（應為 {N_SEEDS[m]}）")
            if len(st["daily_ic"]) != N_DAYS:
                problems.append(f"{arm}: {len(st['daily_ic'])} 天（應為 {N_DAYS}）")
            S[m][l] = st
    smkt = R.neural_stats("tw50_smktSA")
    lasso = {l: R.nonneural_stats(p) for l, p in LASSO.items()}
    if problems:
        msg = "run 不完整：\n  " + "\n  ".join(problems)
        if not args.allow_partial:
            raise SystemExit(msg)
        print(msg)

    res = dict(fold=1, n_days=N_DAYS, lags=list(LAGS), arms=arms, methods={}, tests={},
               single_market=summary(smkt),
               lasso={str(l): dict(ic=lasso[l]["ic"], ric=lasso[l]["ric"]) for l in LAGS})

    print(f"第一折 test，{N_DAYS} 天。IC / RankIC = 種子平均 ± 種子標準差\n")
    print(f"  {'':22s}" + "".join(f"{'落後 ' + str(l) + ' 天':>22s}" for l in LAGS))
    for m in arms:
        res["methods"][m] = {str(l): summary(S[m][l]) for l in S[m]}
        for key, sd, lab in (("ic", "ic_sd", "IC"), ("ric", "ric_sd", "RankIC")):
            print(f"  {m + ' ' + lab:22s}" + "".join(
                f"{S[m][l][key]:+13.4f} ± {S[m][l][sd]:.4f}" if l in S[m] else f"{'—':>22s}" for l in LAGS))
    for key, lab in (("ic", "IC"), ("ric", "RankIC")):
        print(f"  {'LASSO [24] ' + lab:22s}" + "".join(f"{lasso[l][key]:+13.4f}{'':9s}" for l in LAGS))
    print(f"  {'單市場消融（不用美股）':16s} IC {smkt['ic']:+.4f} ± {smkt['ic_sd']:.4f}   "
          f"RankIC {smkt['ric']:+.4f} ± {smkt['ric_sd']:.4f}")

    M = S["MAGNET"]
    lags = [l for l in LAGS[1:] if l in M]
    if 1 in M and lags:
        print("\nMAGNET (a) 落後 1 天 − 落後 l 天（逐日 IC，HAC；三個一起 Holm）")
        rows = [R.paired_daily(M[1], M[l], "ic") for l in lags]
        for l, (d, p), ph in zip(lags, rows, holm([r[1] for r in rows])):
            dr, pr = R.paired_daily(M[1], M[l], "ric")
            res["tests"][f"magnet_lag1-lag{l}"] = dict(ic_diff=d, ic_p=p, ic_p_holm=ph, ric_diff=dr, ric_p=pr)
            print(f"  l = {l}: IC {d:+.4f}（p {p:.4f}，Holm {ph:.4f}）   RankIC {dr:+.4f}（p {pr:.4f}）")

        print("\nMAGNET (b) 落後 l 天 − 單市場消融（逐日 IC，HAC；三個一起 Holm）")
        rows = [R.paired_daily(M[l], smkt, "ic") for l in lags]
        for l, (d, p), ph in zip(lags, rows, holm([r[1] for r in rows])):
            dr, pr = R.paired_daily(M[l], smkt, "ric")
            res["tests"][f"magnet_lag{l}-single_market"] = dict(ic_diff=d, ic_p=p, ic_p_holm=ph, ric_diff=dr, ric_p=pr)
            print(f"  l = {l}: IC {d:+.4f}（p {p:.4f}，Holm {ph:.4f}）   RankIC {dr:+.4f}（p {pr:.4f}）")

        print("\nMAGNET (c) 同一個 l 下 MAGNET − LASSO [24]（只作描述）")
        for l in [1, *lags]:
            d, p = R.paired_daily(M[l], lasso[l], "ic")
            dr, pr = R.paired_daily(M[l], lasso[l], "ric")
            res["tests"][f"magnet-lasso_lag{l}"] = dict(ic_diff=d, ic_p=p, ric_diff=dr, ric_p=pr)
            print(f"  l = {l}: IC {d:+.4f}（p {p:.4f}）   RankIC {dr:+.4f}（p {pr:.4f}）")

        print("\nMAGNET (d) 同種子比較：落後 l 天的 IC 低於落後 1 天的種子數")
        for l in lags:
            common = sorted(set(M[1]["per_ic"]) & set(M[l]["per_ic"]), key=int)
            lower = sum(M[l]["per_ic"][s] < M[1]["per_ic"][s] for s in common)
            res["tests"][f"magnet_seeds_lower_lag{l}"] = dict(lower=int(lower), n=len(common))
            print(f"  l = {l}: {lower} / {len(common)}")

    G = S["MEIG [1]"]
    if G:
        print("\nMEIG：逐日 IC 對 0（HAC；落後 2–4 天三個一起 Holm）。不對 MEIG 的時效作主張")
        ls = sorted(G)
        rows = {l: vs_zero(G[l]) for l in ls}
        hl = [l for l in ls if l != 1]
        ph = dict(zip(hl, holm([rows[l][1] for l in hl]))) if hl else {}
        for l in ls:
            mu, p = rows[l]
            res["tests"][f"meig_lag{l}_vs_0"] = dict(mean=mu, p=p, p_holm=ph.get(l))
            print(f"  l = {l}: IC {mu:+.4f}（p {p:.4f}" + (f"，Holm {ph[l]:.4f}）" if l in ph else "）"))

    if args.write:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
        print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
