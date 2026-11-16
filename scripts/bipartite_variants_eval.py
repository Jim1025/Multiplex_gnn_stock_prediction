"""
bipartite_variants_eval.py — 評估議題 2、3：[24] 換來源市場與輸入落後天數的結果彙整（第一折）

登記：docs/proposal_1209_revision.md §1.8 附錄的 B、C 節（計算前寫下）。只做第一折，描述性分析。
run 由 scripts/bipartite_screen.py 產生（方法不動，只換預測特徵）：

    us      主表的 [24]：來源 = 美股，落後 1 天        *_bipartite_rolling_t2_*
    tw      同市場對照（原文 §5.2.2）：來源 = 台股      *_bipartite_srcTW_rolling_t2_*
    both    來源 = 美股 + 台股（原文 Appendix E）       *_bipartite_srcUSTW_rolling_t2_*
    lag2–4  輸入時效（原文 §5.4）：來源 = 美股，落後 l 天  *_bipartite_lag{l}_rolling_t2_*

輸出：
    10 個模型在每個變體的 IC / RankIC，以及 10 個模型的中位數（原文的時效圖報中位數）
    LASSO：us − tw、both − us 的逐日 IC 差與 HAC p；落後 l 天的逐日 IC 對 0 的 HAC p
    對照：MAGNET 主結果與單市場消融（種子平均）
--write：寫 docs/figures/data/eval_bipartite_variants.json，供 gen_discussion_figures.py 畫圖。

HAC 與資料讀法都用 results_table.py 的（paired_daily、nonneural_stats）。不需要 torch。

用法：
    .venv/bin/python scripts/bipartite_variants_eval.py [--write]
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

VARIANTS = {"us": "*_bipartite_rolling_t2_{}",
            "tw": "*_bipartite_srcTW_rolling_t2_{}",
            "both": "*_bipartite_srcUSTW_rolling_t2_{}",
            "lag2": "*_bipartite_lag2_rolling_t2_{}",
            "lag3": "*_bipartite_lag3_rolling_t2_{}",
            "lag4": "*_bipartite_lag4_rolling_t2_{}"}
MODELS = ["OLS", "LASSO", "RIDGE", "RF", "SVM", "AdaBoost", "XGBoost", "LGBM", "ens-avg", "ens-med"]
OUT_JSON = ROOT / "docs" / "figures" / "data" / "eval_bipartite_variants.json"


def vs_zero(st: dict, key: str = "ic"):
    """逐日序列對 0 的 HAC 檢定，回傳 (平均, p)。"""
    zero = {"daily_" + key: np.zeros_like(st["daily_" + key])}
    return R.paired_daily(st, zero, key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"寫 {OUT_JSON.relative_to(ROOT)}")
    args = ap.parse_args()

    S = {v: {m: R.nonneural_stats(pat.format(m)) for m in MODELS} for v, pat in VARIANTS.items()}
    missing = [f"{v}/{m}" for v in S for m in MODELS if S[v][m] is None]
    if missing:
        raise SystemExit(f"缺少 run：{missing}")
    n_days = {len(S[v][m]["daily_ic"]) for v in S for m in MODELS}
    if n_days != {246}:
        raise SystemExit(f"天數不一致：{n_days}")

    res = dict(fold=1, n_days=246, models={}, median={}, tests={}, magnet={})
    for key, lab in (("ic", "IC"), ("ric", "RankIC")):
        print(f"\n{lab}（第一折 test，246 天）")
        print(f"  {'model':10s}" + "".join(f"{v:>9s}" for v in VARIANTS))
        for m in MODELS:
            vals = {v: S[v][m][key] for v in VARIANTS}
            res["models"].setdefault(m, {})[key] = vals
            print(f"  {m:10s}" + "".join(f"{vals[v]:+9.4f}" for v in VARIANTS))
        med = {v: float(np.median([S[v][m][key] for m in MODELS])) for v in VARIANTS}
        res["median"][key] = med
        print(f"  {'10 個中位數':8s}" + "".join(f"{med[v]:+9.4f}" for v in VARIANTS))

    L = {v: S[v]["LASSO"] for v in VARIANTS}
    print("\nLASSO 的逐日檢定（HAC）")
    for name, a, b in (("us − tw（[24] 的跨市場資訊價值）", "us", "tw"),
                       ("both − us（加入台股的增量）", "both", "us"),
                       ("both − tw", "both", "tw")):
        for key, lab in (("ic", "IC"), ("ric", "RankIC")):
            d, p = R.paired_daily(L[a], L[b], key)
            res["tests"][f"{a}-{b}:{key}"] = dict(diff=d, p=p)
            print(f"  {name:32s} {lab:6s} {d:+.4f}（p {p:.4f}）")
    for l in (2, 3, 4):
        v = f"lag{l}"
        mu, p0 = vs_zero(L[v])
        d, p = R.paired_daily(L["us"], L[v], "ic")
        res["tests"][f"{v}:ic_vs_0"] = dict(mean=mu, p=p0)
        res["tests"][f"us-{v}:ic"] = dict(diff=d, p=p)
        print(f"  落後 {l} 天：IC {mu:+.4f}，對 0 的 p {p0:.4f}；落後 1 天 − 落後 {l} 天 {d:+.4f}（p {p:.4f}）")

    print("\n對照：MAGNET（種子平均；與 LASSO 的差不在登記的判準內，只作描述）")
    for name, arm in (("MAGNET（美股 + 台股）", "tw50_rgcnSA"), ("MAGNET-single market（只用台股）", "tw50_smktSA")):
        st = R.neural_stats(arm)
        res["magnet"][arm] = dict(ic=st["ic"], ic_sd=st["ic_sd"], ric=st["ric"], ric_sd=st["ric_sd"], n=st["n"])
        print(f"  {name:28s} IC {st['ic']:+.4f}  RankIC {st['ric']:+.4f}  n={st['n']}")
    sa, sm = R.neural_stats("tw50_rgcnSA"), R.neural_stats("tw50_smktSA")
    for name, a, b in (("MAGNET − LASSO(both)", sa, L["both"]), ("MAGNET-single market − LASSO(tw)", sm, L["tw"])):
        d, p = R.paired_daily(a, b, "ic")
        dr, pr = R.paired_daily(a, b, "ric")
        res["tests"][name] = dict(ic_diff=d, ic_p=p, ric_diff=dr, ric_p=pr)
        print(f"  {name:32s} IC {d:+.4f}（p {p:.4f}）  RankIC {dr:+.4f}（p {pr:.4f}）")

    if args.write:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
        print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
