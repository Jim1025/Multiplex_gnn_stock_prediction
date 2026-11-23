"""
us_only_eval.py — 評估議題 2：MAGNET 只用美股的彙整與檢定（第一折）

登記：docs/proposal_1209_revision.md §7 附錄（訓練前寫下）。只做第一折，描述性分析；--write 只跑一次。

方法與資料（讀法與主表 results_table.py 相同；MAGNET 各版本是種子平均的逐日 IC，LASSO 是單一 run）：
    MAGNET（台股 + 美股）      tw50_rgcnSA（10 顆）
    MAGNET（只用美股）         tw50_rgcnSAusonly（10 顆，本次訓練）
    MAGNET-single market      tw50_smktSA（10 顆，只用台股）
    LASSO [24] 只用美股        runs/linear/20261021_2130_bipartite_rolling_t2_LASSO（主表）
    LASSO [24] 台股 + 美股     runs/linear/20261116_1321_bipartite_srcUSTW_rolling_t2_LASSO（Appendix E）
    LASSO [24] 只用台股        runs/linear/20261116_1316_bipartite_srcTW_rolling_t2_LASSO（§5.2.2）

三個比較（逐日序列，results_table.paired_daily 的 HAC 雙尾 p；IC 三個一起 Holm，RankIC 另一族）：
    (a) MAGNET(台股 + 美股) − MAGNET(只用美股)
    (b) [MAGNET(台股 + 美股) − MAGNET(只用美股)] − [LASSO(台股 + 美股) − LASSO(只用美股)]
    (c) MAGNET(只用美股) − LASSO(只用美股)
描述：MAGNET(只用美股) − 單市場；只用美股低於同種子 SA 的種子數。

檢查（登記之外另加，只用已知的數字）：既有 arm 的全期 IC 必須重現 proposal_1209 §1.3 的值，日期必須一致。

    .venv/bin/python -I scripts/us_only_eval.py --write     # 只跑一次；結果檔已存在就拒絕
    .venv/bin/python -I scripts/us_only_eval.py --report    # 之後只讀結果檔重印
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import results_table as R  # noqa: E402

OUT_JSON = ROOT / "docs" / "figures" / "data" / "eval_us_only.json"
NEURAL = {"MAGNET (TW + US)": "tw50_rgcnSA", "MAGNET (US only)": "tw50_rgcnSAusonly",
          "MAGNET (TW only)": "tw50_smktSA"}
LASSO = {"LASSO (US only)": "20261021_2130_bipartite_rolling_t2_LASSO",
         "LASSO (TW + US)": "20261116_1321_bipartite_srcUSTW_rolling_t2_LASSO",
         "LASSO (TW only)": "20261116_1316_bipartite_srcTW_rolling_t2_LASSO"}
EXPECT = {"MAGNET (TW + US)": 0.1127, "MAGNET (TW only)": 0.0250,
          "LASSO (US only)": 0.1042, "LASSO (TW + US)": 0.0987, "LASSO (TW only)": 0.0225}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


def load() -> dict:
    S, dates = {}, None
    for name, arm in NEURAL.items():
        st = R.neural_stats(arm)
        if st is None:
            raise SystemExit(f"找不到 {arm}")
        st["best_val"] = [json.loads((Path(d) / "meta.json").read_text()).get("best_val_IC")
                          for _, d in sorted(R.find_seeds(arm).items(), key=lambda kv: int(kv[0]))]
        S[name] = st
    for name, run in LASSO.items():
        st = R.nonneural_stats(run)
        if st is None:
            raise SystemExit(f"找不到 {run}")
        S[name] = st
    print("檢查：")
    for name, st in S.items():
        d = np.asarray([str(x)[:10] for x in st["dates"]])
        if dates is None:
            dates = d
        elif not np.array_equal(dates, d):
            raise SystemExit(f"{name} 的日期與其他方法不同")
        ic = float(np.nanmean(st["daily_ic"]))
        exp = EXPECT.get(name)
        good = exp is None or abs(ic - exp) < 5e-5
        print(f"  {name:20s} 逐日 IC 平均 {ic:+.4f}" + (f"（已知 {exp:+.4f}）" if exp is not None else "（本次）")
              + f"  n={st['n']}" + ("" if good else "  <-- 不符"))
        if not good:
            raise SystemExit("既有 arm 與已知數字不符，停止")
    usd = sorted(R.find_seeds("tw50_rgcnSAusonly"), key=int)
    if usd != sorted(R.find_seeds("tw50_rgcnSA"), key=int) or len(usd) != 10:
        raise SystemExit(f"只用美股的種子與 SA 不同：{usd}")
    return S


def diff(a: dict, b: dict) -> dict:
    return {"daily_ic": np.asarray(a["daily_ic"]) - np.asarray(b["daily_ic"]),
            "daily_ric": np.asarray(a["daily_ric"]) - np.asarray(b["daily_ric"])}


def compute(S: dict) -> dict:
    full, uso, smk = S["MAGNET (TW + US)"], S["MAGNET (US only)"], S["MAGNET (TW only)"]
    lu, lb = S["LASSO (US only)"], S["LASSO (TW + US)"]
    pairs = {"a": (full, uso), "b": (diff(full, uso), diff(lb, lu)), "c": (uso, lu)}
    res = dict(fold=1, n_days=int(len(full["dates"])), arms={}, contrasts={}, descriptive={})
    for name, st in S.items():
        res["arms"][name] = dict(ic=float(st["ic"]), ic_sd=None if np.isnan(st["ic_sd"]) else float(st["ic_sd"]),
                                 ric=float(st["ric"]), ric_sd=None if np.isnan(st["ric_sd"]) else float(st["ric_sd"]),
                                 n=int(st["n"]))
        if "per_ic" in st and st.get("seeds"):
            res["arms"][name]["per_seed_ic"] = {s: float(v) for s, v in st["per_ic"].items()}
        if "best_val" in st:
            res["arms"][name]["best_val_ic_mean"] = float(np.mean(st["best_val"]))
    for key in ("ic", "ric"):
        rows = {k: dict(zip(("diff", "p"), R.paired_daily(a, b, key))) for k, (a, b) in pairs.items()}
        for k, ph in zip(rows, holm([rows[k]["p"] for k in rows])):
            rows[k]["p_holm"] = ph
        res["contrasts"][key] = rows
        d, p = R.paired_daily(uso, smk, key)
        res["descriptive"][f"US only - TW only ({key})"] = dict(diff=d, p=p)
    lower = [s for s in uso["per_ic"] if uso["per_ic"][s] < full["per_ic"][s]]
    res["descriptive"]["seeds US only < SA"] = dict(n=len(lower), of=len(uso["per_ic"]), seeds=sorted(lower, key=int))
    return res


WORDING = {
    "b": ("MAGNET 比 [24] 更能利用台股自身的資訊", "加入台股自身資訊的增量，MAGNET 與 [24] 分不出差別",
          "[24] 加入台股的增量較大"),
    "a": ("對 MAGNET，台股自身的資訊在美股之外仍有增量", "對 MAGNET，加入台股自身的資訊分不出增量", "加入台股反而較差"),
    "c": ("同樣只用美股時，MAGNET 勝過 [24]", "同樣只用美股時分不出差別", "同樣只用美股時 [24] 較高"),
}


def report(res: dict) -> None:
    print(f"\n第一折 test，{res['n_days']} 天")
    for name, a in res["arms"].items():
        sd = f" ± {a['ic_sd']:.4f}" if a["ic_sd"] is not None else ""
        rsd = f" ± {a['ric_sd']:.4f}" if a["ric_sd"] is not None else ""
        bv = f"  最佳 val IC 平均 {a['best_val_ic_mean']:.4f}" if "best_val_ic_mean" in a else ""
        print(f"  {name:20s} IC {a['ic']:+.4f}{sd}  RankIC {a['ric']:+.4f}{rsd}  n={a['n']}{bv}")
    seeds = res["arms"]["MAGNET (US only)"].get("per_seed_ic", {})
    print("  只用美股逐種子 IC：" + "  ".join(f"s{s} {v:+.4f}" for s, v in seeds.items()))
    lab = {"a": "(a) MAGNET(TW+US) − MAGNET(US only)", "b": "(b) [MAGNET 增量] − [LASSO 增量]",
           "c": "(c) MAGNET(US only) − LASSO(US only)"}
    for key, kl in (("ic", "IC"), ("ric", "RankIC")):
        print(f"\n比較（{kl}；HAC 雙尾 p，三個一起 Holm）：")
        for k, r in res["contrasts"][key].items():
            print(f"  {lab[k]:40s} {r['diff']:+.4f}  p {r['p']:.4f}  Holm {r['p_holm']:.4f}")
    print("\n描述（未校正）：")
    for k, v in res["descriptive"].items():
        print(f"  {k}: " + ", ".join(f"{kk}={vv:+.4f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items()))
    print("\n寫法（依登記第 6 節，以 IC 為準）：")
    for k in ("b", "a", "c"):
        r, rr = res["contrasts"]["ic"][k], res["contrasts"]["ric"][k]
        w = WORDING[k][0] if (r["diff"] > 0 and r["p_holm"] < 0.05) else \
            WORDING[k][2] if (r["diff"] < 0 and r["p_holm"] < 0.05) else WORDING[k][1]
        note = "（RankIC 方向相反，加註）" if np.sign(r["diff"]) != np.sign(rr["diff"]) else ""
        print(f"  ({k}) {w}{note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(json.loads(OUT_JSON.read_text()))
        return
    if OUT_JSON.exists():
        raise SystemExit(f"{OUT_JSON.relative_to(ROOT)} 已存在：登記規定只跑一次，請改用 --report")
    res = compute(load())
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    report(res)
    print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
