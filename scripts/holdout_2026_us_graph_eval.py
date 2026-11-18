"""
holdout_2026_us_graph_eval.py — 美股層內圖與跨市場輸入在 2026 窗的檢驗（凍結的第一折模型，proposal_1209 §4）

規則：reg_holdout_2026_us_graph.md（計算前寫下並記 SHA-256，原文在 proposal_1209 §4 附錄）。
    比較 A（主要）  SA − noA1SA     美股層內圖（tw50_noA1SA：SA 加 --graph-ablate empty_both）
    比較 B          SA − 單市場     跨市場輸入（tw50_smktSA：SA 加 disable_a12）
SA 的 2026 預測直接讀 proposal_819 §81 存下的檔案，不重算。預測、健全性與統計函式沿用
scripts/holdout_2026_eval.py（§75 的登記版）與 scripts/holdout_2026_sa_eval.py（§81），兩者都只讀不改。

用法：
    .venv/bin/python -I scripts/holdout_2026_us_graph_eval.py --check        # 只用既有資料：重評兩個 arm 的第一折 test
    .venv/bin/python -I scripts/holdout_2026_us_graph_eval.py --validate-81  # 讀 §81 存下的預測重現 §81
    .venv/bin/python -I scripts/holdout_2026_us_graph_eval.py --run          # 健全性通過才存 2026 預測並報告
    .venv/bin/python -I scripts/holdout_2026_us_graph_eval.py --report       # 只讀存下的預測重算報告

只寫 noA1SA 與單市場 run 目錄裡的 predictions/holdout2026_predictions.csv，不動其他檔案。
"""

from __future__ import annotations

import argparse
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
import holdout_2026_eval as H  # noqa: E402  §75 的登記版，只讀不改
import holdout_2026_sa_eval as S  # noqa: E402  §81，只讀不改

SA = "tw50_rgcnSA"
ARMS = {"noA1SA": "tw50_noA1SA", "單市場": "tw50_smktSA"}
COMPARISONS = (("A（美股層內圖，主要）", "noA1SA"), ("B（跨市場輸入）", "單市場"))
SEEDS = [7, 11, 42, 99, 123, 314, 555, 777, 888, 2026]


def holm_adj(ps: list[float]) -> list[float]:
    """Holm 校正後的 p（順序與輸入相同）。"""
    order = np.argsort(ps)
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def compare(sa: dict, other: dict, lab_other: str) -> dict:
    """SA − other：種子層（同種子配對 t）與期間層（種子平均逐日差的 HAC），IC 與 RankIC 各自一族做 Holm。"""
    seeds = sorted(set(sa) & set(other))
    Da = {s: H.daily(sa[s]) for s in seeds}
    Db = {s: H.daily(other[s]) for s in seeds}
    days = sorted(set.intersection(*[set(D[s].index) for D in (Da, Db) for s in seeds]))
    print(f"  {days[0]} ~ {days[-1]}，{len(days)} 個快照日，同種子配對 {len(seeds)} 顆：{seeds}")
    res = {}
    for k in ("ic", "ric"):
        a = np.array([np.nanmean(Da[s].loc[days, k]) for s in seeds])
        b = np.array([np.nanmean(Db[s].loc[days, k]) for s in seeds])
        d = a - b
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        tq = stats.t.ppf(0.975, n - 1)
        ma = np.nanmean(np.vstack([Da[s].loc[days, k] for s in seeds]), 0)
        mb = np.nanmean(np.vstack([Db[s].loc[days, k] for s in seeds]), 0)
        res[k] = dict(sa=float(a.mean()), other=float(b.mean()), d=float(d.mean()), sd=float(d.std(ddof=1)),
                      pos=int((d > 0).sum()), n=n, lo=float(d.mean() - tq * se), hi=float(d.mean() + tq * se),
                      p_seed=float(stats.ttest_rel(a, b).pvalue),
                      d_daily=float(np.nanmean(ma - mb)), p_daily=H.hac_p(ma - mb))
    for layer in ("p_seed", "p_daily"):
        adj = holm_adj([res["ic"][layer], res["ric"][layer]])
        res["ic"][layer + "_holm"], res["ric"][layer + "_holm"] = adj
    for k, lab in (("ic", "IC"), ("ric", "RankIC")):
        r = res[k]
        print(f"  {lab:7s} SA {r['sa']:+.4f}、{lab_other} {r['other']:+.4f}；差 {r['d']:+.4f}（sd {r['sd']:.4f}，"
              f"{r['pos']}/{r['n']} 為正）；95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]")
        print(f"          種子層 配對 p {r['p_seed']:.2e}（Holm {r['p_seed_holm']:.2e}）；"
              f"期間層 逐日差 {r['d_daily']:+.4f}，HAC p {r['p_daily']:.4f}（Holm {r['p_daily_holm']:.4f}）")
    print("  一併報：")
    for lab, P in (("SA", sa), (lab_other, other)):
        ens = (pd.concat([P[s] for s in seeds]).groupby(["target_date", "ticker"])
               .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
        e = H.daily(ens).loc[days]
        D = Da if P is sa else Db
        ir = np.mean([H.icir(D[s].loc[days, "ic"]) for s in seeds])
        rir = np.mean([H.icir(D[s].loc[days, "ric"]) for s in seeds])
        print(f"      {lab}：{len(seeds)} 顆集成 IC {np.nanmean(e.ic):+.4f}、RankIC {np.nanmean(e.ric):+.4f}；"
              f"ICIR {ir:.4f}、RankICIR {rir:.4f}（逐種子算完再平均）")
    print(f"      逐種子 IC（SA / {lab_other}）：" + "；".join(
        f"s{s} {np.nanmean(Da[s].loc[days, 'ic']):+.4f} / {np.nanmean(Db[s].loc[days, 'ic']):+.4f}" for s in seeds))
    return res


def wording(name: str, res: dict) -> None:
    """規則 §6 的寫法表（IC、RankIC 分開）。"""
    for k, lab in (("ic", "IC"), ("ric", "RankIC")):
        r = res[k]
        seed_ok = r["lo"] > 0 and r["p_seed_holm"] < 0.05
        daily_ok = r["d_daily"] > 0 and r["p_daily_holm"] < 0.05
        if name == "noA1SA":
            if seed_ok and daily_ok:
                txt = "在沒看過的 2026 年，美股層內圖仍有增益，而且在逐日層級可以分辨"
            elif seed_ok:
                txt = ("在 2026 年，關掉美股層內圖後幾乎每顆種子都變差（方向與兩個 test 期一致），"
                       "但增益相對每天的波動很小，逐日檢定分不出")
            elif r["hi"] < 0:
                txt = "2026 年關掉美股層內圖反而較好"
            elif r["lo"] <= 0 <= r["hi"]:
                txt = "2026 年看不出美股層內圖的增益；論文不寫任何美股層內圖有貢獻的說法"
            else:
                txt = "種子層 95% CI 在 0 以上但 Holm 後 p ≥ 0.05：規則沒有對應的寫法，照實報數字"
        else:
            if seed_ok and daily_ok:
                txt = "在沒看過的 2026 年，跨市場輸入的增益仍然存在"
            else:
                txt = ("種子層" if not seed_ok else "") + ("、" if not seed_ok and not daily_ok else "") + \
                      ("期間層" if not daily_ok else "") + "不成立：照實寫，不作延伸解讀"
        print(f"    {lab:7s} {txt}")


def report(preds: dict[str, dict[int, pd.DataFrame]]) -> None:
    sa = S._load_saved(SA)
    if sorted(sa) != SEEDS:
        raise SystemExit(f"§81 存下的 SA 2026 預測不是 10 顆：{sorted(sa)}")
    out = {}
    for title, name in COMPARISONS:
        print(f"\n######## 比較 {title}：SA − {name}")
        out[name] = compare(sa, preds[name], name)
    print("\n== 寫法（規則 §6；任何結果都不能寫成「圖結構勝過現有方法」）")
    for title, name in COMPARISONS:
        print(f"  比較 {title}")
        wording(name, out[name])


def main() -> None:
    ap = argparse.ArgumentParser(description="美股層內圖與跨市場輸入在 2026 窗的檢驗（proposal_1209 §4）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--validate-81", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.validate_81:
        saved = H.load_saved()
        S.compare(saved["主 arm"], S._load_saved(SA), "主 arm", "SA", S.N_LOOKS)
        return
    if a.report:
        report({name: S._load_saved(arm) for name, arm in ARMS.items()})
        return
    dirs = {name: H.runs(arm) for name, arm in ARMS.items()}
    for name, d in dirs.items():
        if sorted(d) != SEEDS:
            raise SystemExit(f"{name} 的種子不是規則寫的 10 顆：{sorted(d)}")
        if a.run and any((p / "predictions" / H.OUT_NAME).exists() for p in d.values()):
            raise SystemExit(f"{name} 已有 2026 預測；--run 只跑一次，要重看請用 --report")
    if a.check:
        worst = 0.0
        for name, d in dirs.items():
            for s, p in d.items():
                df = H.predict(p, ROOT / "data/graphs/snapshots_tw50", ROOT / "data/features", None)
                m = H.max_diff_vs_stored(p, df, *H.FOLD1_TEST)
                worst = max(worst, m)
                print(f"  {name} s{s}: 重評第一折 test 與存檔的最大差 {m:.2e}")
        print(f"\n--check：最大差 {worst:.2e}（門檻 {H.TOL}） -> {'通過' if worst <= H.TOL else '未通過'}")
        return
    # --run
    snap = H.HOLD / "graphs" / "snapshots_tw50"
    files = sorted(snap.glob("graph_*.pt"))
    val_end = sum(1 for f in files if f.stem.split("_")[1] < H.FOLD1_TEST[0])   # 從 2024-12-26 開始評估
    preds, worst = {}, 0.0
    for name, d in dirs.items():
        preds[name] = {}
        for s, p in d.items():
            df = H.predict(p, snap, H.HOLD / "features", val_end)
            worst = max(worst, H.max_diff_vs_stored(p, df, *H.FOLD1_TEST))
            preds[name][s] = df[(df.target_date >= H.H_START) & (df.target_date <= H.H_END)].reset_index(drop=True)
    print(f"健全性：holdout 資料重評第一折 test 與存檔的最大差 {worst:.2e}（門檻 {H.TOL}）")
    if worst > H.TOL:
        raise SystemExit("健全性未通過，停止；不印、不存任何 2026 的結果")
    for name, d in dirs.items():
        for s, p in d.items():
            preds[name][s].to_csv(p / "predictions" / H.OUT_NAME, index=False)
    report(preds)


if __name__ == "__main__":
    main()
