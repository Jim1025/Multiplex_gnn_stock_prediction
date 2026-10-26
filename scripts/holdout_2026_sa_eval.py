"""
holdout_2026_sa_eval.py — 凍結的 SA（自身 + 同產業，§80）在 2026 窗的描述性評估（proposal §81）

**探索性，不是確認。** 2026 窗已在 §75 用過一次（投影後的關係型台股層，未通過）。這是同一個假設在同一段期間的
第二次檢驗，而且是看過第一次的結果之後才決定的。規則見 §81（reg_holdout_2026_sa.md，計算前寫下並記 SHA-256）。

沿用 scripts/holdout_2026_eval.py（§75 的登記版，未改）的預測、健全性與統計函式；主 arm 直接讀 §75 存下的 2026 預測。

用法：
    .venv/bin/python -I scripts/holdout_2026_sa_eval.py --check        # 只用既有資料：重評 SA 第一折 test，與存檔比對
    .venv/bin/python -I scripts/holdout_2026_sa_eval.py --validate-75  # 用 §75 存下的預測重現 §75 的數字
    .venv/bin/python -I scripts/holdout_2026_sa_eval.py --run          # 健全性通過才存 SA 的 2026 預測並報告
    .venv/bin/python -I scripts/holdout_2026_sa_eval.py --report       # 只讀存下的預測重算報告

只寫 SA run 目錄裡的 predictions/holdout2026_predictions.csv，不動其他檔案。
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

SA = "tw50_rgcnSA"
N_SEEDS = 10
N_LOOKS = 2              # 同一個假設在 2026 窗的第二次檢驗：p 一併報 Bonferroni x2（登記 §4）


def _load_saved(arm: str) -> dict[int, pd.DataFrame]:
    out = {}
    for s, d in H.runs(arm).items():
        f = d / "predictions" / H.OUT_NAME
        if f.exists():
            df = pd.read_csv(f, dtype={"ticker": str})
            df["target_date"] = df["target_date"].astype(str).str[:10]
            out[s] = df
    return out


def compare(pa: dict, pb: dict, la: str, lb: str, n_looks: int) -> dict:
    """b − a：逐種子 IC / RankIC 的同種子配對差、95% 信賴區間、配對 t p（與 Bonferroni x n_looks）與描述項。"""
    seeds = sorted(set(pa) & set(pb))
    Da = {s: H.daily(pa[s]) for s in seeds}
    Db = {s: H.daily(pb[s]) for s in seeds}
    days = sorted(set.intersection(*[set(D[s].index) for D in (Da, Db) for s in seeds]))
    print(f"\n######## {lb} − {la}：{days[0]} ~ {days[-1]}，{len(days)} 個快照日，同種子配對 {len(seeds)} 顆：{seeds}")
    if len(seeds) != N_SEEDS:
        print(f"    注意：規則寫的是 {N_SEEDS} 顆種子，實際 {len(seeds)} 顆")
    out = {}
    for k, lab in (("ic", "IC（主要）"), ("ric", "RankIC（第二）")):
        a = np.array([np.nanmean(Da[s].loc[days, k]) for s in seeds])
        b = np.array([np.nanmean(Db[s].loc[days, k]) for s in seeds])
        d = b - a
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        tq = stats.t.ppf(0.975, n - 1)
        p = stats.ttest_rel(b, a).pvalue
        pb_ = min(1.0, p * n_looks)
        out[k] = dict(d=float(d.mean()), lo=float(d.mean() - tq * se), hi=float(d.mean() + tq * se), p=float(p), p_adj=pb_)
        print(f"  {lab:14s} {la} {a.mean():+.4f} -> {lb} {b.mean():+.4f}  差 {d.mean():+.4f}（sd {d.std(ddof=1):.4f}，"
              f"{int((d > 0).sum())}/{n} 為正）；95% CI [{d.mean() - tq * se:+.4f}, {d.mean() + tq * se:+.4f}]；"
              f"配對 p {p:.4f}" + (f"，Bonferroni x{n_looks} {pb_:.4f}" if n_looks > 1 else ""))
    print("  一併報：")
    for k, lab in (("ic", "IC"), ("ric", "RankIC")):
        ma = np.nanmean(np.vstack([Da[s].loc[days, k] for s in seeds]), 0)
        mb = np.nanmean(np.vstack([Db[s].loc[days, k] for s in seeds]), 0)
        print(f"      種子平均的逐日 {lab} 差 {np.nanmean(mb - ma):+.4f}（逐日 HAC p {H.hac_p(mb - ma):.4f}）")
    for lab, P in ((la, pa), (lb, pb)):
        ens = (pd.concat([P[s] for s in seeds]).groupby(["target_date", "ticker"])
               .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
        e = H.daily(ens).loc[days]
        D = Da if P is pa else Db
        ir = np.mean([H.icir(D[s].loc[days, "ic"]) for s in seeds])
        rir = np.mean([H.icir(D[s].loc[days, "ric"]) for s in seeds])
        print(f"      {lab}：{len(seeds)} 顆集成 IC {np.nanmean(e.ic):+.4f}、RankIC {np.nanmean(e.ric):+.4f}；"
              f"ICIR {ir:.4f}、RankICIR {rir:.4f}（逐種子算完再平均）")
    print("      逐種子 IC（" + la + " / " + lb + "）：" + "；".join(
        f"s{s} {np.nanmean(Da[s].loc[days, 'ic']):+.4f} / {np.nanmean(Db[s].loc[days, 'ic']):+.4f}" for s in seeds))
    return out


def wording(res: dict) -> None:
    """規則 §6 的寫法表（各指標分開）。"""
    print("\n== 寫法（規則 §6；探索性的第二次檢驗，不是確認）")
    for k, lab in (("ic", "IC"), ("ric", "RankIC")):
        r = res[k]
        if r["lo"] > 0 and r["p_adj"] < 0.05:
            txt = f"2026 窗上 SA 的 {lab} 高於主 arm（探索性的第二次檢驗，不能當確認）"
        elif r["hi"] < 0:
            txt = f"2026 窗上 SA 的 {lab} 低於主 arm"
        else:
            txt = "2026 窗上看不出 SA 與主 arm 的差別（與 §75 投影後版本的結果對照）"
        print(f"    {lab:7s} {txt}")


def report(sa: dict) -> None:
    saved = H.load_saved()
    res = compare(saved["主 arm"], sa, "主 arm", "SA", N_LOOKS)
    print("\n  對照（不檢定）：§75 投影後關係型層在同一段期間")
    seeds = sorted(saved["關係型"])
    D = {s: H.daily(saved["關係型"][s]) for s in seeds}
    print(f"      IC {np.mean([np.nanmean(D[s].ic) for s in seeds]):+.4f}、RankIC {np.mean([np.nanmean(D[s].ric) for s in seeds]):+.4f}"
          f"（{len(seeds)} 顆，種子平均）")
    wording(res)


def main() -> None:
    ap = argparse.ArgumentParser(description="凍結的 SA 在 2026 窗的描述性評估（proposal §81）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--validate-75", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.validate_75:
        saved = H.load_saved()
        compare(saved["主 arm"], saved["關係型"], "主 arm", "關係型", 1)
        return
    if a.report:
        report(_load_saved(SA))
        return
    if a.check:
        worst = 0.0
        for s, d in H.runs(SA).items():
            df = H.predict(d, ROOT / "data/graphs/snapshots_tw50", ROOT / "data/features", None)
            m = H.max_diff_vs_stored(d, df, *H.FOLD1_TEST)
            worst = max(worst, m)
            print(f"  SA s{s}: 重評第一折 test 與存檔的最大差 {m:.2e}")
        print(f"\n--check：最大差 {worst:.2e}（門檻 {H.TOL}） -> {'通過' if worst <= H.TOL else '未通過'}")
        return
    # --run
    snap = H.HOLD / "graphs" / "snapshots_tw50"
    files = sorted(snap.glob("graph_*.pt"))
    val_end = sum(1 for f in files if f.stem.split("_")[1] < H.FOLD1_TEST[0])   # 從 2024-12-26 開始評估
    preds, worst, dirs = {}, 0.0, H.runs(SA)
    for s, d in dirs.items():
        df = H.predict(d, snap, H.HOLD / "features", val_end)
        worst = max(worst, H.max_diff_vs_stored(d, df, *H.FOLD1_TEST))
        preds[s] = df[(df.target_date >= H.H_START) & (df.target_date <= H.H_END)].reset_index(drop=True)
    print(f"健全性：holdout 資料重評 SA 第一折 test 與存檔的最大差 {worst:.2e}（門檻 {H.TOL}），{len(preds)} 顆")
    if worst > H.TOL or len(preds) != N_SEEDS:
        raise SystemExit("健全性未通過，停止；不印、不存任何 2026 的結果")
    for s, d in dirs.items():
        preds[s].to_csv(d / "predictions" / H.OUT_NAME, index=False)
    report(preds)


if __name__ == "__main__":
    main()
