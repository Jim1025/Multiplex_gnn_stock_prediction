"""
check_linkage_state.py — 教授 11-25 意見的檢驗 1：分組 × 美股漲跌幅五分位（第一折）

登記：docs/proposal_1209_revision.md §9 附錄（計算前寫下）。描述性分析，--write 只跑一次。

問題：議題 5「未配對 43 檔 IC 輸 [24]」與議題 4「Q5 IC 輸 [24]」是不是同一件事？

定義（與 §6、§1.2 相同，沿用 stock_linkage_eval.py 與 market_state_eval.py 的函式）：
    s_jt     z(ŷ)_jt · z(y)_jt（當天 50 檔橫截面標準化，ddof=0）；神經方法逐種子算再對種子平均
    c_gt     組 g 內 s_jt 的平均；IC_t = Σ_g (n_g/50) c_gt
    Δ_g(t)   c_g,MAGNET(t) − c_g,LASSO(t)
    五分位    |m(t)|，30 檔美股層節點在目標日之前最後一個交易日的 log_return 等權平均
主要量：
    S_E,Q    Σ_{t∈Q} (15/50)·Δ_E(t)，未配對電子在各五分位對 IC 差的總量；share_Q5 = S_E,Q5 / S_E,全期
    G_g      (n_g/50)·mean_{t∈Q5} Δ_g(t)；Σ_g G_g = Q5 的 MAGNET − LASSO 平均 IC 差

用法：
    .venv/bin/python -I scripts/check_linkage_state.py --smoke    # 隨機數字，不讀任何預測，只確認程式能跑完
    .venv/bin/python -I scripts/check_linkage_state.py --check    # 只做登記的檢查，不印分組結果
    .venv/bin/python -I scripts/check_linkage_state.py --write    # 只跑一次；結果檔已存在就拒絕
    .venv/bin/python -I scripts/check_linkage_state.py --report   # 之後只讀結果檔重印

不需要 torch：只讀各 run 的 predictions/test_predictions*.csv 與 data/features/adr/*.csv。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import market_state_eval as MS  # noqa: E402
import results_table as R  # noqa: E402
import stock_linkage_eval as SL  # noqa: E402

OUT_JSON = ROOT / "docs" / "figures" / "data" / "eval_check_linkage_state.json"
METHODS = [("MAGNET", "neural", "tw50_rgcnSA"),
           ("LASSO [24]", "nonneural", R.BIP.format("LASSO"))]
EXPECT_IC = {"MAGNET": 0.1127, "LASSO [24]": 0.1042}
EXPECT_Q = {"MAGNET": [0.0854, 0.0194, 0.1506, 0.1159, 0.1943],
            "LASSO [24]": [0.0480, 0.0146, 0.1171, 0.0925, 0.2532]}
EXPECT_DE = -0.033
GN = list(SL.GROUPS) + ["Unpaired (all 43)"]
E = "Unpaired electronics"
N_Q = 5


def load() -> tuple[np.ndarray, dict]:
    """(日期, {方法: {"ic": [天, 檔], "ric": [天, 檔], "tickers", "n", "ref_ic", "ref_ric"}})。"""
    dates, out = None, {}
    for name, kind, key in METHODS:
        parts = [SL.contributions(f) for f in SL.pred_files(kind, key)]
        d0, tk0 = parts[0][0], parts[0][1]
        if any(not np.array_equal(p[0], d0) or p[1] != tk0 for p in parts):
            raise ValueError(f"{name} 各種子的日期或股票不同")
        d = np.asarray([str(x)[:10] for x in d0])
        if dates is None:
            dates = d
        elif not np.array_equal(dates, d):
            raise ValueError(f"{name} 的日期與其他方法不同")
        st = R.neural_stats(key) if kind == "neural" else R.nonneural_stats(key)
        out[name] = dict(ic=SL.seed_mean(np.stack([p[2] for p in parts])),
                         ric=SL.seed_mean(np.stack([p[3] for p in parts])),
                         tickers=tk0, n=len(parts),
                         ref_ic=np.asarray(st["daily_ic"], float), ref_ric=np.asarray(st["daily_ric"], float))
    return dates, out


def group_series(M: dict, groups: dict[str, set[str]]) -> dict:
    """c[方法][指標][組] = 逐日序列。"""
    members = dict(groups)
    members["Unpaired (all 43)"] = set().union(*(groups[g] for g in SL.GROUPS[1:]))
    c = {}
    for name, d in M.items():
        idx = {g: [d["tickers"].index(t) for t in sorted(members[g])] for g in GN}
        c[name] = {k: {g: d[k][:, idx[g]].mean(axis=1) for g in GN} for k in ("ic", "ric")}
    return c, {g: len(members[g]) for g in GN}


def verify(M: dict, c: dict, n_g: dict, q: np.ndarray) -> None:
    """登記第 3 節的檢查；不過就停止，不印結果。"""
    print("檢查（登記第 3 節）：")
    ok_all = True
    for name, d in M.items():
        ic_t = sum(n_g[g] / 50 * c[name]["ic"][g] for g in SL.GROUPS)
        ric_t = sum(n_g[g] / 50 * c[name]["ric"][g] for g in SL.GROUPS)
        same_nan = (np.array_equal(np.isnan(ic_t), np.isnan(d["ref_ic"]))
                    and np.array_equal(np.isnan(ric_t), np.isnan(d["ref_ric"])))
        ok = ~np.isnan(d["ref_ic"])
        dmax = max(np.abs(ic_t[ok] - d["ref_ic"][ok]).max(), np.nanmax(np.abs(ric_t - d["ref_ric"])))
        ic = float(np.nanmean(ic_t))
        qm = [float(np.nanmean(ic_t[q == k])) for k in range(N_Q)]
        good = (same_nan and dmax <= 1e-9 and abs(ic - EXPECT_IC[name]) < 5e-5
                and all(abs(a - b) < 5e-5 for a, b in zip(qm, EXPECT_Q[name])))
        ok_all &= good
        print(f"  {name:12s} IC {ic:+.4f}（主表 {EXPECT_IC[name]:+.4f}）  逐日最大差 {dmax:.1e}  "
              f"五分位 " + " / ".join(f"{v:+.4f}" for v in qm) + f"  n={d['n']}{'' if good else '  <-- 不符'}")
    de = float(np.nanmean(c["MAGNET"]["ic"][E] - c["LASSO [24]"]["ic"][E]))
    good = abs(de - EXPECT_DE) < 5e-4
    ok_all &= good
    print(f"  未配對電子全期 Δ_E {de:+.4f}（§6.2 {EXPECT_DE:+.3f}）{'' if good else '  <-- 不符'}")
    if not ok_all:
        raise SystemExit("檢查未通過，停止（不印分組結果）")
    print("  通過\n")


def _cell(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return dict(mean=float(np.mean(x)) if len(x) else float("nan"), se=float("nan"), p=float("nan"), n=int(len(x)))
    return dict(mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))),
                p=float(stats.ttest_1samp(x, 0.0).pvalue), n=int(len(x)))


def compute(dates: np.ndarray, c: dict, n_g: dict, q: np.ndarray, qinfo: list) -> dict:
    res = dict(fold=1, n_days=int(len(dates)), start=str(dates[0]), end=str(dates[-1]),
               definition=("Delta_g(t) = c_g,MAGNET(t) - c_g,LASSO(t), c_g = mean over group g of z(yhat)*z(y) "
                           "(cross-sectional z over 50 stocks, ddof=0); quintiles of |m(t)| as in proposal_1209 §1.2"),
               quintiles=qinfo, groups=n_g, cells={}, S_E={}, G_Q5={}, judgement={})
    for k in ("ic", "ric"):
        res["cells"][k] = {}
        for g in GN:
            mag, las = c["MAGNET"][k][g], c["LASSO [24]"][k][g]
            res["cells"][k][g] = [dict(q=i + 1, magnet=_cell(mag[q == i]), lasso=_cell(las[q == i]),
                                       diff=_cell((mag - las)[q == i])) for i in range(N_Q)]
        # 主要量一：未配對電子的總量
        dE = n_g[E] / 50 * (c["MAGNET"][k][E] - c["LASSO [24]"][k][E])
        S = [float(np.nansum(dE[q == i])) for i in range(N_Q)]
        S_all = float(np.nansum(dE))
        res["S_E"][k] = dict(by_q=S, total=S_all, q1_q4=float(sum(S[:4])),
                             share_q5=(S[4] / S_all) if S_all != 0 else float("nan"))
        # 主要量二：Q5 的差拆成四組
        G = {g: float(n_g[g] / 50 * np.nanmean((c["MAGNET"][k][g] - c["LASSO [24]"][k][g])[q == N_Q - 1]))
             for g in SL.GROUPS}
        tot = float(sum(G.values()))
        res["G_Q5"][k] = dict(by_group=G, total=tot,
                              share={g: (v / tot if tot != 0 else float("nan")) for g, v in G.items()})
        res["judgement"][k] = judge(res["S_E"][k], res["G_Q5"][k])
    return res


def judge(SE: dict, GQ: dict) -> dict:
    """登記第 3 節的判讀規則。"""
    S_all, S5, share = SE["total"], SE["by_q"][4], SE["share_q5"]
    if S_all < 0 and S5 < 0 and share >= 0.5:
        r1 = "全部來自 Q5" if SE["q1_q4"] >= 0 else "集中在 Q5"
        r1_ok = True
    elif S5 >= 0:
        r1, r1_ok = "不在 Q5", False
    elif S_all < 0:
        r1, r1_ok = "分散在各種日子，不集中在大行情日", False
    else:
        r1, r1_ok = "全期沒有差距可分配（S_E,全期 ≥ 0）", False
    G, tot = GQ["by_group"], GQ["total"]
    r2_ok = tot < 0 and G[E] < 0 and G[E] / tot >= 0.5
    worst = min(G, key=G.get)
    r2 = ("主要來自未配對電子" if r2_ok else
          f"貢獻最負的是 {worst}（占 {G[worst] / tot:.0%}）" if tot != 0 else "Q5 沒有差距")
    if r1_ok and r2_ok:
        both = "同一件事：[24] 在美股大行情日的未配對電子股上 IC 較高"
    elif r1_ok or r2_ok:
        both = "只有 " + ("R1" if r1_ok else "R2") + " 成立"
    else:
        both = "兩者都不成立：兩頁分開解釋"
    return dict(R1=r1, R1_holds=bool(r1_ok), R2=r2, R2_holds=bool(r2_ok), combined=both)


ZH = {"ADR-paired": "ADR 配對", "Unpaired electronics": "未配對電子", "Financials": "金融",
      "Traditional": "傳產", "Unpaired (all 43)": "未配對合計"}


def report(res: dict) -> None:
    print(f"第一折 {res['start']} ~ {res['end']}，{res['n_days']} 天")
    print("五分位 |m(t)|（%）：" + "；".join(f"Q{x['q']} {x['n']} 天 {x['lo']:.2f}~{x['hi']:.2f}" for x in res["quintiles"]))
    for k, lab in (("ic", "IC"), ("ric", "RankIC")):
        print(f"\n每檔平均貢獻（{lab}）：MAGNET / LASSO / 差（組內 t 檢定 p）")
        print(f"  {'':10s}" + "".join(f"{'Q' + str(i + 1):>30s}" for i in range(N_Q)))
        for g in GN:
            row = res["cells"][k][g]
            print(f"  {ZH[g]:8s}" + "".join(
                f"  {x['magnet']['mean']:+.3f}/{x['lasso']['mean']:+.3f}/{x['diff']['mean']:+.3f}({x['diff']['p']:.2f})"
                for x in row))
        SE = res["S_E"][k]
        print(f"\n  主要量一（{lab}）：未配對電子對 IC 差的總量 S_E,Q = "
              + " / ".join(f"{v:+.4f}" for v in SE["by_q"])
              + f"；全期 {SE['total']:+.4f}；Q1–Q4 合計 {SE['q1_q4']:+.4f}；share_Q5 {SE['share_q5']:.2f}")
        GQ = res["G_Q5"][k]
        print(f"  主要量二（{lab}）：Q5 的差 {GQ['total']:+.4f} = "
              + " + ".join(f"{ZH[g]} {v:+.4f}（{GQ['share'][g]:.0%}）" for g, v in GQ["by_group"].items()))
        j = res["judgement"][k]
        print(f"  判讀（{lab}）：R1 {j['R1']}；R2 {j['R2']}；合起來：{j['combined']}")


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
    groups = SL.build_groups()
    if args.smoke:
        rng = np.random.default_rng(0)
        tick = sorted(set().union(*groups.values()))
        M = {name: dict(ic=rng.normal(size=(246, 50)), ric=rng.normal(size=(246, 50)), tickers=tick)
             for name, _, _ in METHODS}
        c, n_g = group_series(M, groups)
        q = MS.quintile(rng.random(246))
        qinfo = [dict(q=i + 1, n=int((q == i).sum()), lo=0.0, hi=0.0) for i in range(N_Q)]
        report(compute(np.array(["2000-01-01"] * 246), c, n_g, q, qinfo))
        print("\n冒煙測試完成（隨機數字，沒有讀任何預測，沒有寫檔）")
        return
    if args.write and OUT_JSON.exists():
        raise SystemExit(f"{OUT_JSON.relative_to(ROOT)} 已存在，--write 只能跑一次；請用 --report")

    dates, M = load()
    state = np.abs(MS.us_move(dates))
    q = MS.quintile(state)
    qinfo = [dict(q=i + 1, n=int((q == i).sum()), lo=float(state[q == i].min() * 100),
                  hi=float(state[q == i].max() * 100)) for i in range(N_Q)]
    c, n_g = group_series(M, groups)
    verify(M, c, n_g, q)
    if args.check:
        return
    res = compute(dates, c, n_g, q, qinfo)
    OUT_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    report(res)
    print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
