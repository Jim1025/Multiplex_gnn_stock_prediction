"""
market_state_eval.py — 評估議題 4：依美股前一晚的波動幅度分組，看各方法的逐日 IC（第一折）

登記：docs/proposal_1209_revision.md §1.8 附錄的 A 節（計算前寫下）。只做第一折，描述性分析。

定義：
    m(t)  對台股目標日 t，30 檔美股層節點各自在 t 之前最後一個交易日的 log_return（嚴格 <，與資料集相同），
          等權平均
    狀態  |m(t)|，在 test 期內依名次分五等分（Q1 最平靜、Q5 波動最大；平手依日期先後）
    分界用整段 test 期算，是事後分組，不是可交易的訊號。

方法（資料讀法與主表 results_table.py 相同，逐日 IC 為種子平均）：
    MAGNET                 tw50_rgcnSA（10 顆）
    MAGNET-single market   tw50_smktSA（10 顆）
    LASSO [24]             *_bipartite_rolling_t2_LASSO 的最新一批
    MEIG [1]               tw50_bl_meig（3 顆）

輸出：每組各方法的平均逐日 IC / RankIC、標準誤、天數；MAGNET − LASSO 與 MAGNET − 單市場的逐日 IC 差
（組內雙尾 t 檢定；組內日子不連續，不做 HAC）；|m(t)| 與逐日 IC 的 Spearman 相關。
--validate：只印四個方法的全期平均（須等於主表 0.1127 / 0.0250 / 0.1042 / 0.0061），不分組。
--write：把結果寫到 docs/figures/data/eval_market_state.json，供 gen_discussion_figures.py 畫圖。

不需要 torch：只讀各 run 的 predictions/test_predictions*.csv 與 data/features/adr/*.csv。

用法：
    .venv/bin/python scripts/market_state_eval.py --validate
    .venv/bin/python scripts/market_state_eval.py --write
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
from src.dataset.config import load_universe  # noqa: E402

METHODS = [("MAGNET", "neural", "tw50_rgcnSA"),
           ("MAGNET-single market", "neural", "tw50_smktSA"),
           ("LASSO [24]", "nonneural", R.BIP.format("LASSO")),
           ("MEIG [1]", "neural", "tw50_bl_meig")]
EXPECT_IC = {"MAGNET": 0.1127, "MAGNET-single market": 0.0250, "LASSO [24]": 0.1042, "MEIG [1]": 0.0061}
N_Q = 5
OUT_JSON = ROOT / "docs" / "figures" / "data" / "eval_market_state.json"
FEATURES = ROOT / "data" / "features" / "adr"


def load_methods() -> tuple[np.ndarray, dict]:
    """回傳 (日期, {方法: {"ic": 逐日 IC, "ric": 逐日 RankIC, "n_seeds": 顆數}})；四個方法的日期必須相同。"""
    dates, out = None, {}
    for name, kind, key in METHODS:
        st = R.neural_stats(key) if kind == "neural" else R.nonneural_stats(key)
        if st is None or st.get("daily_ic") is None:
            raise FileNotFoundError(f"找不到 {name}（{key}）的預測檔")
        d = np.asarray([str(x)[:10] for x in st["dates"]])
        if dates is None:
            dates = d
        elif not np.array_equal(dates, d):
            raise ValueError(f"{name} 的日期與其他方法不同")
        out[name] = dict(ic=np.asarray(st["daily_ic"], float), ric=np.asarray(st["daily_ric"], float),
                         n_seeds=st["n"])
    return dates, out


def us_move(dates: np.ndarray) -> np.ndarray:
    """m(t)：30 檔美股層節點在 t 之前最後一個交易日的 log_return 等權平均。"""
    us = load_universe("tw50").us_nodes
    cols = []
    for tk in us:
        s = pd.read_csv(FEATURES / f"{tk}.csv", index_col="Date", parse_dates=True)["log_return"].sort_index()
        # 嚴格早於目標日的最後一列（與 MultiplexDataset._build_sequence 相同的規則）
        pos = s.index.searchsorted(pd.to_datetime(dates), side="left") - 1
        if (pos < 0).any():
            raise ValueError(f"{tk} 在最早的目標日之前沒有資料")
        cols.append(s.to_numpy()[pos])
    M = np.column_stack(cols)                       # [天, 30]
    if np.isnan(M).any():
        raise ValueError("美股報酬有缺值")
    return M.mean(axis=1)


def quintile(state: np.ndarray) -> np.ndarray:
    """依名次分五等分，回傳 0..4（0 = 最平靜）。平手依原順序（日期先後）。"""
    rank = stats.rankdata(state, method="ordinal")  # 1..n，平手依出現順序
    return np.minimum((rank - 1) * N_Q // len(state), N_Q - 1).astype(int)


def mean_se(x: np.ndarray) -> tuple[float, float, int]:
    x = x[~np.isnan(x)]
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), int(len(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="只印全期平均，與主表比對，不分組")
    ap.add_argument("--write", action="store_true", help=f"寫 {OUT_JSON.relative_to(ROOT)}")
    args = ap.parse_args()

    dates, M = load_methods()
    print(f"第一折 test：{dates[0]} ~ {dates[-1]}，{len(dates)} 天\n")
    print("全期平均（須等於主表）：")
    ok = True
    for name, d in M.items():
        ic = float(np.nanmean(d["ic"]))
        good = abs(ic - EXPECT_IC[name]) < 5e-5
        ok &= good
        print(f"  {name:22s} IC {ic:+.4f}（主表 {EXPECT_IC[name]:+.4f}）{'' if good else '  <-- 不符'}  "
              f"RankIC {np.nanmean(d['ric']):+.4f}  n={d['n_seeds']}")
    if not ok:
        raise SystemExit("全期平均與主表不符，停止（登記的檢查未通過）")
    if args.validate:
        return

    m = us_move(dates)
    state = np.abs(m)
    q = quintile(state)
    res = dict(fold=1, n_days=int(len(dates)), start=str(dates[0]), end=str(dates[-1]),
               definition="quintiles of |equal-weighted log return of the 30 US-layer nodes on the last US "
                          "session before the TW target day|, ranked within the test period",
               quintiles=[], methods={}, diffs={}, spearman={})

    print("\n分組（|m(t)|，%）：")
    for k in range(N_Q):
        s = state[q == k] * 100
        res["quintiles"].append(dict(q=k + 1, n=int((q == k).sum()), lo=float(s.min()), hi=float(s.max()),
                                     mean=float(s.mean())))
        print(f"  Q{k + 1}: {int((q == k).sum())} 天，|m| {s.min():.3f} ~ {s.max():.3f}（平均 {s.mean():.3f}）")

    for key, lab in (("ic", "IC"), ("ric", "RankIC")):
        print(f"\n各組的平均逐日 {lab}（± 標準誤）：")
        print(f"  {'':22s}" + "".join(f"{'Q' + str(k + 1):>18s}" for k in range(N_Q)))
        for name, d in M.items():
            cells = [mean_se(d[key][q == k]) for k in range(N_Q)]
            res["methods"].setdefault(name, {})[key] = [dict(mean=c[0], se=c[1], n=c[2]) for c in cells]
            print(f"  {name:22s}" + "".join(f"{c[0]:+10.4f} ± {c[1]:.4f}" for c in cells))

    print("\n逐日 IC 差（組內雙尾 t 檢定）：")
    for a, b in (("MAGNET", "LASSO [24]"), ("MAGNET", "MAGNET-single market")):
        d = M[a]["ic"] - M[b]["ic"]
        rows = []
        for k in range(N_Q):
            x = d[q == k]
            x = x[~np.isnan(x)]
            t = stats.ttest_1samp(x, 0.0)
            rows.append(dict(q=k + 1, mean=float(x.mean()), p=float(t.pvalue), n=int(len(x))))
        res["diffs"][f"{a} - {b}"] = rows
        print(f"  {a} − {b}")
        print("    " + "   ".join(f"Q{r['q']} {r['mean']:+.4f} (p {r['p']:.3f})" for r in rows))

    print("\n|m(t)| 與逐日 IC 的 Spearman 相關（全期）：")
    for name, d in M.items():
        okd = ~np.isnan(d["ic"])
        sp = stats.spearmanr(state[okd], d["ic"][okd])
        res["spearman"][name] = dict(rho=float(sp.statistic), p=float(sp.pvalue))
        print(f"  {name:22s} rho {sp.statistic:+.3f}（p {sp.pvalue:.4f}）")

    if args.write:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
        print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
