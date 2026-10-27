"""
best_vs_literature.py — MAGNET 能不能稱最好（proposal §73）

主指標 IC、第二指標 RankIC（2026-10-17 起）。對主表的每一個文獻方法（6 個文獻神經模型、[24] 的 10 個模型；
第二折的文獻神經模型 2026-10-17 補跑，§74），
比較 MAGNET 本版（10 顆種子平均）與 10 顆集成：

  - 差值與逐日 HAC p；每折、每個指標各自做 Holm 校正（文獻方法為一族）
  - 附錄 B 的非文獻同輸入對照（KTW+、R2、TW+、關係型、兩區塊、RC）一併列出，不進 Holm

第一折直接用 results_table.py 的函式（與主表同一套算法，p 值與主表相同）；
第二折照 results_table._fold2_rows 的載入與 HAC（常態近似）重寫，p 值與第二折主表相同。
屬描述性：Holm 是本腳本新加的，表上原本的 p 都未校正。

只讀預測檔，不寫入任何檔案。

2026-10-27 起主結果改為 SA（MAGNET + 同產業關係，§83）：預設比較 SA；--arm prior 改比原主結果（不帶關係），兩者都列出才透明
（SA 是事後選的，對文獻方法新出現的顯著要保留）。

用法：
    .venv/bin/python scripts/best_vs_literature.py              # 主結果（SA）
    .venv/bin/python scripts/best_vs_literature.py --arm prior   # 原主結果（不帶關係）
"""

from __future__ import annotations

import argparse
import glob
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

CUT = "2024-12-25"          # 第二折評估窗上界，與 results_table._fold2_rows 相同


def holm(ps, alpha: float = 0.05) -> np.ndarray:
    """Holm 逐步校正：回傳每個假設是否拒絕。"""
    order = np.argsort(ps)
    m = len(ps)
    keep = np.zeros(m, bool)
    for k, i in enumerate(order):
        if ps[i] <= alpha / (m - k):
            keep[i] = True
        else:
            break
    return keep


def report(fold: str, rows: list[dict], mic, mric, eic, eric) -> None:
    print(f"\n######## {fold}：{ARM['label']} IC {mic:+.4f} / RankIC {mric:+.4f}；"
          f"10 顆集成 {eic:+.4f} / {eric:+.4f}")
    lit = [r for r in rows if r["lit"]]
    for who, name in (("m", ARM["label"]), ("e", "10 顆集成")):
        for key, kn in (("ic", "IC"), ("ric", "RankIC")):
            ps = np.array([r[f"{who}_{key}_p"] for r in lit])
            keep = holm(ps)
            print(f"  [{name} 對文獻 {len(lit)} 個，{kn}] 差全部為正：{all(r[f'{who}_{key}_d'] > 0 for r in lit)}；"
                  f"未校正 p<0.05 {int((ps < 0.05).sum())} 個；Holm 後仍顯著 {int(keep.sum())} 個："
                  + "、".join(r["lab"] for r, k in zip(lit, keep) if k))
            print("      Holm 後不顯著：" + "；".join(
                f"{r['lab']} {r[f'{who}_{key}_d']:+.4f}（p {r[f'{who}_{key}_p']:.3f}）"
                for r, k in zip(lit, keep) if not k))
    top_ic = max(lit, key=lambda r: r["ic"])
    top_ric = max(lit, key=lambda r: r["ric"])
    print(f"  文獻方法 IC 最高：{top_ic['lab']} {top_ic['ic']:+.4f}；RankIC 最高：{top_ric['lab']} {top_ric['ric']:+.4f}")
    print("  附錄 B（非文獻，不進 Holm）：")
    for r in rows:
        if not r["lit"]:
            print(f"      {r['lab']:24s} IC {r['ic']:+.4f}（MAGNET − 它 {r['m_ic_d']:+.4f}，p {r['m_ic_p']:.3f}；"
                  f"集成 − 它 {r['e_ic_d']:+.4f}，p {r['e_ic_p']:.3f}） RankIC {r['ric']:+.4f}"
                  f"（MAGNET {r['m_ric_d']:+.4f}，p {r['m_ric_p']:.3f}；集成 {r['e_ric_d']:+.4f}，p {r['e_ric_p']:.3f}）")


# ---------------------------------------------------------------- 第一折
def fold1() -> None:
    lit = [(lab, arm, "n") for lab, arm, cat, _ in R.NEURAL if cat == "文獻"] + \
          [(lab, pat, "b") for lab, pat, _, _ in R.NONNEURAL if lab.startswith("[24]")]
    app = [(lab, pat, "b") for lab, pat, _, _ in R.NONNEURAL if lab in R.SUPPLEMENT]
    base, ens = R.neural_stats(ARM["f1"]), R.ensemble_stats(ARM["f1"])
    rows = []
    for lab, x, kind in lit + app:
        s = R.neural_stats(x) if kind == "n" else R.nonneural_stats(x)
        r = dict(lab=lab, lit=(lab, x, kind) in lit, ic=s["ic"], ric=s["ric"])
        for who, B in (("m", base), ("e", ens)):
            for key in ("ic", "ric"):
                r[f"{who}_{key}_d"], r[f"{who}_{key}_p"] = R.paired_daily(B, s, key)
        rows.append(r)
    report("第一折", rows, base["ic"], base["ric"], ens["ic"], ens["ric"])


# ---------------------------------------------------------------- 第二折
def _daily(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["target_date"] = df.target_date.astype(str).str[:10]
    o = {}
    for d, g in df.groupby("target_date"):
        # 50 檔完全相同才略過，與 results_table 相同（§77）
        if d > CUT or g.y.nunique() == 1 or g.y_hat.nunique() == 1:
            continue
        o[d] = (np.corrcoef(g.y_hat, g.y)[0, 1], stats.spearmanr(g.y_hat, g.y).statistic)
    return o


def series(files) -> tuple[list, np.ndarray]:
    """逐 run 的逐日 (IC, RankIC)，取全部 run 共同的日子後跨 run 平均。"""
    out = [_daily(pd.read_csv(f)) for f in files]
    k = sorted(set.intersection(*[set(x) for x in out]))
    # nan-aware（§74）：RankIC 無定義的日子略過，與 results_table._fold2_rows 相同；§77 起 _daily 已先略過這些天，這裡只是保險
    return k, np.nanmean(np.array([[[x[d][i] for d in k] for i in (0, 1)] for x in out]), axis=0)


def ens_series(files) -> tuple[list, np.ndarray]:
    """先逐 (日, 檔) 平均全部 run 的 y_hat，再算逐日 (IC, RankIC)。"""
    df = (pd.concat([pd.read_csv(f) for f in files]).groupby(["target_date", "ticker"])
          .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
    o = _daily(df)
    k = sorted(o)
    return k, np.array([[o[d][i] for d in k] for i in (0, 1)])


def hac(d) -> float:
    """與 results_table._fold2_rows 的 _hac 相同（Bartlett 核，常態近似）。"""
    n = len(d)
    lag = int(4 * (n / 100) ** (2 / 9))
    s2 = np.var(d, ddof=1)
    for l in range(1, lag + 1):
        s2 += 2 * (1 - l / (lag + 1)) * np.cov(d[l:], d[:-l], ddof=1)[0, 1]
    return float(2 * (1 - stats.norm.cdf(abs(d.mean() / (s2 / n) ** 0.5))))


def fold2() -> None:
    bf = sorted(glob.glob(str(ROOT / f"runs/**/*{ARM['f2']}_s*/predictions/test_predictions.csv"), recursive=True))
    kb, B = series(bf)
    ke, E = ens_series(bf)
    pats = [(R.bip_label(m), R._latest(f"runs_f2/{R.BIP.format(m)}") + "/predictions/test_predictions.csv", True)
            for m, _ in R.BIP_MODELS]
    # 6 個文獻神經模型的第二折（2026-10-17 補跑，§74），與第一折同一族
    pats += [(lab, f"runs/**/*f2_bl_{arm[len('tw50_bl_'):]}_s*/predictions/test_predictions.csv", True)
             for lab, arm, cat, _ in R.NEURAL if cat == "文獻"]
    pats += [("關係型線性（z 目標）", "runs_f2/*_lin_relational/predictions/test_predictions.csv", False),
             ("兩區塊 ridge（z 目標）", "runs_f2/*_lin_twoblock/predictions/test_predictions.csv", False),
             ("KTW+（rank 目標）", "runs_f2/*fvg_KTWp/predictions/*.csv", False),
             ("R2 per-target ridge", "runs_f2/*ridge_R2/predictions/*.csv", False),
             ("TW+（美股+台股 80 維）", "runs_f2/*fvg_TWp/predictions/*.csv", False),
             ("RC 常數對照", "runs_f2/*ridge_RC/predictions/*.csv", False)]
    rows = []
    for lab, pat, lit in pats:
        files = sorted(glob.glob(str(ROOT / pat), recursive=True))
        if not files:
            print(f"  [skip] 第二折 {lab}（找不到預測檔）")
            continue
        k, V = series(files)
        r = dict(lab=lab, lit=lit)
        for who, (kk, M) in (("m", (kb, B)), ("e", (ke, E))):
            ii = [k.index(d) for d in k if d in kk]
            jj = [kk.index(d) for d in k if d in kk]
            r["ic"], r["ric"] = np.nanmean(V[0][ii]), np.nanmean(V[1][ii])
            for i, key in ((0, "ic"), (1, "ric")):
                dd = M[i][jj] - V[i][ii]
                dd = dd[np.isfinite(dd)]
                r[f"{who}_{key}_d"], r[f"{who}_{key}_p"] = float(dd.mean()), hac(dd)
        rows.append(r)
    report("第二折", rows, B[0].mean(), B[1].mean(), E[0].mean(), E[1].mean())


ARM: dict = {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MAGNET 能不能稱最好（proposal §73、§83）")
    ap.add_argument("--arm", choices=["main", "prior"], default="main",
                    help="main = 主結果（SA，2026-10-27 起）；prior = 原主結果（不帶關係）")
    a = ap.parse_args()
    ARM.update(dict(f1=R.BEST, f2=R.BEST_F2, label=R.MAIN_LABEL) if a.arm == "main"
               else dict(f1=R.PRIOR, f2=R.PRIOR_F2, label=R.PRIOR_LABEL))
    R.PREDICTIONS = "reeval"
    fold1()
    fold2()
