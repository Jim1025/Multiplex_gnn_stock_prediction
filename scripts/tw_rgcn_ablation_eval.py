"""
tw_rgcn_ablation_eval.py — 方案二 b 的元件消融（proposal §79）的事前登記分析

登記見 proposal §79（reg_rgcn_ablation.md，實作之前寫下並記 SHA-256）。本檔在第一個 run 開始之前寫好、
一併記 SHA-256，run 全部完成後才跑。口徑與 §71、§78 相同，共用 tw_lag_channel 的載入與統計：

  區塊 A（判準）  四個對照：其他產業（F − SA）、同產業（F − SO）、自身（F − AO）、產業結構（F − SH）。
                  每一折、每個指標（IC 為主、RankIC 為次）：逐種子 test 差的同種子配對 t 檢定（雙尾，n = 10），
                  四個對照做 Holm。「貢獻存在」= 兩折的差都 > 0，且 Holm 校正後的 p 兩折都 < 0.05
  區塊 B（描述）  各 arm 的 IC、RankIC、ICIR、sd(IC)、離散比與參數量；F − S（關係整體）；
                  四個對照的種子平均逐日 IC 差與其 HAC p

--validate：把四個消融 arm 都指到主 arm，只跑區塊 A。每個對照都等於 2b − 主 arm，
            應重現 §78 的 IC +0.0015 / +0.0060、RankIC +0.0103 / +0.0062。

只讀資料、runs/ 的預測檔與設定快照，不寫入任何檔案。

用法：
    .venv/bin/python scripts/tw_rgcn_ablation_eval.py
    .venv/bin/python scripts/tw_rgcn_ablation_eval.py --validate
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tw_lag_channel as T  # noqa: E402  載入、summ、hac_p、disp 與 §71、§78 同一份
import tw_relational_eval as E  # noqa: E402
from src.models import build_model  # noqa: E402

N_SEEDS = 10
PREFIX = {"第一折": "tw50", "第二折": "f2"}
ARM_TAGS = {"F": "rgcnTW", "SA": "rgcnSA", "SO": "rgcnSO", "AO": "rgcnAO", "SH": "rgcnShuf"}
CONTRASTS = [("其他產業（F − SA）", "SA"), ("同產業（F − SO）", "SO"),
             ("自身（F − AO）", "AO"), ("產業結構（F − SH）", "SH")]


def _pattern(fold: str, arm: str) -> str:
    return f"runs/**/*{PREFIX[fold]}_{ARM_TAGS[arm]}_s*/predictions/{{}}_predictions.csv"


def holm(ps) -> np.ndarray:
    """Holm 校正後的 p（逐步下修、單調化）。"""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        adj[k] = run
    return adj


def _per_seed(D, X, sx, Y, sy):
    A, B, common = E._align(X, sx, Y, sy)
    ra = [T.summ(a, D["y_test"]) for a in A]
    rb = [T.summ(b, D["y_test"]) for b in B]
    return {k: (np.array([r[k] for r in ra]), np.array([r[k] for r in rb])) for k in ("IC", "RIC")}, common


# ---------------------------------------------------------------- 區塊 A：判準
def block_a(D, P: dict) -> dict:
    F, sf = P["F"]
    out = {}
    for key, lab in (("IC", "IC"), ("RIC", "RankIC")):
        rows = []
        for name, arm in CONTRASTS:
            X, sx = P[arm]
            vals, common = _per_seed(D, X, sx, F, sf)          # (ablated, full)
            if len(common) != N_SEEDS:
                print(f"    注意：{name} 的共同種子 {len(common)} 顆（登記的是 {N_SEEDS} 顆）")
            x, f = vals[key]
            d = f - x
            rows.append((name, x.mean(), f.mean(), d, stats.ttest_rel(f, x).pvalue))
        adj = holm([r[4] for r in rows])
        print(f"[A] {lab}：完整 2b − 消融（同種子配對，雙尾；四個對照做 Holm）")
        for (name, xm, fm, d, p), pa in zip(rows, adj):
            print(f"      {name:16s} 消融 {xm:+.4f} -> 完整 {fm:+.4f}  差 {d.mean():+.4f}（sd {d.std(ddof=1):.4f}，"
                  f"{int((d > 0).sum())}/{len(d)} 為正）；p {p:.4f}，Holm {pa:.4f}")
            out[(key, name)] = dict(d=float(d.mean()), p=float(p), p_holm=float(pa))
    return out


def verdict(res: dict) -> None:
    """登記 §5：兩折的差都 > 0，且 Holm 校正後的 p 兩折都 < 0.05。"""
    if len(res) < 2 or any(not v for v in res.values()):
        print("\n== 判定：資料不足，不判定")
        return
    print("\n== 判定（每個元件、每個指標）")
    for key, lab in (("IC", "IC"), ("RIC", "RankIC")):
        for name, _ in CONTRASTS:
            v = [r[(key, name)] for r in res.values()]
            ok = all(x["d"] > 0 and x["p_holm"] < 0.05 for x in v)
            one = sum(x["d"] > 0 and x["p_holm"] < 0.05 for x in v)
            neg = any(x["d"] < 0 and x["p_holm"] < 0.05 for x in v)
            txt = ("貢獻存在（兩折都成立）" if ok else "只有一折成立" if one == 1 else
                   "反向（消融較好且顯著）" if neg else "兩折都不成立")
            print(f"    {lab:7s} {name:16s} {txt}")


# ---------------------------------------------------------------- 區塊 B：描述
def _n_params(pattern: str) -> int | None:
    snaps = sorted(glob.glob(str(ROOT / pattern.replace("predictions/{}_predictions.csv", "config_snapshot.yaml")),
                             recursive=True))
    if not snaps:
        return None
    return sum(p.numel() for p in build_model(yaml.safe_load(open(snaps[0]))).parameters())


def block_b(D, P: dict, pats: dict) -> None:
    Y = D["y_test"]
    print("[B] 描述（不計入判準）")
    for arm in ("S", "F", "SA", "SO", "AO", "SH"):
        X, sx = P[arm]
        if not len(sx):
            continue
        r = [T.summ(x, Y) for x in X]
        print(f"    {arm:2s}（{len(sx)} 顆，參數 {_n_params(pats[arm])}）：IC {np.mean([q['IC'] for q in r]):+.4f}、"
              f"RankIC {np.mean([q['RIC'] for q in r]):+.4f}、ICIR {np.mean([q['ICIR'] for q in r]):.4f}、"
              f"sd(IC) {np.mean([q['sd'] for q in r]):.4f}、離散比 {np.mean([T.disp(x, Y) for x in X]):.3f}")
    F, sf = P["F"]
    for name, arm in [("關係整體（F − S）", "S")] + CONTRASTS:
        X, sx = P[arm]
        A, B, common = E._align(X, sx, F, sf)
        if len(common) < 2:
            continue
        sa = np.nanmean([T.summ(a, Y)["ser"] for a in A], 0)
        sb = np.nanmean([T.summ(b, Y)["ser"] for b in B], 0)
        print(f"    種子平均的逐日 IC：{name} {np.nanmean(sb - sa):+.4f}，逐日 HAC p {T.hac_p(sb - sa):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="方案二 b 的元件消融（proposal §79）的登記分析")
    ap.add_argument("--blocks", nargs="*", default=list("AB"), choices=list("AB"))
    ap.add_argument("--validate", action="store_true",
                    help="四個消融 arm 都指到主 arm，只跑區塊 A（應重現 §78 的 2b − 主 arm）")
    args = ap.parse_args()
    res = {}
    for fold, f in T.FOLDS.items():
        D = T.load_fold(f["cfg"], f["f2"])
        pats = {arm: _pattern(fold, arm) for arm in ARM_TAGS}
        pats["S"] = f["main"]
        if args.validate:
            for arm in ("SA", "SO", "AO", "SH"):
                pats[arm] = f["main"]
        P = {arm: T.preds(p, "test", D["d_test"], D["codes"]) for arm, p in pats.items()}
        print(f"\n######## {fold}：測試窗 {D['d_test'][0]} ~ {D['d_test'][-1]}（{len(D['d_test'])} 天）；"
              + "、".join(f"{a} {len(P[a][1])} 顆" for a in ("S", "F", "SA", "SO", "AO", "SH")))
        if "A" in args.blocks or args.validate:
            res[fold] = block_a(D, P) if len(P["F"][1]) else {}
        if "B" in args.blocks and not args.validate:
            block_b(D, P, pats)
    if "A" in args.blocks and not args.validate:
        verdict(res)


if __name__ == "__main__":
    main()
