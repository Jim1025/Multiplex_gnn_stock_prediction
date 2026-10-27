"""
sa_optional_eval.py — SA 的三項選做實驗（proposal §83）的事前登記分析

登記見 §83（reg_sa_optional.md，計算前寫下並記 SHA-256）。本檔在第一個 run 開始之前寫好並記 SHA-256，
每一項的 run 全部完成後才跑那一項，各只跑一次。口徑與 §64.9、§79 相同（共用 tw_lag_channel 的載入與統計）：

  A  打亂產業標籤的安慰劑   SA − SAshuf：每一折、每個指標，差 > 0 且配對 p < 0.05；兩折都成立才算該指標成立
  B  beta 層元件消融         SA − SAno1 / SAno3 / SAonly2：每一折、每個指標內三個對照做 Holm；
                             「貢獻存在」= 兩折的差都 > 0 且 Holm 後 p 兩折都 < 0.05
  C  beta_init_std 敏感度    SAbis01、SAbis10 對 SA 的 test IC，Welch t；兩折同向且 Welch p 都 < 0.05 才算「敏感」

--validate：把各項的 arm 指到既有的 run，重現登記引用的數字
  A  SA -> 完整 2b（rgcnTW）、SAshuf -> §79 的 SH（rgcnShuf）：§79 的 F − SH，IC +0.0048 / +0.0066、RankIC +0.0113 / +0.0062
  B  SA -> beta 層前版（tw50_beta）、三個消融 -> tw50_beta_no1 / no3 / only2（只有第一折）：§26 的 +0.0073 / +0.0179 / +0.0295
  C  SA -> 原主結果、兩個值 -> tw50_bis01 / bis10（第二折 f2_）：§64.9 的 +0.0043 / −0.0004 與 −0.0178 / −0.0147

只讀 runs/ 的預測檔與 meta.json，不寫入任何檔案。

用法：
    .venv/bin/python scripts/sa_optional_eval.py --part A
    .venv/bin/python scripts/sa_optional_eval.py --part B --validate
"""

from __future__ import annotations

import argparse
import glob
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
import tw_lag_channel as T  # noqa: E402
import tw_relational_eval as E  # noqa: E402
from tw_rgcn_ablation_eval import holm  # noqa: E402  §79 用的同一份 Holm

PREFIX = {"第一折": "tw50", "第二折": "f2"}
N_SEEDS = 10
METRICS = (("IC", "IC"), ("RIC", "RankIC"))
PARTS = {
    "A": dict(base="rgcnSA", arms=[("產業結構（SA − SAshuf）", "rgcnSAshuf")]),
    "B": dict(base="rgcnSA", arms=[("① ADR 配對（SA − SAno1）", "rgcnSAno1"),
                                   ("③ 殘差耦合（SA − SAno3）", "rgcnSAno3"),
                                   ("①+③（SA − SAonly2）", "rgcnSAonly2")]),
    "C": dict(base="rgcnSA", arms=[("init 0.1（SAbis01 − SA）", "rgcnSAbis01"),
                                   ("init 1.0（SAbis10 − SA）", "rgcnSAbis10")]),
}
# --validate 的對應（登記 §3–§5 引用的既有結果）
VALIDATE = {
    "A": dict(base="rgcnTW", arms=["rgcnShuf"]),
    "B": dict(base="beta", arms=["beta_no1", "beta_no3", "beta_only2"]),
    "C": dict(base=None, arms=["bis01", "bis10"]),
}


def pat(fold: str, tag: str) -> str:
    return f"runs/**/*{PREFIX[fold]}_{tag}_s*/predictions/{{}}_predictions.csv"


def run_meta(pattern: str) -> dict[int, dict]:
    """種子 -> meta.json（best epoch 與總 epoch 數，C 用）。"""
    out = {}
    for f in glob.glob(str(ROOT / pattern.format("test")), recursive=True):
        d = Path(f).parents[1]
        tail = d.name.rsplit("_s", 1)[-1]
        if tail.isdigit() and (d / "meta.json").exists():
            out[int(tail)] = json.load(open(d / "meta.json"))
    return out


def seed_metrics(D, X):
    r = [T.summ(x, D["y_test"]) for x in X]
    return {k: np.array([q[k] for q in r]) for k in ("IC", "RIC")}, np.nanmean([q["ser"] for q in r], 0)


def paired(D, A, sa, B, sb):
    """同種子配對：回傳 (A 的逐種子指標, B 的逐種子指標, A 的種子平均逐日 IC, B 的, 共同種子)。"""
    X, Y, common = E._align(A, sa, B, sb)
    mx, ex = seed_metrics(D, X)
    my, ey = seed_metrics(D, Y)
    return mx, my, ex, ey, common


# ---------------------------------------------------------------- A、B：基準 − 對照，同種子配對
def contrasts(D, fold: str, part: str, pats: dict, main) -> dict:
    base, sb = pats["base"]
    if not len(sb):
        print("    基準 arm 沒有 run，跳過")
        return {}
    rows = {k: [] for k, _ in METRICS}
    for (name, _), (X, sx) in zip(PARTS[part]["arms"], pats["arms"]):
        if not len(sx):
            print(f"    {name}：沒有 run，跳過")
            continue
        mx, mb, ex, eb, common = paired(D, X, sx, base, sb)
        if len(common) != N_SEEDS:
            print(f"    注意：{name} 的共同種子 {len(common)} 顆（登記的是 {N_SEEDS} 顆）")
        for k, lab in METRICS:
            d = mb[k] - mx[k]
            rows[k].append(dict(name=name, d=d, p=float(stats.ttest_rel(mb[k], mx[k]).pvalue),
                                xm=float(mx[k].mean()), bm=float(mb[k].mean()),
                                xsd=float(mx[k].std(ddof=1)), bsd=float(mb[k].std(ddof=1))))
        print(f"    種子平均的逐日 IC：{name} {np.nanmean(eb - ex):+.4f}，逐日 HAC p {T.hac_p(eb - ex):.4f}")
    out = {}
    for k, lab in METRICS:
        if not rows[k]:
            continue
        adj = holm([r["p"] for r in rows[k]]) if part == "B" else [r["p"] for r in rows[k]]
        print(f"  [{lab}] 基準 − 對照（同種子配對，雙尾{'；三個對照做 Holm' if part == 'B' else ''}）")
        for r, pa in zip(rows[k], adj):
            d = r["d"]
            print(f"      {r['name']:22s} 對照 {r['xm']:+.4f}（sd {r['xsd']:.4f}）-> 基準 {r['bm']:+.4f}（sd {r['bsd']:.4f}）  "
                  f"差 {d.mean():+.4f}（sd {d.std(ddof=1):.4f}，{int((d > 0).sum())}/{len(d)} 為正）；p {r['p']:.4f}"
                  + (f"，Holm {pa:.4f}" if part == "B" else ""))
            out[(k, r["name"])] = dict(d=float(d.mean()), p=float(pa))
    if part == "A" and main is not None and len(pats["arms"][0][1]):
        X, sx = pats["arms"][0]
        mx, mm, ex, em, common = paired(D, X, sx, *main)
        for k, lab in METRICS:
            d = mx[k] - mm[k]
            print(f"    描述：SAshuf − 原主結果 {lab} {d.mean():+.4f}（配對 p {stats.ttest_rel(mx[k], mm[k]).pvalue:.4f}，"
                  f"{int((d > 0).sum())}/{len(d)} 為正）")
        print(f"    描述：SAshuf − 原主結果 種子平均的逐日 IC {np.nanmean(ex - em):+.4f}，HAC p {T.hac_p(ex - em):.4f}")
    return out


def verdict_ab(part: str, res: dict) -> None:
    if len(res) < 2 or any(not v for v in res.values()):
        print("\n== 判定：資料不足，不判定")
        return
    print("\n== 判定（登記 §3 / §4）")
    for k, lab in METRICS:
        for name, _ in PARTS[part]["arms"]:
            v = [r[(k, name)] for r in res.values()]
            ok = sum(x["d"] > 0 and x["p"] < 0.05 for x in v)
            neg = any(x["d"] < 0 and x["p"] < 0.05 for x in v)
            txt = ("兩折都成立" if ok == 2 else "對照較好且顯著（至少一折）" if neg else
                   "只有一折成立" if ok == 1 else "兩折都不成立")
            print(f"    {lab:7s} {name:22s} {txt}")


# ---------------------------------------------------------------- C：Welch（§64.9 的口徑）
def sensitivity(D, fold: str, pats: dict, metas: dict) -> dict:
    base, sb = pats["base"]
    out = {}
    mb_all, _ = seed_metrics(D, base)
    for (name, _), (X, sx), meta in zip(PARTS["C"]["arms"], pats["arms"], metas["arms"]):
        if not len(sx):
            print(f"    {name}：沒有 run，跳過")
            continue
        mx_all, _ = seed_metrics(D, X)
        mx, mb, ex, eb, common = paired(D, X, sx, base, sb)
        for k, lab in METRICS:
            d = mx_all[k].mean() - mb_all[k].mean()
            pw = float(stats.ttest_ind(mx_all[k], mb_all[k], equal_var=False).pvalue)
            pp = float(stats.ttest_rel(mx[k], mb[k]).pvalue)
            print(f"    {name:22s} {lab:6s} {mx_all[k].mean():+.4f}（sd {mx_all[k].std(ddof=1):.4f}，{len(sx)} 顆）vs 基準 "
                  f"{mb_all[k].mean():+.4f}（sd {mb_all[k].std(ddof=1):.4f}）：差 {d:+.4f}；Welch p {pw:.4f}、配對 p {pp:.4f}")
            out[(k, name)] = dict(d=float(d), p=pw)
        print(f"      種子平均的逐日 IC 差 {np.nanmean(ex - eb):+.4f}，HAC p {T.hac_p(ex - eb):.4f}")
        ep = {s: (m.get("best_epoch"), m.get("n_epochs")) for s, m in sorted(meta.items())}
        bad = [s for s, (b, _) in ep.items() if b is not None and b <= 3]
        print("      best epoch / 總 epoch：" + "、".join(f"s{s} {b}/{n}" for s, (b, n) in ep.items()))
        if bad:
            keep = [i for i, s in enumerate(sx) if s not in bad]
            print(f"      best epoch <= 3 的種子：{bad}；排除後 IC {mx_all['IC'][keep].mean():+.4f}（{len(keep)} 顆，描述）")
    return out


def verdict_c(res: dict) -> None:
    if len(res) < 2 or any(not v for v in res.values()):
        print("\n== 判定：資料不足，不判定")
        return
    print("\n== 判定（登記 §5：兩折同向且 Welch p 都 < 0.05 才算敏感；主判準是 IC）")
    for name, _ in PARTS["C"]["arms"]:
        v = [r[("IC", name)] for r in res.values()]
        sens = all(x["p"] < 0.05 for x in v) and np.sign(v[0]["d"]) == np.sign(v[1]["d"])
        print(f"    {name:22s} {'敏感（' + ('較好' if v[0]['d'] > 0 else '較差') + '）' if sens else '穩健'}"
              f"：兩折差 {v[0]['d']:+.4f} / {v[1]['d']:+.4f}，Welch p {v[0]['p']:.4f} / {v[1]['p']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SA 的三項選做實驗（proposal §83）")
    ap.add_argument("--part", required=True, choices=list(PARTS))
    ap.add_argument("--validate", action="store_true", help="arm 指到既有的 run，重現登記引用的數字")
    a = ap.parse_args()
    res = {}
    for fold, f in T.FOLDS.items():
        D = T.load_fold(f["cfg"], f["f2"])
        if a.validate:
            v = VALIDATE[a.part]
            bp = f["main"] if v["base"] is None else pat(fold, v["base"])
            aps = [pat(fold, t) for t in v["arms"]]
        else:
            bp = pat(fold, PARTS[a.part]["base"])
            aps = [pat(fold, t) for _, t in PARTS[a.part]["arms"]]
        pats = dict(base=T.preds(bp, "test", D["d_test"], D["codes"]),
                    arms=[T.preds(p, "test", D["d_test"], D["codes"]) for p in aps])
        print(f"\n######## {fold}：測試窗 {D['d_test'][0]} ~ {D['d_test'][-1]}（{len(D['d_test'])} 天）；"
              f"基準 {len(pats['base'][1])} 顆、" + "、".join(f"{len(x[1])} 顆" for x in pats["arms"]))
        if a.part == "C":
            res[fold] = sensitivity(D, fold, pats, dict(arms=[run_meta(p) for p in aps]))
        else:
            main_arm = None if a.validate else T.preds(f["main"], "test", D["d_test"], D["codes"])
            res[fold] = contrasts(D, fold, a.part, pats, main_arm)
    if a.validate:
        return
    verdict_c(res) if a.part == "C" else verdict_ab(a.part, res)


if __name__ == "__main__":
    main()
