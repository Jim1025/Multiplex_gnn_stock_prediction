"""
tw_rgcn_eval.py — 方案二 b：關係型卷積取代 GAT_L2（proposal §78）的事前登記分析

登記見 proposal §78（reg_tw_rgcn.md，實作之前寫下並記 SHA-256）。本檔在第一個 run 開始之前寫好、
一併記 SHA-256，run 全部完成後才跑。口徑與 §71 相同（scripts/tw_relational_eval.py），
共用 tw_lag_channel 的載入與統計：

  區塊 A（判準）  每一折：2b − 現行關係型層的逐種子 test IC、RankIC 差（同種子配對，n = 10），
                  單尾 95% 下界（t 分布，自由度 n − 1）> −δ，δ = 0.005。兩折、兩個指標都要成立
                  （交集聯集檢定，不做多重比較校正）
  區塊 B（描述）  2b 對主 arm 的 IC、RankIC、ICIR、sd(IC)、離散比（同種子配對，雙尾 p）；
                  種子平均的逐日 IC 差（2b − 現行、2b − 主 arm）的 HAC p；10 顆集成
  區塊 C（描述）  ‖W_r‖（meta.json）；把 W 全部歸零後的 val / test（tw_relational_eval.block_c）
  區塊 D（描述）  只歸零一種關係、同業訊息的方向、§68 的準度指標（tw_relational_posthoc 的 (a)–(d)）

--validate：把「2b」指到現行關係型層、「現行」指到主 arm，只跑區塊 A。應重現 §71 的
            IC / RankIC 差（+0.0010 / +0.0067、+0.0113 / +0.0081），用來在第一個 run 之前驗證程式。

只讀資料、runs/ 的預測檔與 checkpoint，不寫入任何檔案。

用法：
    .venv/bin/python scripts/tw_rgcn_eval.py              # 區塊 A B C D
    .venv/bin/python scripts/tw_rgcn_eval.py --validate
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tw_lag_channel as T  # noqa: E402  載入、summ、hac_p 與 §71 同一份
import tw_relational_eval as E  # noqa: E402
import tw_relational_posthoc as P  # noqa: E402

DELTA = 0.005           # 非劣界線：約等於主 arm 的跨種子 IC sd（0.0051 / 0.0039），登記 §5
N_SEEDS = 10
ARMS = {
    "第一折": "runs/**/*tw50_rgcnTW_s*/predictions/{}_predictions.csv",
    "第二折": "runs/**/*f2_rgcnTW_s*/predictions/{}_predictions.csv",
}


def _noninf(lab, ref, cand) -> dict:
    """cand − ref 的單尾 95% 下界是否 > −DELTA。"""
    d = cand - ref
    n = len(d)
    lo = d.mean() - stats.t.ppf(0.95, n - 1) * d.std(ddof=1) / np.sqrt(n)
    ok = bool(lo > -DELTA)
    print(f"      {lab:7s} {ref.mean():+.4f} -> {cand.mean():+.4f}  差 {d.mean():+.4f}（sd {d.std(ddof=1):.4f}，"
          f"{int((d > 0).sum())}/{n} 為正）；單尾 95% 下界 {lo:+.4f} -> {'非劣' if ok else '未成立'}（界線 −{DELTA}）")
    return dict(d=float(d.mean()), lo=float(lo), ok=ok, n=n)


# ---------------------------------------------------------------- 區塊 A：判準
def block_a(D, R, sr, C, sc, lab_ref="現行關係型層", lab_cand="2b") -> dict:
    A, B, common = E._align(R, sr, C, sc)
    print(f"[A] 判準：{lab_cand} − {lab_ref}，共同種子 {len(common)} 顆：{common}")
    if len(common) != N_SEEDS:
        print(f"    注意：登記的是 {N_SEEDS} 顆種子，實際 {len(common)} 顆")
    if len(common) < 2:
        return {}
    ra = [T.summ(a, D["y_test"]) for a in A]
    rb = [T.summ(b, D["y_test"]) for b in B]
    return {key: _noninf(lab, np.array([r[key] for r in ra]), np.array([r[key] for r in rb]))
            for key, lab in (("IC", "IC"), ("RIC", "RankIC"))}


def verdict(res: dict) -> None:
    """登記 §8 的判讀表。"""
    if len(res) < 2 or any(not v for v in res.values()):
        print("\n== 判定：資料不足，不判定")
        return
    ok_ic = all(v["IC"]["ok"] for v in res.values())
    ok_ric = all(v["RIC"]["ok"] for v in res.values())
    row = ("IC、RankIC 兩折都非劣" if ok_ic and ok_ric else
           "IC 兩折非劣，RankIC 有一折不成立" if ok_ic else "IC 任一折不成立")
    print(f"\n== 判定：IC {'兩折都非劣' if ok_ic else '未成立'}；RankIC {'兩折都非劣' if ok_ric else '未成立'}"
          f" -> 判讀表的列：「{row}」")


# ---------------------------------------------------------------- 區塊 B：描述
def block_b(D, M, sm, R, sr, C, sc) -> None:
    Y = D["y_test"]
    print("[B] 描述（不計入判準）")
    E._compare("主 arm", M, sm, "2b", C, sc, Y)
    for lab, X, sx in (("現行關係型層", R, sr), ("主 arm", M, sm)):
        A, B, common = E._align(X, sx, C, sc)
        if len(common) < 2:
            continue
        sa = np.nanmean([T.summ(a, Y)["ser"] for a in A], 0)
        sb = np.nanmean([T.summ(b, Y)["ser"] for b in B], 0)
        print(f"    種子平均的逐日 IC：2b − {lab} {np.nanmean(sb - sa):+.4f}，逐日 HAC p {T.hac_p(sb - sa):.4f}")
        ea, eb = T.summ(np.nanmean(A, 0), Y), T.summ(np.nanmean(B, 0), Y)
        print(f"    {len(common)} 顆集成：{lab} IC {ea['IC']:+.4f} / RankIC {ea['RIC']:+.4f}；"
              f"2b IC {eb['IC']:+.4f} / RankIC {eb['RIC']:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="方案二 b（proposal §78）的登記分析")
    ap.add_argument("--blocks", nargs="*", default=list("ABCD"), choices=list("ABCD"))
    ap.add_argument("--validate", action="store_true",
                    help="2b 指到現行關係型層、現行指到主 arm，只跑區塊 A（應重現 §71）")
    args = ap.parse_args()
    res = {}
    for fold, f in T.FOLDS.items():
        D = T.load_fold(f["cfg"], f["f2"])
        M, sm = T.preds(f["main"], "test", D["d_test"], D["codes"])
        R, sr = T.preds(E.ARMS[fold]["rel"], "test", D["d_test"], D["codes"])
        print(f"\n######## {fold}：測試窗 {D['d_test'][0]} ~ {D['d_test'][-1]}（{len(D['d_test'])} 天）")
        if args.validate:
            block_a(D, M, sm, R, sr, lab_ref="主 arm", lab_cand="現行關係型層")
            continue
        C, sc = T.preds(ARMS[fold], "test", D["d_test"], D["codes"])
        print(f"    主 arm {len(sm)} 顆、現行關係型層 {len(sr)} 顆、2b {len(sc)} 顆")
        if "A" in args.blocks:
            res[fold] = block_a(D, R, sr, C, sc) if len(sc) else {}
        if "B" in args.blocks and len(sc):
            block_b(D, M, sm, R, sr, C, sc)
        if "C" in args.blocks and len(sc):
            print("    （區塊 C、D 沿用 §71 的程式，輸出中的「關係型」指 2b）")
            E.block_c(D, {"rel": ARMS[fold]})
        if "D" in args.blocks and len(sc):
            P.block_abc(D, f, {"rel": ARMS[fold]})
            P.block_d(D, f, {"rel": ARMS[fold]})
    if "A" in args.blocks and not args.validate:
        verdict(res)


if __name__ == "__main__":
    main()
