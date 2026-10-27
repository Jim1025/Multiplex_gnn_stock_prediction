"""
sa_single_market_eval.py — SA 的單市場消融（proposal §83 第 1 步）的事前登記分析

登記見 §83（reg_sa_single_market.md，計算前寫下並記 SHA-256）。本檔在第一個 run 開始之前寫好並記 SHA-256，
run 全部完成後才跑。口徑與 §41、§64.8 相同（共用 tw_lag_channel 的載入與統計）：

  主判準    SA − SA_smkt 的 test IC，每一折：同種子配對 t p < 0.05，且種子平均逐日 IC 差的 HAC p < 0.05
  一併報    跨市場佔 SA 表現的比例（IC、RankIC）、Welch p
  描述      SA_smkt − 本版單市場（同種子配對）：沒有跨市場資訊時，同產業關係在台股側加多少

--validate：把 SA 指到本版、SA_smkt 指到本版單市場，重現 §41 / §64.8（主 arm − 單市場 +0.1004 / +0.1201）。

只讀 runs/ 的預測檔，不寫入任何檔案。
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
import tw_lag_channel as T  # noqa: E402
import tw_relational_eval as E  # noqa: E402

PREFIX = {"第一折": "tw50", "第二折": "f2"}
N_SEEDS = 10


def pat(fold: str, tag: str) -> str:
    return f"runs/**/*{PREFIX[fold]}_{tag}_s*/predictions/{{}}_predictions.csv"


def per_seed(D, A, sa, B, sb):
    X, Y, common = E._align(A, sa, B, sb)
    rx = [T.summ(x, D["y_test"]) for x in X]
    ry = [T.summ(y, D["y_test"]) for y in Y]
    return X, Y, common, rx, ry


def main() -> None:
    ap = argparse.ArgumentParser(description="SA 的單市場消融（proposal §83）")
    ap.add_argument("--validate", action="store_true", help="SA -> 本版、SA_smkt -> 本版單市場，重現 §41 / §64.8")
    a = ap.parse_args()
    res = {}
    for fold, f in T.FOLDS.items():
        D = T.load_fold(f["cfg"], f["f2"])
        full = f["main"] if a.validate else pat(fold, "rgcnSA")
        smkt = pat(fold, "smktA") if a.validate else pat(fold, "smktSA")
        M, sm = T.preds(full, "test", D["d_test"], D["codes"])
        S, ss = T.preds(smkt, "test", D["d_test"], D["codes"])
        print(f"\n######## {fold}：測試窗 {D['d_test'][0]} ~ {D['d_test'][-1]}（{len(D['d_test'])} 天）；"
              f"完整 {len(sm)} 顆、單市場 {len(ss)} 顆")
        if not len(ss):
            continue
        X, Y, common, rx, ry = per_seed(D, S, ss, M, sm)          # X = 單市場、Y = 完整
        if len(common) != N_SEEDS:
            print(f"    注意：登記的是 {N_SEEDS} 顆種子，實際 {len(common)} 顆")
        out = {}
        for k, lab in (("IC", "IC"), ("RIC", "RankIC")):
            s_, m_ = np.array([r[k] for r in rx]), np.array([r[k] for r in ry])
            d = m_ - s_
            print(f"  {lab:6s} 完整 {m_.mean():+.4f}、單市場 {s_.mean():+.4f}；差 {d.mean():+.4f}（{int((d > 0).sum())}/{len(d)} 為正，"
                  f"配對 p {stats.ttest_rel(m_, s_).pvalue:.2e}，Welch p {stats.ttest_ind(m_, s_, equal_var=False).pvalue:.2e}）；"
                  f"跨市場佔 {d.mean() / m_.mean() * 100:.0f}%")
            out[k] = dict(d=float(d.mean()), p=float(stats.ttest_rel(m_, s_).pvalue))
        sx = np.nanmean([r["ser"] for r in rx], 0)
        sy = np.nanmean([r["ser"] for r in ry], 0)
        out["hac"] = T.hac_p(sy - sx)
        print(f"  種子平均的逐日 IC 差 {np.nanmean(sy - sx):+.4f}，逐日 HAC p {out['hac']:.2e}")
        res[fold] = out
        if not a.validate:
            B, sb = T.preds(pat(fold, "smktA"), "test", D["d_test"], D["codes"])
            _, _, common2, rb, rs = per_seed(D, B, sb, S, ss)       # rb = 本版單市場、rs = SA 單市場
            for k, lab in (("IC", "IC"), ("RIC", "RankIC")):
                b_, s_ = np.array([r[k] for r in rb]), np.array([r[k] for r in rs])
                print(f"  描述：SA 單市場 − 本版單市場 {lab} {np.mean(s_ - b_):+.4f}（配對 p {stats.ttest_rel(s_, b_).pvalue:.4f}，"
                      f"{int((s_ > b_).sum())}/{len(s_)} 為正；本版單市場 {b_.mean():+.4f}）")
    if a.validate or len(res) < 2:
        return
    ok = all(v["IC"]["d"] > 0 and v["IC"]["p"] < 0.05 and v["hac"] < 0.05 for v in res.values())
    print(f"\n== 判定：{'兩折都通過' if ok else '未全部通過'}（主判準：每折 SA − SA_smkt 的 IC，配對 p 與逐日 HAC p 都 < 0.05）")


if __name__ == "__main__":
    main()
