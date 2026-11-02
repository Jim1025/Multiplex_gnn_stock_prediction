"""
sa_us_graph_eval.py — 主結果（SA）關掉美股層內圖 A₁ 的消融

arm：SA 的設定加 --graph-ablate empty_both（SA 本來就是 empty_l2，A₂ 由同產業關係卷積取代；
empty_both 再把 A₁ 只留 self-loop，GAT_L1 的參數與計算圖不變，只切斷美股之間的鄰居）。
兩折各 10 顆種子，tag tw50_noA1SA_s*、f2_noA1SA_s*。不是事前登記的實驗，為口試備詢而跑。

輸出（每折）：SA 與 noA1SA 的 IC / RankIC 種子平均與 SD、同種子配對 t p、種子平均逐日 IC 差的 HAC p。
--write：把種子平均與 SD 寫到 docs/figures/data/eval_us_graph_off.json，供 gen_eval_figures.py 畫消融圖。
--validate：把 noA1SA 換成 smktSA，SA / 單市場的平均應與主表相同（+0.1127 / +0.0250，第二折 +0.1611 / +0.0597）。

用法：
    .venv/bin/python scripts/sa_us_graph_eval.py [--write] [--validate]
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
import tw_lag_channel as T  # noqa: E402
import tw_relational_eval as E  # noqa: E402

PREFIX = {"第一折": "tw50", "第二折": "f2"}
FOLD_NO = {"第一折": 1, "第二折": 2}
N_SEEDS = 10
OUT_JSON = ROOT / "docs" / "figures" / "data" / "eval_us_graph_off.json"


def pat(fold: str, tag: str) -> str:
    return f"runs/**/*{PREFIX[fold]}_{tag}_s*/predictions/{{}}_predictions.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="SA 關掉美股層內圖 A₁ 的消融")
    ap.add_argument("--write", action="store_true", help=f"寫出 {OUT_JSON.relative_to(ROOT)}")
    ap.add_argument("--validate", action="store_true", help="noA1SA -> smktSA，對照主表")
    a = ap.parse_args()
    tag = "smktSA" if a.validate else "noA1SA"
    summary = {}
    for fold, f in T.FOLDS.items():
        D = T.load_fold(f["cfg"], f["f2"])
        M, sm = T.preds(pat(fold, "rgcnSA"), "test", D["d_test"], D["codes"])
        X, sx = T.preds(pat(fold, tag), "test", D["d_test"], D["codes"])
        print(f"\n######## {fold}：測試窗 {D['d_test'][0]} ~ {D['d_test'][-1]}（{len(D['d_test'])} 天）；"
              f"SA {len(sm)} 顆、{tag} {len(sx)} 顆")
        if not len(sx):
            continue
        A, B, common = E._align(M, sm, X, sx)           # A = SA、B = 消融
        if len(common) != N_SEEDS:
            print(f"    注意：預期 {N_SEEDS} 顆種子，實際 {len(common)} 顆")
        ra = [T.summ(x, D["y_test"]) for x in A]
        rb = [T.summ(x, D["y_test"]) for x in B]
        out = {"n": len(common)}
        for k, lab, key in (("IC", "IC", "ic"), ("RIC", "RankIC", "rk")):
            m_, b_ = np.array([r[k] for r in ra]), np.array([r[k] for r in rb])
            d = m_ - b_
            print(f"  {lab:6s} SA {m_.mean():+.4f} ± {m_.std(ddof=1):.4f}、{tag} {b_.mean():+.4f} ± {b_.std(ddof=1):.4f}；"
                  f"差 {d.mean():+.4f}（{int((d > 0).sum())}/{len(d)} 為正，配對 p {stats.ttest_rel(m_, b_).pvalue:.2e}）")
            out[key], out[key + "_sd"] = float(b_.mean()), float(b_.std(ddof=1))
            out["d_" + key], out["p_pair_" + key] = float(d.mean()), float(stats.ttest_rel(m_, b_).pvalue)
        sa_ser = np.nanmean([r["ser"] for r in ra], 0)
        sb_ser = np.nanmean([r["ser"] for r in rb], 0)
        out["p_hac_ic"] = T.hac_p(sa_ser - sb_ser)
        print(f"  種子平均的逐日 IC 差 {np.nanmean(sa_ser - sb_ser):+.4f}，逐日 HAC p {out['p_hac_ic']:.2e}")
        summary[FOLD_NO[fold]] = out
    if a.write and not a.validate:
        if len(summary) < 2 or any(v["n"] != N_SEEDS for v in summary.values()):
            sys.exit("兩折的 run 還沒全部完成，不寫出")
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({"arm": "noA1SA", "folds": summary}, ensure_ascii=False, indent=2) + "\n")
        print(f"\n-> {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
