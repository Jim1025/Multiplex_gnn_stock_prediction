"""hybrid_channel.py — P12：MAGNET + 凍結的完整線性通道（寬路徑）（proposal §63，事前登記）

§62 的 ④ 只取了 per-target ridge 的台股區塊；P12 取完整的設計矩陣（美股 30 + 台股 50 的昨日
報酬，與 KTW+ 相同），以封閉解凍結後疊在 MAGNET 的輸出上。依據與限制見 §63.1。

區塊：

  A. 登記值   寬路徑的 alpha、通道自身的 val IC、組合權重 w（只用 train / val）
  B. P12a     事後組合：主 arm 每顆種子 + w·z(ĉ)，不重訓。**不是確認性檢定**——§63.1 的探索
              就是在同樣的 test 窗上做的，這裡只用登記版的規則重算一次
  C. P12b     內建重訓（寬路徑在 forward 裡、凍結），20 個 run 完成後才跑
  D. 判定     §63.4 的判準（對象是 P12b）

比較對象（權重都在 val 上以同一規則選，test 只做評估）：

  KTW+         最強線性 baseline（runs/linear、runs_f2 的預測檔）
  線性通道單獨   寬路徑本身
  線性 + 線性    a·z(KTW+) + (1−a)·z(ĉ)，a 在 val 選——把 GNN 換成另一個線性模型的對照，
               混合模型對它的增量才是 GNN 的貢獻

逐日 z 組合 z(ŷ^GNN) + w·z(ĉ) 與模型內的 ŷ^GNN + w·sd_t(ŷ^GNN)·z_t(ĉ) 逐日 IC 等價（§62.7）。
只讀 MultiplexDataset、runs/、runs_f2/ 的預測檔與 meta.json，不寫入任何檔案。

用法：
    .venv/bin/python scripts/hybrid_channel.py --blocks A B     # 登記值與 P12a
    .venv/bin/python scripts/hybrid_channel.py --blocks C D     # P12b 的 run 完成之後
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
sys.path.insert(0, str(ROOT / "scripts"))

from tw_lag_channel import (FOLDS, GRID, MDE, _mean_corr, disp, gnn_part, hac_p,  # noqa: E402
                            load_fold, pick_w, preds, ridge, summ, zrow)

# §63.2 登記的值（區塊 A 會重算並核對；不符就停）
REG = {"第一折": dict(alpha=1000.0, w=0.7), "第二折": dict(alpha=1000.0, w=0.7)}
P12B = {"第一折": "runs/**/*tw50_p12b_s*/predictions/{}_predictions.csv",
        "第二折": "runs/**/*f2_p12b_s*/predictions/{}_predictions.csv"}
BLOCKS = "ABCD"


def wide(D) -> dict:
    """寬路徑：與 src/models/tw_lag_channel.py（inputs = l1_l2）同協定的 per-target ridge。"""
    return ridge(D, "US30+TW50", "zscore", intercept=False)


def lin_lin(D, c) -> tuple[np.ndarray, float]:
    """線性 + 線性：KTW+ 與寬路徑的逐日 z 凸組合，a 以 val 平均 IC 選（網格同 w）。"""
    k = ridge(D, "US30+TW50", "rank", intercept=True)          # 與 KTW+ 預測檔逐位元相同的協定
    a = float(GRID[int(np.argmax([summ(g * zrow(k["val"]) + (1 - g) * zrow(c["val"]), D["y_val"])["IC"]
                                  for g in GRID]))])
    return a * zrow(k["test"]) + (1 - a) * zrow(c["test"]), a


def hybrid(P, c_test, w) -> np.ndarray:
    return np.stack([zrow(p) + w * zrow(c_test) for p in P])


def _pair(lab, a, b):
    d = b - a
    p = stats.ttest_rel(b, a).pvalue
    print(f"      {lab:24s} {a.mean():.4f} -> {b.mean():.4f}  差 {d.mean():+.4f}"
          f"（配對 p {p:.4f}，{int((d > 0).sum())}/{len(d)} 為正）")
    return d.mean(), p


def _daily(lab, ser_h, ser_ref):
    d = ser_h - ser_ref
    p = hac_p(d)
    print(f"      對 {lab:10s} 逐日 IC 差 {np.nanmean(d):+.4f}，HAC p {p:.4f}")
    return float(np.nanmean(d)), p


def evaluate(D, A, H, K, c, LL, label) -> dict:
    """A = 主 arm 逐種子、H = 混合模型逐種子（同種子順序）。印 §63.4 要的數字。"""
    Y = D["y_test"]
    ra, rh = [summ(a, Y) for a in A], [summ(h, Y) for h in H]
    print(f"    [{label}] 種子 {len(H)} 顆")
    out = {}
    for key, lab in (("ICIR", "ICIR"), ("IC", "IC"), ("RIC", "RankIC"), ("RICIR", "RankICIR"), ("sd", "sd(IC)")):
        out[f"d{key}"], out[f"p{key}"] = _pair(lab + "（主 arm -> 混合）", np.array([r[key] for r in ra]),
                                               np.array([r[key] for r in rh]))
    _pair("離散比（主 arm -> 混合）", np.array([disp(a, Y) for a in A]), np.array([disp(h, Y) for h in H]))
    sk = summ(K, Y)
    icir_h = np.array([r["ICIR"] for r in rh])
    print(f"      ICIR 對 KTW+（{sk['ICIR']:.4f}）：差 {icir_h.mean() - sk['ICIR']:+.4f}，"
          f"{int((icir_h > sk['ICIR']).sum())}/{len(icir_h)} 高於（單樣本 p {stats.ttest_1samp(icir_h, sk['ICIR']).pvalue:.4f}）")
    ser_h = np.nanmean([r["ser"] for r in rh], 0)
    out["d_ktw"], out["p_ktw"] = _daily("KTW+", ser_h, sk["ser"])
    _daily("線性通道單獨", ser_h, summ(c["test"], Y)["ser"])
    out["d_ll"], out["p_ll"] = _daily("線性 + 線性", ser_h, summ(LL, Y)["ser"])
    out["icir"], out["icir_ktw"] = float(icir_h.mean()), float(sk["ICIR"])
    out["H"] = H
    return out


def load(D, M, seeds, pattern):
    P, s = preds(pattern, "test", D["d_test"], D["codes"])
    common = sorted(set(seeds) & set(s))
    return (M["test"][[seeds.index(x) for x in common]], P[[s.index(x) for x in common]] if len(s) else P,
            common, s)


def block_a(D, M, fold):
    c = wide(D)
    w = pick_w(np.nanmean(M["val"], 0), c["val"], D["y_val"])
    reg = REG[fold]
    ok = np.isclose(c["alpha"], reg["alpha"]) and np.isclose(w, reg["w"])
    print(f"[A] 寬路徑：alpha {c['alpha']:.4g}，通道自身 val IC {c['val_IC']:+.4f}；組合權重 w = {w:.1f}"
          f"（登記值 alpha {reg['alpha']:.4g}、w {reg['w']:.1f}：{'相符' if ok else '不符'}）")
    if not ok:
        raise SystemExit("登記值不符，停止（§63.2）")
    return c, w


def block_b(D, M, seeds, K, c, w, LL, a):
    print(f"[B] P12a：事後組合，w = {w:.1f}（不重訓；不是確認性檢定，見 §63.1）；線性 + 線性的 a = {a:.1f}")
    return evaluate(D, M["test"], hybrid(M["test"], c["test"], w), K, c, LL, "P12a")


def block_c(D, M, seeds, K, c, w, LL, fold, res_b):
    A, P, common, s_b = load(D, M, seeds, P12B[fold])
    print(f"[C] P12b：內建重訓（寬路徑凍結在 forward 裡），找到 {len(s_b)} 顆種子")
    if len(s_b) == 0:
        print("    尚無 run")
        return None
    files = sorted(glob.glob(str(ROOT / P12B[fold].format("test")), recursive=True))
    metas = [json.load(open(Path(f).parents[1] / "meta.json")).get("tw_lag_channel") or {} for f in files]
    same = all(m.get("inputs") == "l1_l2" and np.isclose(m.get("alpha", np.nan), c["alpha"])
               and np.isclose(m.get("val_ic", np.nan), c["val_IC"]) and np.isclose(m.get("blend_w", np.nan), w)
               for m in metas)
    print(f"    健全性：{len(metas)} 個 run 的 inputs / alpha / 通道 val IC / w 與登記值相同：{'是' if same else '否'}")
    out = evaluate(D, A, P, K, c, LL, "P12b")
    # 內建 vs 事後組合（同種子）
    Ha = hybrid(A, c["test"], w)
    out["d_ab"], out["p_ab"] = _pair("ICIR（P12a -> P12b）", np.array([summ(h, D["y_test"])["ICIR"] for h in Ha]),
                                     np.array([summ(p, D["y_test"])["ICIR"] for p in P]))
    # 機制：GNN 部分是否改學線性通道抓不到的殘差
    zc = zrow(c["test"])
    U = np.stack([gnn_part(p, zc, w) for p in P])
    print("    機制：P12b 的 GNN 部分（由 ŷ = g + w·sd(g)·z(ĉ) 反解）對主 arm")
    ca = np.array([_mean_corr(a, c["test"]) for a in A])
    cu = np.array([_mean_corr(u, c["test"]) for u in U])
    out["d_corr"], out["p_corr"] = _pair("corr(GNN 部分, ĉ)", ca, cu)
    _pair("GNN 部分 IC", np.array([summ(a, D["y_test"])["IC"] for a in A]),
          np.array([summ(u, D["y_test"])["IC"] for u in U]))
    _pair("GNN 部分 ICIR", np.array([summ(a, D["y_test"])["ICIR"] for a in A]),
          np.array([summ(u, D["y_test"])["ICIR"] for u in U]))
    return out


def verdict(res: dict):
    """§63.4 的判準（P12b）。單折顯著不算；兩折都要成立。"""
    if len(res) < 2 or any(v is None for v in res.values()):
        return
    v = list(res.values())
    main = all(x["dICIR"] > 0 and x["pICIR"] < 0.05 and x["dIC"] >= -MDE for x in v)
    s1 = all(x["d_ktw"] > 0 and x["p_ktw"] < 0.05 for x in v)
    s1_dir = all(x["icir"] > x["icir_ktw"] for x in v)
    s2 = all(x["d_ll"] > 0 and x["p_ll"] < 0.05 for x in v)
    s3 = ("內建較好" if all(x["d_ab"] > 0 and x["p_ab"] < 0.05 for x in v) else
          "內建較差" if all(x["d_ab"] < 0 and x["p_ab"] < 0.05 for x in v) else "無一致差異")
    s4 = all(x["d_corr"] < 0 and x["p_corr"] < 0.05 for x in v)
    yn = lambda b: "通過" if b else "未通過"
    print("\n== P12b 判定（§63.4）")
    print(f"   主判準（ICIR 對主 arm，兩折 p < 0.05；IC 護欄 −{MDE}）：{yn(main)}")
    print(f"   S1 對 KTW+ 的逐日 IC（兩折 HAC p < 0.05）：{yn(s1)}；ICIR 兩折同向高於 KTW+：{'是' if s1_dir else '否'}")
    print(f"   S2 對線性 + 線性的逐日 IC（兩折 HAC p < 0.05）：{yn(s2)}")
    print(f"   S3 內建 vs 事後組合：{s3}")
    print(f"   S4 GNN 部分與 ĉ 的相關兩折都顯著下降（改學殘差）：{'是' if s4 else '否'}")
    if not main:
        print("   -> 判讀：內建後增益消失；改用 P12a 的形式報，或退回原版（§63.4 判讀表）")
    elif not s1:
        print("   -> 判讀：增益只對自家主 arm 成立；退回原版重新思考（§63.4 判讀表）")
    elif not s2:
        print("   -> 判讀：混合模型勝過 KTW+，但 GNN 的增量只能寫單折；主張降一級（§63.4 判讀表）")
    else:
        print("   -> 判讀：混合模型成立，主張依 §63.5 改寫；主 arm 與 2026 確認一起決定（§63.4 判讀表）")


def main() -> None:
    ap = argparse.ArgumentParser(description="P12：MAGNET + 凍結的完整線性通道（proposal §63）")
    ap.add_argument("--blocks", nargs="*", default=["A"], choices=list(BLOCKS))
    args = ap.parse_args()
    res_c = {}
    for fold, f in FOLDS.items():
        D = load_fold(f["cfg"], f["f2"])
        Mte, seeds = preds(f["main"], "test", D["d_test"], D["codes"])
        Mva, _ = preds(f["main"], "val", D["d_val"], D["codes"])
        M = {"test": Mte, "val": Mva}
        K = preds(f["ktw"], "", D["d_test"], D["codes"])[0][0]
        print(f"\n######## {fold}：測試窗 {D['d_test'][0]} ~ {D['d_test'][-1]}（{len(D['d_test'])} 天），"
              f"主 arm {len(seeds)} 顆種子")
        c, w = block_a(D, M, fold)
        LL, a = lin_lin(D, c)
        res_b = block_b(D, M, seeds, K, c, w, LL, a) if "B" in args.blocks else None
        if "C" in args.blocks or "D" in args.blocks:
            res_c[fold] = block_c(D, M, seeds, K, c, w, LL, fold, res_b)
    if "D" in args.blocks and res_c:
        verdict(res_c)


if __name__ == "__main__":
    main()
