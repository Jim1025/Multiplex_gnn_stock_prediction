"""tilt_vs_truth.py — `w̄` 學到的持久傾斜，跟真實報酬對不對得上（proposal §55.7(g)）

§55.7 證明的是模型**學到**一個跨種子可重現的持久橫截面傾斜，內容是偏多半導體。
本腳本問的是下一個問題：**那個傾斜是對的嗎？** 三個區塊分別回答：

  A. 對訓練窗對不對   corr(w̄, 真實 v̄)，分訓練 / 驗證 / 測試三段看衰減
  B. 對測試窗值多少   靜態 IC，與總 IC / 動態 IC 並排看尺度
  C. 有沒有真相可對   真實傾斜 v̄ 自己跨期穩不穩；跨折 +0.718 有多少是共用訓練資料

定義（與 §55.7 / §60.2a(k) 同）：
    z_t = (x_t − 當日均值) / 當日 sd
    w̄   = (1/T) Σ_t z(ŷ_t)     模型的持久傾斜
    v̄   = (1/T) Σ_t z(y_t)     **真實報酬**的持久傾斜（同一個算子，換成 y）

結論寫在 §55.7(g)：w̄ 是訓練窗真實傾斜的忠實副本（corr +0.62 / +0.46），
但**出了訓練窗就沒有值**（靜態 IC t = 1.29 / 0.39，兩折都不顯著），
而且**沒有持久真相可以對**（真實傾斜自己跨期 corr −0.194）。
這與 §60.2a(l) 的 P9 一致：加強靜態傾斜讓第二折 ICIR 掉 0.0450。

只讀 `data/processed/tw/*.csv` 的 log_return，不寫入任何資料層檔案。

用法：
    .venv/bin/python scripts/tilt_vs_truth.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import residual_fingerprint as rf             # noqa: E402  窗界（與 §55.9 同一套）
import static_tilt as stl                      # noqa: E402
from src.dataset.config import load_universe   # noqa: E402

SE50 = 1 / np.sqrt(47)      # n=50 的相關係數標準誤，與 §55.7 用同一把尺

# 窗界從各 arm 的 run 讀（rf.windows）：驗證窗 / 測試窗取自預測檔的實際日期，
# 訓練窗 = [第一個 graph snapshot, 驗證窗開始)，與模型實際的訓練樣本同一段。
# 初版把訓練窗下界放在資料起點 2019-01-02，多算約 65 天（§55.9(b) 同一個問題）。
FOLDS = {"第一折": dict(arm="tw50_betaF1nA2r1", fold2=False),
         "第二折": dict(arm="f2_best", fold2=True)}


def daily_z(arm: str, fold2: bool = False):
    """逐顆種子吐出 (Z[T, n], ticker 順序)。Z 是逐日橫截面標準化的預測。

    static_tilt.tilts() 只回傳時間平均，這裡要逐日的，所以自己讀一次。
    """
    import glob
    for d in sorted(glob.glob(str(ROOT / "runs" / "**" / f"*{arm}_s*"), recursive=True)):
        if not d.rsplit("_s", 1)[-1].isdigit():
            continue
        f = glob.glob(f"{d}/predictions/test_predictions.csv")
        if not f:
            continue
        df = pd.read_csv(f[0])
        df["target_date"] = df["target_date"].astype(str).str[:10]
        if fold2:
            df = df[df.target_date <= stl.CUT]
        H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
        yield zrow(H.to_numpy(float)), [str(c) for c in H.columns], list(H.index)


def zrow(A: np.ndarray) -> np.ndarray:
    m, sd = np.nanmean(A, 1, keepdims=True), np.nanstd(A, 1, keepdims=True)
    return (A - m) / np.where(sd > 1e-15, sd, np.nan)


def nc(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan


def tstat(x) -> tuple[float, int]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))), len(x)


def partial(a, b, c) -> float:
    """扣掉 c 之後 a 與 b 的偏相關。"""
    a, b, c = (np.asarray(x, float) for x in (a, b, c))
    r = lambda v: v - np.poly1d(np.polyfit(c, v, 1))(c)
    return float(np.corrcoef(r(a), r(b))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="w̄ 對不對得上真實報酬（§55.7(g)）")
    ap.add_argument("--universe", default="tw50")
    ap.add_argument("--strict", action="store_true",
                    help="敏感度：另外排除 2021-08-17（佔位列）與 4 個兩日報酬日（§55.9(h)）")
    a = ap.parse_args()
    drop = sorted(rf.PLACEHOLDER | rf.TWO_DAY) if a.strict else []
    warnings.filterwarnings("ignore")

    u = load_universe(a.universe)
    tw = [str(c) for c in u.tw_nodes]
    ind = {c: u.industry.get(c, "其他") for c in tw}
    semi = np.array([ind[c] == "半導體業" for c in tw])
    tel = np.array([ind[c] == "通信網路業" for c in tw])

    P = pd.DataFrame({c: (lambda d: d.set_index(d.Date.astype(str).str[:10])["log_return"])(
        pd.read_csv(ROOT / "data" / "processed" / "tw" / f"{c}.csv",
                    usecols=["Date", "log_return"])) for c in tw}).sort_index()
    print(f"規則：{'嚴格（敏感度）' if a.strict else '現行'}   "
          f"真實報酬面板  {P.shape[0]} 天 x {P.shape[1]} 檔   {P.index[0]} .. {P.index[-1]}")

    idx_all = np.asarray(P.index)

    def vbar(w, key):
        lo, hi = w[key]
        m = ((idx_all >= lo) & (idx_all < hi)) if key == "訓練窗" else \
            ((idx_all >= lo) & (idx_all <= hi))
        m &= ~np.isin(idx_all, drop)
        Z = zrow(P.loc[m].to_numpy(float))
        # 50 檔報酬全為 0 的日子（3 天休市、3 天資料商缺漏，§55.9(h)）z 是 NaN，
        # nanmean 本來就會略過；T 也不算它
        return np.nanmean(Z, 0), int(np.isfinite(Z).any(1).sum())

    for cfg in FOLDS.values():
        cfg["w"] = rf.windows(cfg["arm"], cfg["fold2"])

    W, YT = {}, {}
    for flab, cfg in FOLDS.items():
        _, Wm, cols = stl.tilts(cfg["arm"], cfg["fold2"])
        if Wm.size == 0:
            print(f"找不到 {cfg['arm']}")
            return
        idx = [cols.index(c) for c in tw]
        W[flab] = Wm.mean(0)[idx]
        YT[flab] = cfg["w"]["測試窗"]

    print("\n" + "=" * 78)
    print("A. 模型傾斜 w̄ 對真實傾斜 v̄：訓練 -> 驗證 -> 測試的衰減")
    print("=" * 78)
    print("\n  w̄ 是固定的（每折一個），變的是拿去比的那段真實報酬。")
    for flab, cfg in FOLDS.items():
        w = W[flab]
        print(f"\n  {flab}   模型 Δw̄(半導體) {w[semi].mean()-w[~semi].mean():+.4f}"
              f"   Δw̄(電信) {w[tel].mean()-w[~tel].mean():+.4f}")
        for lab in ("訓練窗", "驗證窗", "測試窗"):
            lo, hi = cfg["w"][lab]
            v, n = vbar(cfg["w"], lab)
            r = nc(w, v)
            print(f"    {lab} {lo}..{hi}  T={n:4d}   corr(w̄, v̄) {r:+.4f}  z {r/SE50:+.2f}"
                  f"   |   真實 Δz 半導體 {v[semi].mean()-v[~semi].mean():+.4f}"
                  f"  電信 {v[tel].mean()-v[~tel].mean():+.4f}")

    print("\n" + "=" * 78)
    print("B. 靜態傾斜在測試窗值多少：靜態 IC 與總 IC / 動態 IC 並排")
    print("=" * 78)
    print("\n  靜態 IC = mean_t corr(w̄, y_t)      每天拿同一個固定向量去押")
    print("  動態 IC = mean_t corr(d_t, y_t)     d_t = z(ŷ_t) − w̄，扣掉傾斜後的當日部分")
    print("  兩者不相加（§55.7 的靜態/動態拆解）。")
    for flab, cfg in FOLDS.items():
        w = W[flab]
        runs = list(daily_z(cfg["arm"], cfg["fold2"]))
        cols, dates = runs[0][1], runs[0][2]
        idx = [cols.index(c) for c in tw]
        Zh = np.mean([Z for Z, _, _ in runs], axis=0)[:, idx]
        # 用預測檔自己的日期去對真實報酬，不靠兩邊長度剛好一樣
        Yv = P.reindex(dates)[tw].to_numpy(float)
        Yv[np.isin(np.asarray(dates), drop)] = np.nan      # --strict 才有作用
        D = Zh - w[None, :]
        tot = np.array([nc(Zh[t], Yv[t]) for t in range(len(Yv))])
        sta = np.array([nc(w, Yv[t]) for t in range(len(Yv))])
        dyn = np.array([nc(D[t], Yv[t]) for t in range(len(Yv))])
        print(f"\n  {flab}  T={len(Yv)}")
        for lab, s in (("總 IC  ", tot), ("靜態 IC", sta), ("動態 IC", dyn)):
            t, n = tstat(s)
            print(f"    {lab}  {np.nanmean(s):+.5f}   t = {t:+5.2f}"
                  f"   正號 {int(np.nansum(s > 0))}/{n}")
        for gl, m in (("半導體 6 檔", semi), ("電信   3 檔", tel)):
            dz = np.nanmean(zrow(Yv)[:, m], 1) - np.nanmean(zrow(Yv)[:, ~m], 1)
            t, n = tstat(dz)
            print(f"    [{gl}] 真實 Δz {np.nanmean(dz):+.4f}   t = {t:+5.2f}"
                  f"   正號 {int(np.nansum(dz > 0))}/{n}")

    print("\n" + "=" * 78)
    print("C. 有沒有持久真相可以對；跨折 +0.718 有多少是共用訓練資料")
    print("=" * 78)
    v_t1, _ = vbar(FOLDS["第一折"]["w"], "測試窗")
    v_t2, _ = vbar(FOLDS["第二折"]["w"], "測試窗")
    r = nc(v_t1, v_t2)
    print(f"\n  真實傾斜自己跨期  corr(v̄_第二折測試窗, v̄_第一折測試窗) {r:+.4f}   z {r/SE50:+.2f}")
    print("  -> 真實的橫截面持久傾斜**本身就不持久**，沒有穩定的靶可以打。")
    v_sh, n_sh = vbar(FOLDS["第二折"]["w"], "訓練窗")      # 共用段 = 第二折的訓練窗
    v_tr1, n_tr1 = vbar(FOLDS["第一折"]["w"], "訓練窗")
    w1, w2 = W["第一折"], W["第二折"]
    print(f"\n  共用訓練段 T={n_sh}，佔第一折訓練窗 {n_tr1} 天的 {100*n_sh/n_tr1:.0f}%")
    print(f"  corr(兩折訓練窗的真實傾斜)              {nc(v_tr1, v_sh):+.4f}")
    print(f"  corr(w̄_f1, w̄_f2)                       {nc(w1, w2):+.4f}   <- §55.7(a) 報的")
    print(f"  扣掉共用段真實傾斜後的偏相關             {partial(w1, w2, v_sh):+.4f}")
    print("  -> 跨折一致性**不是**獨立重現：兩折的訓練窗重疊八成。")
    print("     扣掉共用段之後還剩得下來，所以也不是全部由共用資料解釋。")
    print(f"\n  n=50 的相關係數標準誤 {SE50:.3f}")


if __name__ == "__main__":
    main()
