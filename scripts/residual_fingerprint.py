"""residual_fingerprint.py — 跨市場殘差領先指紋：B（模型學出）的資料端對照

圖①（§55.8）畫的是 B 的欄與欄之間的相似度。B 從零學起、產業標籤從未進入訓練，
而資料層只建市場**內**的圖（A₁、A₂）與 7 組寫死的 ADR 配對——**跨市場結構從未
餵給模型**。本腳本直接從報酬量出同一個概念，拿來對照 B 學到的是不是資料裡的結構。

定義（與 B 的分析同構）：
  1. 兩邊都扣當日橫截面均值（美股端對應 h₁ᵢ − h̄₁；台股端的共同部分由 γ 承擔）
  2. 美股 t−1 -> 台股 t（兩市共同交易日的前一日）
  3. F[i, j] = corr(台股殘差_j,t , 美股殘差_i,t−1)，每檔台股一條 30 維剖面
  4. 剖面跨 30 檔美股去均值再單位化（與 B 同一個零空間），兩兩內積 = 相關
  5. 產業區塊均值、分離度、排列檢定，與 beta_industry_clustering.separation() 同一套

對照：
  Mantel r    B 的兩兩相似度 vs 指紋的兩兩相似度（1,225 對，排列 2,000 次）
  指紋分別算在訓練窗 / 驗證窗 / 測試窗。**測試窗是該折模型從未看過的期間**，
  用它比才不會被說成「模型當然會複製訓練資料」。
  另附 Ŵ（ridge 估的 30x50 殘差耦合矩陣，alpha 選在驗證窗的橫截面 IC）當穩健性。

事先登記的判準（2026-09-27，第二折執行前寫）：
  主判準  第二折 Mantel(B, 測試窗指紋) p < 0.05 且 r >= +0.17（第一折 +0.3369 的一半）
  次判準  測試窗指紋本身的產業分離 p < 0.05；B 的產業分離 p < 0.05
  失敗    r <= 0 或 p >= 0.05
  介於之間（p < 0.05 但 r < +0.17）：成立但較弱，兩折量級並報

只讀 data/processed 與 runs/ 的預測檔、checkpoint，不寫入任何資料層檔案。

用法：
    .venv/bin/python scripts/residual_fingerprint.py
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import beta_industry_clustering as bic        # noqa: E402
from src.dataset.config import load_universe  # noqa: E402

CUT = "2024-12-25"          # 第二折評估窗上界，與 results_table.py 同值
N_PERM_SEP = 1000
N_PERM_MANTEL = 2000
R_MIN = 0.17                # 主判準的量級門檻：第一折 +0.3369 的一半
FOLDS = {"第一折": ("tw50_betaF1nA2r1", False), "第二折": ("f2_best", True)}


def universe():
    u = load_universe("tw50")
    tw = [str(c) for c in u.tw_nodes]
    us = [str(c) for c in u.us_nodes]
    return tw, us, [u.industry.get(c, "其他") for c in tw]


def returns_panel(tw, us, residual: bool = True):
    def panel(codes, sub):
        d = {}
        for c in codes:
            x = pd.read_csv(ROOT / "data" / "processed" / sub / f"{c}.csv",
                            usecols=["Date", "log_return"])
            d[c] = x.set_index(x.Date.astype(str).str[:10])["log_return"]
        return pd.DataFrame(d).sort_index()
    PT, PU = panel(tw, "tw"), panel(us, "adr")
    dates = list(PT.index.intersection(PU.index))
    A = PT.loc[dates][tw].to_numpy(float)
    U = PU.loc[dates][us].to_numpy(float)
    # 兩邊都扣當日橫截面均值。residual=False 只給 --detail 的反例用：
    # 不扣大盤的領先相關被全市場共通的型態主導，拿來對照 B 會得到錯的結論。
    Ar = A - np.nanmean(A, 1, keepdims=True) if residual else A
    Ur = U - np.nanmean(U, 1, keepdims=True) if residual else U
    Y, X, D = Ar[1:], Ur[:-1], np.array(dates[1:])    # 美股 t−1 -> 台股 t
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    # data/processed/tw 裡有 6 天 50 檔成交量全為 0、報酬全為 0，且 is_imputed 標為
    # False（2019-09-09、2021-04-06、2022-02-04、2023-01-18、2024-10-31、2025-08-01）。
    # 那是休市日，不是市場觀測值。資料層不動，只在分析端排除。
    ok &= np.nanstd(A[1:], 1) > 1e-12
    return Y, X, D, ok


def windows(arm: str, fold2: bool) -> dict[str, tuple[str, str]]:
    """窗界從該 arm 的 run 讀，不寫死。

    訓練窗 = [第一個 graph snapshot 的日期, 驗證窗開始)，與模型實際吃到的訓練樣本同一段。
    （資料從 2019-01-02 開始，但第一個 snapshot 要等 60 日相關窗暖機，是 2019-04-11。
    早期版本把訓練窗下界放在資料起點，多算了約 65 天，「同期」的比較因此不精確。）
    """
    run = sorted(bic.find_seeds(arm).values())[0]
    cfg = yaml.safe_load(open(f"{run}/config_snapshot.yaml", encoding="utf-8"))
    snap = sorted(glob.glob(str(ROOT / cfg["data"]["snapshot_dir"] / "graph_*.pt")))
    first = Path(snap[0]).stem.replace("graph_", "")
    rd = lambda k: pd.read_csv(f"{run}/predictions/{k}_predictions.csv")["target_date"] \
        .astype(str).str[:10]
    v, t = rd("val"), rd("test")
    if fold2:
        t = t[t <= CUT]
    return {"訓練窗": (first, v.min()), "驗證窗": (v.min(), v.max()),
            "測試窗": (t.min(), t.max())}


def in_window(D, w, key):
    lo, hi = w[key]
    return ((D >= lo) & (D < hi)) if key == "訓練窗" else ((D >= lo) & (D <= hi))


def unit_cols(W: np.ndarray) -> np.ndarray:
    W = W - W.mean(0, keepdims=True)
    return W / np.linalg.norm(W, axis=0, keepdims=True)


def fingerprint(Y, X, mask) -> np.ndarray:
    """F[i, j] = corr(台股殘差_j,t , 美股殘差_i,t−1)，[30, 50]。"""
    Yw, Xw = Y[mask], X[mask]
    Yz = (Yw - Yw.mean(0)) / Yw.std(0)
    Xz = (Xw - Xw.mean(0)) / Xw.std(0)
    return Xz.T @ Yz / len(Yw)


def similarity(F: np.ndarray) -> np.ndarray:
    Fn = unit_cols(F)
    return Fn.T @ Fn


def b_similarity(arm: str, fine: list[str]):
    mats, Bs = [], []
    for _, d in sorted(bic.find_seeds(arm).items(), key=lambda kv: int(kv[0])):
        B = bic.load_B(d)
        if B is not None:
            mats.append(bic.separation(B, fine)[3])
            Bs.append(B)
    return np.mean(mats, 0), Bs


def sep_stats(Mx, fine, n_perm=N_PERM_SEP):
    iu = np.triu_indices(len(fine), 1)
    lab = np.array(fine)
    def s_of(l):
        same = (l[:, None] == l[None, :])[iu]
        v = Mx[iu]
        return v[same].mean() - v[~same].mean()
    obs = s_of(lab)
    rng = np.random.default_rng(42)
    null = np.array([s_of(rng.permutation(lab)) for _ in range(n_perm)])
    return obs, (1 + (null >= obs).sum()) / (1 + n_perm)


def mantel(Ma, Mb, n_perm=N_PERM_MANTEL):
    n = len(Ma)
    iu = np.triu_indices(n, 1)
    r = float(np.corrcoef(Ma[iu], Mb[iu])[0, 1])
    rng = np.random.default_rng(0)
    c = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        c += np.corrcoef(Ma[iu], Mb[np.ix_(p, p)][iu])[0, 1] >= r
    return r, (1 + c) / (1 + n_perm)


def ridge_coupling(Y, X, D, ok, w) -> tuple[np.ndarray, float, float]:
    """Ŵ：30 檔美股殘差一起進的多輸出 ridge，alpha 選在驗證窗的橫截面 IC。"""
    tr, va = in_window(D, w, "訓練窗") & ok, in_window(D, w, "驗證窗") & ok
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xs = (X - mu) / sd
    best = None
    for a in np.logspace(-1, 6, 15):
        m = Ridge(alpha=a).fit(Xs[tr], Y[tr])
        P, Yv = m.predict(Xs[va]), Y[va]
        ic = float(np.nanmean([np.corrcoef(P[t], Yv[t])[0, 1] for t in range(len(P))]))
        if not np.isfinite(ic):
            raise ValueError(f"alpha={a:g} 的驗證窗 IC 不是有限值")
        if best is None or ic > best[1]:
            best = (a, ic, m.coef_.T)
    return best[2], best[0], best[1]


def alignment(Bs, F) -> np.ndarray:
    """逐檔：B 的欄（種子平均後的單位方向）與指紋欄的 30 維方向相關。"""
    PB = np.mean([unit_cols(B) for B in Bs], 0)
    Fn = unit_cols(F)
    return np.array([np.corrcoef(PB[:, j], Fn[:, j])[0, 1] for j in range(F.shape[1])])


def analyse(fold: str, arm: str, fold2: bool, Y, X, D, ok, fine) -> dict:
    w = windows(arm, fold2)
    M_B, Bs = b_similarity(arm, fine)
    out = {"w": w, "M_B": M_B, "n_seed": len(Bs), "B_sep": sep_stats(M_B, fine)}
    for key in ("訓練窗", "驗證窗", "測試窗"):
        mk = in_window(D, w, key) & ok
        F = fingerprint(Y, X, mk)
        M_F = similarity(F)
        out[key] = dict(T=int(mk.sum()), F=F, M=M_F, sep=sep_stats(M_F, fine),
                        mantel=mantel(M_B, M_F), align=alignment(Bs, F))
    for key in ("驗證窗", "測試窗"):
        out[key]["stab"] = mantel(out["訓練窗"]["M"], out[key]["M"])[0]
    W, a, ic = ridge_coupling(Y, X, D, ok, w)
    M_W = similarity(W)
    out["ridge"] = dict(alpha=a, ic=ic, sep=sep_stats(M_W, fine),
                        vs_B=mantel(M_B, M_W), vs_F=mantel(out["訓練窗"]["M"], M_W)[0])
    return out


def verdict(r: float, p: float) -> str:
    if r <= 0 or p >= 0.05:
        return "失敗"
    return "成立" if r >= R_MIN else "成立但較弱"


NONBANK = {"2881", "2882", "2883", "2885"}   # 富邦金、國泰金、開發金（凱基）、元大金；事先指定


def per_seed(arm: str, fine: list[str]) -> list[np.ndarray]:
    return [bic.separation(B, fine)[3] for B in b_similarity(arm, fine)[1]]


def detail(res: dict, Y, X, D, ok, tw, us, fine) -> None:
    """圖①的讀法（§55.9(g)）：三個概念的區別、兩個偏低格子的成因、非對角格的分布。

    全部用第一折、對齊後的訓練窗（與模型實際的訓練樣本同一段）；8150 另附第二折。
    """
    import itertools

    import torch
    from scipy.stats import spearmanr

    r1 = res["第一折"]
    mk = in_window(D, r1["w"], "訓練窗") & ok
    F, M_F, M_B = r1["訓練窗"]["F"], r1["訓練窗"]["M"], r1["M_B"]
    Bs = b_similarity(FOLDS["第一折"][0], fine)[1]
    PB = np.mean([unit_cols(B) for B in Bs], 0)                 # 種子平均後的方向
    PC = (PB / np.linalg.norm(PB, axis=0)).T @ (PB / np.linalg.norm(PB, axis=0))
    Fn = unit_cols(F)
    TWC = np.corrcoef(Y[mk].T)                                   # ① 台股殘差同日相關
    n = len(tw)
    align = np.array([np.corrcoef(PB[:, j], Fn[:, j])[0, 1] for j in range(n)])
    norm = np.array([np.mean([np.linalg.norm(B[:, j] - B[:, j].mean()) for B in Bs])
                     for j in range(n)])
    thr = 2 / np.sqrt(mk.sum())
    groups = sorted({g for g in fine if fine.count(g) >= 2})
    ix = {g: [j for j in range(n) if fine[j] == g] for g in groups}
    wi = lambda Mx, g: float(np.mean([Mx[a, b] for a, b in itertools.combinations(ix[g], 2)]))

    print("\n" + "=" * 78)
    print("G1. 三個概念（第一折訓練窗）：① 國內同質度 ② 殘差領先指紋 ③ B")
    print("=" * 78)
    print(f"  {'產業':<10s} {'n':>3s} {'①同質':>7s} {'②指紋':>7s} {'③B':>7s} {'③種子平均':>9s}"
          f" {'B↔指紋':>7s} {'有資訊節點':>9s}")
    rows = []
    for g in sorted(groups, key=lambda g: -wi(M_B, g)):
        info = np.median([(np.abs(F[:, j]) > thr).sum() for j in ix[g]])
        rows.append((wi(TWC, g), wi(M_F, g), wi(M_B, g)))
        print(f"  {g:<10s} {len(ix[g]):3d} {rows[-1][0]:+7.4f} {rows[-1][1]:+7.4f} {rows[-1][2]:+7.4f}"
              f" {wi(PC, g):+9.4f} {np.median(align[ix[g]]):+7.3f} {info:9.1f}")
    h, f_, b = (np.array(c) for c in zip(*rows))
    print(f"\n  Spearman（9 個產業）  ①vs② {spearmanr(h, f_)[0]:+.2f}   ②vs③ {spearmanr(f_, b)[0]:+.2f}"
          f"   ①vs③ {spearmanr(h, b)[0]:+.2f}   |   Pearson ②vs③ {np.corrcoef(f_, b)[0, 1]:+.3f}")
    print(f"  逐檔 B↔指紋 對齊：全 50 檔中位數 {np.median(align):+.3f}")

    Yr, Xr, _, okr = returns_panel(tw, us, residual=False)
    Fr = fingerprint(Yr, Xr, in_window(D, r1["w"], "訓練窗") & okr)
    Mr = similarity(Fr)
    alr = np.array([np.corrcoef(PB[:, j], unit_cols(Fr)[:, j])[0, 1] for j in range(n)])
    rc = [wi(Mr, g) for g in groups]
    print(f"\n  反例：不扣大盤的原始領先相關當對照 —— 各產業凝聚全擠在 {min(rc):+.2f} ~ {max(rc):+.2f}"
          f"（被全市場共通型態主導）；B↔原始 對齊中位數 {np.median(alr):+.3f}，"
          f"8150 {alr[tw.index('8150')]:+.3f}、金融 {np.median(alr[ix['金融保險業']]):+.3f}")
    craw = {g: float(np.median(np.abs(Fr[:, ix[g]]).mean(0))) for g in groups}
    order = sorted(craw, key=craw.get)
    print("  原始領先相關的強度 mean|corr|（產業中位數，由低到高）：" +
          "  ".join(f"{g} {craw[g]:.4f}" for g in order))

    print("\n" + "=" * 78)
    print("G2. 半導體格：8150 南茂")
    print("=" * 78)
    j8 = tw.index("8150")
    semi = ix["半導體業"]
    rest = [j for j in semi if j != j8]
    pr = lambda Mx: float(np.mean([Mx[a, b] for a, b in itertools.combinations(rest, 2)]))
    vs = lambda Mx: float(np.mean([Mx[j8, k] for k in rest]))
    for fold in FOLDS:
        S = per_seed(FOLDS[fold][0], fine)
        per = [np.mean([Sk[j8, k] for k in rest]) for Sk in S]
        Mm = np.mean(S, 0)
        print(f"  B {fold}  8150 對其餘 5 檔 {vs(Mm):+.4f}（{sum(p < 0 for p in per)}/{len(per)} 顆種子為負）"
              f"   其餘 5 檔兩兩 {pr(Mm):+.4f}   整格 {wi(Mm, '半導體業'):+.4f}")
    print(f"  指紋（訓練窗）8150 對其餘 5 檔 {vs(M_F):+.4f}   其餘 5 檔兩兩 {pr(M_F):+.4f}")
    dF = Fn[:, j8] - Fn[:, rest].mean(1)
    dB = PB[:, j8] - PB[:, rest].mean(1)
    print(f"  差異型態（8150 減其餘 5 檔形心）跨 30 個美股節點：corr(指紋, B) = "
          f"{np.corrcoef(dF, dB)[0, 1]:+.3f}")
    c5 = F[:, rest].mean(1)
    print("  指紋差最多的美股節點（其餘 5 檔平均 / 8150）：" + "  ".join(
        f"{us[i]} {c5[i]:+.3f}/{F[i, j8]:+.3f}" for i in np.argsort(-np.abs(F[:, j8] - c5))[:6]))
    print(f"  8150 的指紋在這 6 個節點的 |值| 最大 "
          f"{np.abs(F[np.argsort(-np.abs(F[:, j8] - c5))[:6], j8]).max():.3f}")
    keep = [j for j in range(n) if j != j8]
    for fold in FOLDS:
        sep_in = [bic.separation(B, fine)[0] for B in b_similarity(FOLDS[fold][0], fine)[1]]
        sep_ex = [bic.separation(B[:, keep], [fine[j] for j in keep])[0]
                  for B in b_similarity(FOLDS[fold][0], fine)[1]]
        print(f"  敏感度（事後，不取代主數字）{fold} 分離度 含 8150 {np.mean(sep_in):+.4f}"
              f" -> 排除 {np.mean(sep_ex):+.4f}（{sum(e > i for e, i in zip(sep_ex, sep_in))}/{len(sep_in)} 顆種子上升）")
    print("  B↔指紋 對齊：" + "  ".join(f"{tw[j]} {align[j]:+.3f}" for j in semi)
          + f"   （8150 排 {int((align > align[j8]).sum()) + 1}/50）")
    print(f"  B 欄長 {norm[j8]:.4f}（排 {int((norm > norm[j8]).sum()) + 1}/50，中位數 {np.median(norm):.4f}）")
    dom = {g: np.mean([TWC[j8, k] for k in ix[g] if k != j8]) for g in groups}
    top = sorted(dom.items(), key=lambda kv: -kv[1])[:3]
    print("  ① 國內同步（8150 對各產業平均）最高：" + "  ".join(f"{g} {v:+.4f}" for g, v in top)
          + f"   （其餘 5 檔半導體兩兩 {pr(TWC):+.4f}）")
    al = []
    for _, d in sorted(bic.find_seeds(FOLDS["第一折"][0]).items(), key=lambda kv: int(kv[0])):
        sd = torch.load(f"{d}/checkpoints/best.pt", map_location="cpu", weights_only=False)
        al.append(float(sd.get("model_state_dict", sd)["beta_alpha"].flatten()[j8]))
    Xt = X[mk]
    print(f"  已否證的放大假說：α_8150 = {np.mean(al):+.3f} ± {np.std(al, ddof=1):.3f}（項 ① 近乎滿量接入），"
          f"但 IMOS 在美股盤對 SMH 的殘差相關 {np.corrcoef(Xt[:, us.index('IMOS')], Xt[:, us.index('SMH')])[0, 1]:+.3f}"
          f"——① 與 B 同向，不是抵銷")

    print("\n" + "=" * 78)
    print("G3. 金融格")
    print("=" * 78)
    fin = ix["金融保險業"]
    loo = {tw[j]: np.mean([M_B[j, k] for k in fin if k != j]) for j in fin}
    lo = min(loo, key=loo.get)
    print(f"  留一（對其餘 13 檔）範圍 [{min(loo.values()):+.4f}, {max(loo.values()):+.4f}]"
          f"   全部 > 0：{all(v > 0 for v in loo.values())}   最低 {lo}")
    lab0 = np.array(["非銀" if tw[j] in NONBANK else "銀行" for j in fin])
    def split(Mx, lab):
        s, d = [], []
        for a, b in itertools.combinations(range(len(fin)), 2):
            (s if lab[a] == lab[b] else d).append(Mx[fin[a], fin[b]])
        return np.mean(s) - np.mean(d)
    rng = np.random.default_rng(42)
    for nm, Mx in (("B", M_B), ("指紋", M_F)):
        obs = split(Mx, lab0)
        null = [split(Mx, rng.permutation(lab0)) for _ in range(2000)]
        print(f"  銀行 vs 非銀（事先指定 {sorted(NONBANK)}）  {nm:<4s} 差 {obs:+.4f}"
              f"   排列 p {(1 + sum(x >= obs for x in null)) / 2001:.4f}")
    xlf = us.index("XLF")
    print(f"  B 對 XLF 的載重為正：{int((PB[xlf, fin] > 0).sum())}/{len(fin)}"
          f"   為負的：{[tw[j] for j in fin if PB[xlf, j] <= 0]}")
    print(f"  B↔指紋 對齊中位數：金融 {np.median(align[fin]):+.3f}   全 50 檔 {np.median(align):+.3f}")

    print("\n" + "=" * 78)
    print("G4. 圖①的非對角格（9 個產業的區塊均值，36 格）")
    print("=" * 78)
    Mn = M_B.copy()
    np.fill_diagonal(Mn, np.nan)
    blk = lambda a, b: float(np.nanmean(Mn[np.ix_(ix[a], ix[b])]))
    off = [(blk(a, b), a, b) for a, b in itertools.combinations(groups, 2)]
    dia = [blk(g, g) for g in groups]
    v = np.array([o[0] for o in off])
    print(f"  對角 {min(dia):+.2f} ~ {max(dia):+.2f}   非對角 {v.min():+.2f} ~ {v.max():+.2f}"
          f"   |v| >= 0.20 的非對角格 {int((np.abs(v) >= 0.2).sum())}/36")
    for val, a, b in sorted(off)[:2] + sorted(off)[-2:]:
        print(f"    {a} x {b}  {val:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="跨市場殘差領先指紋 vs B（兩折）")
    ap.add_argument("--detail", action="store_true",
                    help="另印圖①的讀法：三個概念、半導體與金融兩格的成因、非對角格（§55.9(g)）")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    tw, us, fine = universe()
    Y, X, D, ok = returns_panel(tw, us)
    print(f"報酬面板  {len(D)} 個台股交易日 x {len(tw)} 檔台股 / {len(us)} 檔美股"
          f"   {D[0]} .. {D[-1]}")
    res = {}
    for fold, (arm, f2) in FOLDS.items():
        r = res[fold] = analyse(fold, arm, f2, Y, X, D, ok, fine)
        w = r["w"]
        print("\n" + "=" * 78)
        print(f"{fold}   arm = {arm}   n = {r['n_seed']} 顆種子")
        print(f"  窗界（取自 run）  訓練 {w['訓練窗'][0]}..{w['訓練窗'][1]} 之前   驗證 {w['驗證窗'][0]}..{w['驗證窗'][1]}"
              f"   測試 {w['測試窗'][0]}..{w['測試窗'][1]}")
        print("=" * 78)
        s, p = r["B_sep"]
        print(f"  B 的產業分離               {s:+.4f}   p {p:.4f}")
        print(f"\n  {'指紋算在':<6s} {'T':>5s} {'指紋分離':>9s} {'p':>7s} {'Mantel(B,指紋)':>15s}"
              f" {'p':>7s} {'逐檔 >0':>8s} {'中位數':>7s} {'與訓練窗指紋':>11s}")
        for key in ("訓練窗", "驗證窗", "測試窗"):
            k = r[key]
            stab = "—" if key == "訓練窗" else f"{k['stab']:+.4f}"
            print(f"  {key:<6s} {k['T']:5d} {k['sep'][0]:+9.4f} {k['sep'][1]:7.4f}"
                  f" {k['mantel'][0]:+15.4f} {k['mantel'][1]:7.4f}"
                  f" {int((k['align'] > 0).sum()):5d}/50 {np.median(k['align']):+7.3f} {stab:>11s}")
        rd = r["ridge"]
        print(f"\n  穩健性 Ŵ（ridge，alpha {rd['alpha']:.3g}，驗證窗 IC {rd['ic']:+.4f}）"
              f"   分離 {rd['sep'][0]:+.4f}   Mantel(B, Ŵ) {rd['vs_B'][0]:+.4f}"
              f"   Mantel(訓練窗指紋, Ŵ) {rd['vs_F']:+.4f}")

    print("\n" + "=" * 78)
    print("判準（事先登記）")
    print("=" * 78)
    for fold in FOLDS:
        r = res[fold]
        m, p = r["測試窗"]["mantel"]
        ok2 = r["測試窗"]["sep"][1] < 0.05 and r["B_sep"][1] < 0.05
        print(f"  {fold}  主判準 Mantel(B, 測試窗指紋) r {m:+.4f}  p {p:.4f}  ->  {verdict(m, p)}"
              f"   |   次判準 {'通過' if ok2 else '未通過'}")
    if args.detail:
        detail(res, Y, X, D, ok, tw, us, fine)


if __name__ == "__main__":
    main()
