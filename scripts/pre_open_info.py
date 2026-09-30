"""pre_open_info.py — 開盤前的資訊能預測什麼（proposal §61）

兩個問題：

  Q1  模型的輸入最晚到美股收盤（台北 04:00／05:00）。台股 09:00 的開盤集合競價
      已經把這些公開資訊定價，所以在市場有效的虛無假設下，盤中腿（09:00 -> 13:30）
      不該可預測。實際剩多少、從哪裡來？
  Q2  台指期類工具（TX / TE / TF / SOF 的夜盤）帶的是大盤／類股層級的隔夜資訊。
      它對現行的隔日收盤目標、以及計劃中的隔夜跳空目標，上限各是多少？

時序（台北）：t−1 日 13:30 台股收盤（y 的起點）-> 15:00 夜盤開始 -> t 日 04:00／05:00
美股收盤（模型輸入的最後一筆）-> 05:00 夜盤收盤 -> 08:45 台指期日盤開盤 ->
09:00 台股開盤（隔夜腿的終點、盤中腿的起點）-> 13:30 收盤（y 的終點）。

區塊：

  A. 對齊閘門            美股輸入必須等於資料集的 X1、重建的 r_tot 必須等於 y，不過就中止
  B. 盤中腿的可預測來源   MAGNET、美股昨夜資訊（ridge 無截距）、逐檔持久傾斜，逐日 HAC
  C. 逐檔持久傾斜         平均隔夜與平均盤中報酬的跨股相關（tug of war）、跨期持續、
                          與相對跳動單位的關係
  D. 通道別線性探針       §58.5 的更正：資料集的對齊、逐檔截距開／關
  E. 神諭上限（交叉擬合） Q2 的主結果
  F. 錯誤建構的反例       留一均值的偏誤（§61.3 記錄的更正）

**神諭**是當天 09:00 實際實現的大盤／類股隔夜漲跌（或全天漲跌，只作參照）。
它比夜盤 05:00 收盤時能揭露的還多，所以是**上限**，不是可實現的值。

**為什麼要交叉擬合**：同一天的留一均值 `(S − r_j)/(n−1)` 會把股票自己的報酬
以 −1/(n−1) 的權重帶進自己的預測子，在 14–22 檔的組內系統性壓低 IC；
全組均值則以 +1/n 帶入、系統性抬高。區塊 E 依類股分層把 50 檔隨機切兩半，
一半的神諭只用另一半的股票建，IC 只在該半內算，20 次切分 x 2 半取平均。
區塊 F 保留留一建構的數字，作為 §61.3 更正的證據。

只讀 `data/processed/{tw,adr}` 的 OHLC 與 `runs/` 的預測檔，不寫入任何檔案。

用法：
    .venv/bin/python scripts/pre_open_info.py
    .venv/bin/python scripts/pre_open_info.py --blocks B E
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
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from factor_vs_graph import RET, collect  # noqa: E402
from src.dataset.config import load_universe  # noqa: E402
from src.dataset.multiplex_dataset import MultiplexDataset  # noqa: E402

CUT = "2024-12-25"          # 第二折評估窗上界，與 results_table.py 同值
ALPHAS = np.logspace(-2, 9, 23)
SEED = 42
N_SPLIT = 20
GRID = np.round(np.arange(0, 2.01, 0.1), 2)
FOLDS = {"第一折": ("configs/tw50.yaml", False,
                    "runs/**/*tw50_betaF1nA2r1_s*/predictions/{}_predictions.csv"),
         "第二折": ("configs/tw50_fold2.yaml", True,
                    "runs/**/*f2_best_s*/predictions/{}_predictions.csv")}
ELEC = {"半導體業", "電腦及週邊設備業", "其他電子業", "光電業", "通信網路業", "電子零組件業"}
FIN = {"金融保險業"}
BLOCKS = "ABCDEF"


# ---------------------------------------------------------------- 資料
def load_fold(cfg_path: str, fold2: bool) -> dict:
    """資料集的 train / val / test 日期、美股輸入（X1 最後一步 log_return）、y。"""
    cfg = yaml.safe_load(open(ROOT / cfg_path))
    kw = dict(snapshot_dir=cfg["data"]["snapshot_dir"], features_dir=cfg["data"]["features_dir"],
              T=cfg["model"]["lstm"]["T_history"], config_path=cfg_path)
    out = {}
    for sp in ("train", "val", "test"):
        ds = MultiplexDataset(split=sp, **kw)
        d = collect(ds)
        dates = np.array([str(x)[:10] for x in d["dates"]])
        keep = dates <= CUT if (fold2 and sp == "test") else np.ones(len(dates), bool)
        out[f"d_{sp}"] = dates[keep]
        out[f"us_{sp}"] = d["X1"][keep, -1, :, RET].astype(np.float64)
        out[f"y_{sp}"] = d["Y"][keep].astype(np.float64)
    out["codes"] = [str(c) for c in ds.tw_codes]
    return out


def panel_tw(codes: list[str]) -> dict[str, pd.DataFrame]:
    out = {k: {} for k in ("r_tot", "r_gap", "r_int", "close")}
    for c in codes:
        d = (pd.read_csv(ROOT / f"data/processed/tw/{c}.csv", parse_dates=["Date"])
             .sort_values("Date").set_index("Date"))
        out["r_tot"][c] = np.log(d.Close / d.Close.shift(1))
        out["r_gap"][c] = np.log(d.Open / d.Close.shift(1))
        out["r_int"][c] = np.log(d.Close / d.Open)
        out["close"][c] = d.Close
    return {k: pd.DataFrame(v) for k, v in out.items()}


def panel_us(tickers) -> pd.DataFrame:
    return pd.DataFrame({t: (lambda d: np.log(d.Close / d.Close.shift(1)))(
        pd.read_csv(ROOT / f"data/processed/adr/{t}.csv", parse_dates=["Date"])
        .sort_values("Date").set_index("Date")) for t in tickers})


def last_before(us_index: pd.DatetimeIndex, dates) -> pd.DatetimeIndex:
    """資料集的慣例：美股取嚴格早於台股 target_date 的最後一列（multiplex_dataset.py）。"""
    pos = np.searchsorted(us_index.values, pd.to_datetime(dates).values, side="left") - 1
    return us_index[pos]


def magnet(pattern: str, split: str, dates, codes) -> np.ndarray:
    Hs = []
    for f in sorted(glob.glob(str(ROOT / pattern.format(split)), recursive=True)):
        df = pd.read_csv(f, dtype={"ticker": str})
        df["target_date"] = df["target_date"].astype(str).str[:10]
        H = df.pivot(index="target_date", columns="ticker", values="y_hat")
        Hs.append(H.reindex(index=dates, columns=codes).to_numpy(float))
    return np.stack(Hs)


# ---------------------------------------------------------------- 統計
def ic_ser(P, Y) -> np.ndarray:
    s = []
    for t in range(len(Y)):
        a, b = P[t], Y[t]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
            s.append(np.nan)
            continue
        s.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return np.array(s)


def ic_rows(P, Y) -> np.ndarray:
    """ic_ser 的向量化版（區塊 E 用）。"""
    P = np.nan_to_num(P - np.nanmean(P, 1, keepdims=True))
    Y = np.nan_to_num(Y - np.nanmean(Y, 1, keepdims=True))
    den = np.sqrt((P ** 2).sum(1) * (Y ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-18, (P * Y).sum(1) / den, np.nan)


def zr(P):
    P = P - np.nanmean(P, 1, keepdims=True)
    s = np.nanstd(P, 1, keepdims=True)
    return P / np.where(s < 1e-12, 1.0, s)


def hac_p(d) -> tuple[float, float]:
    """Newey-West，lag = floor(4 (T/100)^(2/9))，與 overnight_decomposition.py 同式。"""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    x = d - d.mean()
    var = float(x @ x) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float(x[k:] @ x[:-k]) / n
    t = d.mean() / np.sqrt(max(var, 1e-24) / n)
    return float(d.mean()), float(2 * stats.t.sf(abs(t), n - 1))


def summ(s) -> tuple[float, float]:
    s = s[np.isfinite(s)]
    return float(s.mean()), float(s.mean() / s.std(ddof=1))


def ridge_shared(X: dict, Y: dict, intercept: bool = True):
    """per-target ridge（50 檔共用一個 alpha，以 val 平均橫截面 IC 選），封閉解。

    輸入只用訓練窗的均值與標準差標準化。intercept=True 時每檔帶一個截距（= 訓練窗平均報酬），
    也就是逐檔持久傾斜；False 時只剩美股內容。回傳 (val IC, alpha, test 預測, val 預測)。
    """
    mu, sd = X["train"].mean(0), X["train"].std(0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    Z = {sp: np.nan_to_num((X[sp] - mu) / sd) for sp in X}
    T = np.nan_to_num(Y["train"])
    Tm = T.mean(0)
    s, V = np.linalg.eigh(Z["train"].T @ Z["train"])
    VtXtT = V.T @ (Z["train"].T @ (T - Tm))
    best = None
    for a in ALPHAS:
        C = V @ (VtXtT / (s[:, None] + a))
        b = Tm if intercept else 0.0
        v = np.nanmean(ic_ser(Z["val"] @ C + b, Y["val"]))
        if best is None or v > best[0]:
            best = (v, a, Z["test"] @ C + b, Z["val"] @ C + b)
    return best


def loo_mean(M, mask):
    """留一均值：組內的 j 得到 (S − r_j)/(n−1)，組外得到組均值。**區塊 F 專用，有偏。**"""
    Mm = np.where(mask[None, :], np.nan_to_num(M), 0.0)
    tot, cnt = Mm.sum(1, keepdims=True), mask.sum()
    return np.where(mask[None, :], (tot - Mm) / (cnt - 1), tot / cnt)


def tick_size(p):
    """證交所股票升降單位（依成交價格區間）。"""
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


# ---------------------------------------------------------------- 區塊
def block_a(D, X, T, dates, codes):
    g_us = max(np.nanmax(np.abs(X[sp] - D[f"us_{sp}"])) for sp in dates)
    g_y = max(np.nanmax(np.abs(T["r_tot"][sp] - D[f"y_{sp}"])) for sp in dates)
    print(f"[A] 對齊閘門  |美股輸入 − X1| max {g_us:.2e}   |重建 r_tot − y| max {g_y:.2e}")
    if g_us > 1e-5 or g_y > 1e-5:
        raise SystemExit("對齊閘門未通過：美股取列規則或台股資料來源與資料集不一致")


def block_b(X, T, M):
    I, G = T["r_int"], T["r_gap"]
    print("[B] 盤中腿（09:00 -> 13:30）的可預測來源：逐日橫截面 IC 平均、逐日 HAC p")
    tilt = np.tile(np.nanmean(I["train"], 0), (len(I["test"]), 1))
    p_off = ridge_shared(X, I, False)[2]
    p_on = ridge_shared(X, I, True)[2]
    ens = np.nanmean(M["test"], 0)
    for lab, P in (("逐檔持久傾斜（訓練窗平均盤中報酬）", tilt),
                   ("美股昨夜資訊（ridge，無截距）", p_off),
                   ("美股 ridge + 逐檔截距", p_on),
                   ("MAGNET 集成（以隔日收盤為目標訓練）", ens)):
        m, p = hac_p(ic_ser(P, I["test"]))
        print(f"    {lab:34s} IC {m:+.4f}   HAC p {p:.4f}")
    m, p = hac_p(ic_ser(ens, G["test"]))
    print(f"    {'參照：MAGNET 集成 -> 隔夜腿':34s} IC {m:+.4f}   HAC p {p:.1e}")
    per = [hac_p(ic_ser(M["test"][s], I["test"]))[0] for s in range(len(M["test"]))]
    print(f"    MAGNET 逐種子 -> 盤中腿：平均 {np.mean(per):+.4f}，{sum(v > 0 for v in per)}/{len(per)} 顆為正")


def block_c(T, dates, TW, codes):
    g_tr, i_tr = np.nanmean(T["r_gap"]["train"], 0), np.nanmean(T["r_int"]["train"], 0)
    g_te, i_te = np.nanmean(T["r_gap"]["test"], 0), np.nanmean(T["r_int"]["test"], 0)
    print("[C] 逐檔持久傾斜（50 檔的平均報酬）")
    print(f"    corr(平均隔夜, 平均盤中)  訓練窗 {np.corrcoef(g_tr, i_tr)[0, 1]:+.3f}   "
          f"測試窗 {np.corrcoef(g_te, i_te)[0, 1]:+.3f}")
    print(f"    訓練窗 -> 測試窗的持續性  隔夜 {np.corrcoef(g_tr, g_te)[0, 1]:+.3f}   "
          f"盤中 {np.corrcoef(i_tr, i_te)[0, 1]:+.3f}")
    print(f"    訓練窗的跨股平均  隔夜 {np.nanmean(g_tr) * 1e4:+.2f} bp/日   盤中 {np.nanmean(i_tr) * 1e4:+.2f} bp/日")
    px = TW["close"].loc[dates["train"], codes].to_numpy(float)
    rel = np.nanmedian(tick_size(px) / px, 0)
    for lab, v in (("平均隔夜", g_tr), ("平均盤中", i_tr)):
        r, p = stats.spearmanr(rel, v)
        print(f"    Spearman(相對跳動單位, {lab}) {r:+.3f}（p {p:.2f}）"
              f"   相對跳動單位 {np.nanmin(rel) * 1e4:.1f}–{np.nanmax(rel) * 1e4:.1f} bp（還原價，近似）")


def block_d(X, T):
    print("[D] 通道別線性探針（30 檔美股 t−1 -> 該通道；資料集對齊；alpha 以 val IC 選）")
    print(f"    {'通道':6s} {'只有逐檔截距':>12s} {'美股，無截距':>12s} {'美股 + 截距':>12s}")
    for ch, lab in (("r_gap", "隔夜跳空"), ("r_tot", "總報酬"), ("r_int", "盤中")):
        Y = T[ch]
        tilt = np.tile(np.nanmean(Y["train"], 0), (len(Y["test"]), 1))
        a = np.nanmean(ic_ser(tilt, Y["test"]))
        b = np.nanmean(ic_ser(ridge_shared(X, Y, False)[2], Y["test"]))
        c = np.nanmean(ic_ser(ridge_shared(X, Y, True)[2], Y["test"]))
        print(f"    {lab:6s} {a:+12.4f} {b:+12.4f} {c:+12.4f}")


def block_e(X, T, M, dates, g3, g4, rng):
    print(f"[E] 神諭上限（交叉擬合，{N_SPLIT} 次切分 x 2 半，IC 在 25 檔的半邊內計算）")
    G = T["r_gap"]
    # 背景量：類股間變異占比、美股對大盤隔夜跳空的時間序列解釋力
    Gt = G["test"]
    tot = np.nanvar(Gt, 1)
    betw = np.array([np.sum([(g3 == k).sum() * (np.nanmean(Gt[t, g3 == k]) - np.nanmean(Gt[t])) ** 2
                             for k in range(3)]) / 50 for t in range(len(Gt))])
    ok = tot > 1e-12
    mk = {sp: np.nanmean(G[sp], 1) for sp in dates}
    mu, sd = X["train"].mean(0), X["train"].std(0)
    Z = {sp: (X[sp] - mu) / np.where(sd < 1e-10, 1.0, sd) for sp in dates}
    best = None
    for a in ALPHAS:
        w = np.linalg.solve(Z["train"].T @ Z["train"] + a * np.eye(Z["train"].shape[1]),
                            Z["train"].T @ (mk["train"] - mk["train"].mean()))
        v = np.corrcoef(Z["val"] @ w, mk["val"])[0, 1]
        if best is None or v > best[0]:
            best = (v, np.corrcoef(Z["test"] @ w, mk["test"])[0, 1])
    print(f"    電子／金融／其他三大類之間的變異，占每日隔夜跳空橫截面變異 平均 {np.mean(betw[ok] / tot[ok]):.3f}")
    print(f"    美股 ridge 對「50 檔等權隔夜跳空」的時間序列相關（測試窗）{best[1]:+.3f}")

    base = {"r_gap": {"美股 ridge": ridge_shared(X, G, True)},
            "r_tot": {"美股 ridge": ridge_shared(X, T["r_tot"], True),
                      "MAGNET 集成": (None, None, np.nanmean(M["test"], 0), np.nanmean(M["val"], 0))}}
    combos = [("r_gap", "r_gap", "隔夜跳空目標 <- 實現的隔夜跳空（夜盤上限）"),
              ("r_tot", "r_gap", "隔日收盤目標 <- 實現的隔夜跳空（夜盤上限）"),
              ("r_tot", "r_tot", "隔日收盤目標 <- 實現的全天報酬（開盤前不可得，參照）")]
    acc: dict = {}
    for _ in range(N_SPLIT):
        half = np.zeros(50, bool)
        for k in np.unique(g4):                               # 以最細的四類分層
            idx = np.flatnonzero(g4 == k)
            half[rng.choice(idx, len(idx) // 2, replace=False)] = True
        for H in (half, ~half):
            O = ~H
            for tgt, src, lab in combos:
                Y = T[tgt]
                m = {sp: np.nanmean(T[src][sp][:, O], 1) for sp in dates}
                beta = np.array([np.polyfit(np.nan_to_num(m["train"]), np.nan_to_num(Y["train"][:, j]), 1)[0]
                                 for j in np.flatnonzero(H)])
                orc = {"大盤（beta_j x 均值）": {sp: beta[None, :] * m[sp][:, None] for sp in ("val", "test")}}
                for gname, g in (("三大類（電子／金融／其他）", g3), ("四類（再拆出半導體）", g4)):
                    orc[gname] = {}
                    for sp in ("val", "test"):
                        gm = {k: np.nanmean(T[src][sp][:, O & (g == k)], 1) for k in np.unique(g)}
                        orc[gname][sp] = np.stack([gm[k] for k in g[H]], 1)
                for bname, b in base[tgt].items():
                    bva, bte = b[3][:, H], b[2][:, H]
                    acc.setdefault((lab, bname, "基準"), []).append(summ(ic_rows(bte, Y["test"][:, H])))
                    for oname, Ov in orc.items():
                        acc.setdefault((lab, bname, oname + " 單獨"), []).append(
                            summ(ic_rows(Ov["test"], Y["test"][:, H])))
                        zb, zo = np.nan_to_num(zr(bva)), np.nan_to_num(zr(Ov["val"]))
                        vals = [np.nanmean(ic_rows(zb + w * zo, Y["val"][:, H])) for w in GRID]
                        w = GRID[int(np.nanargmax(vals))]
                        s = summ(ic_rows(np.nan_to_num(zr(bte)) + w * np.nan_to_num(zr(Ov["test"])),
                                         Y["test"][:, H]))
                        acc.setdefault((lab, bname, oname + " 疊加"), []).append((s[0], s[1], w))
    last = None
    for (lab, bname, what), v in acc.items():
        if (lab, bname) != last:
            print(f"    [{lab}] 底 = {bname}")
            last = (lab, bname)
        v = np.array(v)
        tail = (f"（val 選的 w 平均 {v[:, 2].mean():.2f}，w=0 的次數 {int((v[:, 2] == 0).sum())}/{len(v)}）"
                if what.endswith("疊加") else "")
        print(f"        {what:24s} IC {v[:, 0].mean():+.4f}  ICIR {v[:, 1].mean():.4f} {tail}")


def block_f(X, T, M, g3):
    print("[F] 錯誤建構的反例：留一均值（自身報酬以 −1/(n−1) 進入自己的預測子）")
    grid = np.round(np.arange(0, 2.01, 0.1), 2)
    for ch in ("r_gap", "r_tot"):
        Y = T[ch]
        mk = {sp: loo_mean(Y[sp], np.ones(50, bool)) for sp in ("train", "val", "test")}
        beta = np.array([np.polyfit(np.nan_to_num(mk["train"][:, j]), np.nan_to_num(Y["train"][:, j]), 1)[0]
                         for j in range(50)])
        orc = {"大盤（留一）": {sp: beta[None, :] * mk[sp] for sp in ("val", "test")},
               "三大類（留一）": {sp: np.take_along_axis(
                   np.stack([loo_mean(Y[sp], g3 == k) for k in range(3)], 2),
                   g3[None, :, None].repeat(len(Y[sp]), 0), 2)[:, :, 0] for sp in ("val", "test")}}
        r = ridge_shared(X, Y, True)
        bases = [("美股 ridge", r[3], r[2])]
        if ch == "r_tot":
            bases.append(("MAGNET 集成", np.nanmean(M["val"], 0), np.nanmean(M["test"], 0)))
        for bname, bva, bte in bases:
            print(f"    目標 {ch}  底 {bname}  IC {np.nanmean(ic_ser(bte, Y['test'])):+.4f}")
            for oname, O in orc.items():
                so = np.nanmean(ic_ser(O["test"], Y["test"]))
                w = grid[int(np.nanargmax([np.nanmean(ic_ser(zr(bva) + g * zr(O["val"]), Y["val"])) for g in grid]))]
                s = np.nanmean(ic_ser(zr(bte) + w * zr(O["test"]), Y["test"]))
                print(f"        + {oname:12s} 單獨 {so:+.4f}   val 選 w={w:.1f} -> IC {s:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="開盤前的資訊能預測什麼（proposal §61）")
    ap.add_argument("--blocks", nargs="*", default=list(BLOCKS), choices=list(BLOCKS))
    args = ap.parse_args()

    u = load_universe("tw50")
    rng = np.random.default_rng(SEED)
    for fold, (cfg, f2, pat) in FOLDS.items():
        D = load_fold(cfg, f2)
        codes = D["codes"]
        dates = {sp: pd.to_datetime(D[f"d_{sp}"]) for sp in ("train", "val", "test")}
        TW, US = panel_tw(codes), panel_us(list(u.us_nodes))
        rows = {sp: last_before(US.index, dates[sp]) for sp in dates}
        X = {sp: US.loc[rows[sp]].to_numpy() for sp in dates}
        T = {ch: {sp: TW[ch].loc[dates[sp], codes].to_numpy() for sp in dates}
             for ch in ("r_tot", "r_gap", "r_int")}
        M = {sp: magnet(pat, sp, D[f"d_{sp}"], codes) for sp in ("val", "test")}
        ind = np.array([u.industry[c] for c in codes])
        g3 = np.where(np.isin(ind, list(ELEC)), 0, np.where(np.isin(ind, list(FIN)), 1, 2))
        g4 = np.where(ind == "半導體業", 3, g3)
        print(f"\n######## {fold}：測試窗 {D['d_test'][0]} ~ {D['d_test'][-1]}（{len(D['d_test'])} 天），"
              f"MAGNET {len(M['test'])} 顆種子")
        block_a(D, X, T, dates, codes)                       # 閘門永遠先跑
        if "B" in args.blocks:
            block_b(X, T, M)
        if "C" in args.blocks:
            block_c(T, dates, TW, codes)
        if "D" in args.blocks:
            block_d(X, T)
        if "E" in args.blocks:
            block_e(X, T, M, dates, g3, g4, rng)
        if "F" in args.blocks:
            block_f(X, T, M, g3)


if __name__ == "__main__":
    main()
