"""
us_relational_precheck.py — 美股層關係型處理的線性預檢（proposal §82）

登記見 §82（reg_us_relational_precheck.md，計算前寫下並記 SHA-256）。**只用 train 與 val**：載入後立刻丟掉 test。

  基準 B：兩區塊 ridge [美股 30 | 台股 50]（§67 的兩區塊）
  候選 C：三區塊 ridge [美股 30 | 台股 50 | 美股同產業平均 30]，同產業平均不含自己，沒有同業的美股該欄為 0
  降維版（描述）：兩區塊 ridge [美股 10 類產業平均 | 台股 50]

線性-z 協定：逐日橫截面 z 目標、無截距、標準化只用擬合段的統計量。每個區塊一個收縮強度，網格為
baselines.relational_linear.twoblock_alpha；候選的第三個區塊另可選「不加」。
強度在 train 內部選（最後 20% 天數當內部驗證），再用整段 train 重擬合、在 val 上評估。

用法：
    .venv/bin/python scripts/us_relational_precheck.py --validate   # 用 val 選強度的兩區塊重現 §67.3 的 val IC
    .venv/bin/python scripts/us_relational_precheck.py              # 登記的預檢
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import relational_linear as RL  # noqa: E402  standardize、grids 與 §67 同一份
import tw_lag_channel as T  # noqa: E402  load_fold、ic_ser、zrow、hac_p
from src.dataset.config import load_universe  # noqa: E402

INNER = 0.8              # train 的前 80% 擬合、後 20% 當內部驗證（登記 §3）


def us_relations():
    """美股的同產業夥伴（不含自己）與 10 類產業的成員，順序同 universe.us_nodes（= 資料集的 L1 順序）。"""
    u = load_universe("tw50")
    lab = [u.industry[c] for c in u.us_nodes]
    peers = [[k for k in range(len(lab)) if k != i and lab[k] == lab[i]] for i in range(len(lab))]
    cats = sorted(set(lab))
    members = [[i for i in range(len(lab)) if lab[i] == c] for c in cats]
    return peers, members


def peer_mean(U, peers):
    return np.column_stack([U[:, p].mean(1) if p else np.zeros(len(U)) for p in peers])


def blocks_for(D, sp, kind, peers, members):
    U, W = D[f"us_{sp}"], D[f"tw_{sp}"]
    if kind == "B":
        return [U, W]
    if kind == "C":
        return [U, W, peer_mean(U, peers)]
    return [np.column_stack([U[:, m].mean(1) for m in members]), W]          # 降維版


def fit_predict(Btr, Ytr, Bev, alphas):
    """各區塊一個強度（None = 不加該區塊）；回傳 eval 段的預測。"""
    keep = [i for i, a in enumerate(alphas) if a is not None]
    Xtr = np.hstack([Btr[i] for i in keep])
    Xev = np.hstack([Bev[i] for i in keep])
    Ztr, Zev = RL.standardize(Xtr, Xev)
    pen = np.concatenate([np.full(Btr[i].shape[1], alphas[i]) for i in keep])
    C = np.linalg.solve(Ztr.T @ Ztr + np.diag(pen), Ztr.T @ T.zrow(Ytr))
    return Zev @ C


def select(Btr, Ytr, Bev, Yev, grid, n_blocks, optional_last):
    """在 (Btr -> Bev) 上選強度，回傳 (最佳強度組合, eval 平均 IC)。"""
    choices = [grid] * n_blocks
    if optional_last:
        choices[-1] = list(grid) + [None]
    best = None
    for al in itertools.product(*choices):
        v = float(np.nanmean(T.ic_ser(fit_predict(Btr, Ytr, Bev, al), Yev)))
        if best is None or v > best[1]:
            best = (al, v)
    return best


def evaluate(D, kind, grid, peers, members, inner: bool):
    """inner=True：train 內部選強度、整段 train 重擬合；False：直接在 val 上選（偏高，只當參考）。"""
    Btr = blocks_for(D, "train", kind, peers, members)
    Bva = blocks_for(D, "val", kind, peers, members)
    nb, opt = len(Btr), kind == "C"
    if inner:
        k = int(len(D["y_train"]) * INNER)
        al, _ = select([b[:k] for b in Btr], D["y_train"][:k], [b[k:] for b in Btr], D["y_train"][k:], grid, nb, opt)
    else:
        al, _ = select(Btr, D["y_train"], Bva, D["y_val"], grid, nb, opt)
    P = fit_predict(Btr, D["y_train"], Bva, al)
    return al, T.ic_ser(P, D["y_val"]), T.ic_ser(P, D["y_val"], rank=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="美股層關係型處理的線性預檢（proposal §82）")
    ap.add_argument("--validate", action="store_true", help="用 val 選強度的兩區塊重現 §67.3 的 val IC")
    a = ap.parse_args()
    peers, members = us_relations()
    print(f"美股 {len(peers)} 檔：{len(members)} 類產業，沒有同業的 {sum(not p for p in peers)} 檔")
    res = {}
    for fold, f in T.FOLDS.items():
        D = T.load_fold(f["cfg"], f["f2"])
        for k in [k for k in D if k.endswith("_test")]:
            del D[k]                                   # 只用 train 與 val
        grid = RL.grids(f["cfg"])["twoblock_alpha"]
        print(f"\n######## {fold}：train {len(D['y_train'])} 天、val {len(D['y_val'])} 天")
        if a.validate:
            al, ic, _ = evaluate(D, "B", grid, peers, members, inner=False)
            print(f"  兩區塊（val 選強度）：val IC {np.nanmean(ic):+.4f}，(aU, aT) = {al}")
            continue
        out = {}
        for kind, lab in (("B", "基準 兩區塊"), ("C", "候選 + 美股同產業平均"), ("R", "降維 美股 10 類平均")):
            al, ic, ric = evaluate(D, kind, grid, peers, members, inner=True)
            out[kind] = (ic, ric)
            print(f"  {lab:18s} val IC {np.nanmean(ic):+.4f}、RankIC {np.nanmean(ric):+.4f}；選到的強度 {al}")
        for kind, lab in (("C", "候選 − 基準"), ("R", "降維 − 基準")):
            d, dr = out[kind][0] - out["B"][0], out[kind][1] - out["B"][1]
            print(f"  {lab}：IC {np.nanmean(d):+.4f}（逐日 HAC p {T.hac_p(d):.4f}）、RankIC {np.nanmean(dr):+.4f}"
                  f"（p {T.hac_p(dr):.4f}）")
        _, icv, _ = evaluate(D, "C", grid, peers, members, inner=False)
        _, icb, _ = evaluate(D, "B", grid, peers, members, inner=False)
        print(f"  參考（在 val 上選強度，偏高）：候選 − 基準 IC {np.nanmean(icv - icb):+.4f}")
        res[fold] = out
    if a.validate:
        return
    print("\n== 兩折串接（登記的主判準：兩折都 > 0 且串接 HAC p < 0.05）")
    for kind, lab in (("C", "候選 − 基準"), ("R", "降維 − 基準（描述）")):
        for i, m in ((0, "IC"), (1, "RankIC")):
            per = [np.nanmean(res[fd][kind][i] - res[fd]["B"][i]) for fd in res]
            d = np.concatenate([res[fd][kind][i] - res[fd]["B"][i] for fd in res])
            print(f"  {lab:16s} {m:6s} 兩折 {per[0]:+.4f} / {per[1]:+.4f}；串接 {np.nanmean(d):+.4f}（HAC p {T.hac_p(d):.4f}）")
    per = [np.nanmean(res[fd]["C"][0] - res[fd]["B"][0]) for fd in res]
    d = np.concatenate([res[fd]["C"][0] - res[fd]["B"][0] for fd in res])
    ok = all(x > 0 for x in per) and T.hac_p(d) < 0.05
    pos = all(x > 0 for x in per)
    print(f"\n== 判定：{'有空間' if ok else '方向為正但不顯著' if pos else '沒有空間'}（主判準：IC）")


if __name__ == "__main__":
    main()
