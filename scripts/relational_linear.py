"""
relational_linear.py — 兩個線性 baseline：關係型線性與兩區塊 ridge（proposal §67）

起點是 §67.3 的事後分析：冷啟動預檢時「直接平均同業的係數」在台股昨日報酬這組特徵上會錯位，
改成依關係共用權重之後，val 上勝過兩區塊 ridge（+0.0109，p 0.028）。2026-10-13 事前登記、
test 只算一次（§67.4–67.5）：主判準未通過，第一折為負。兩個模型依登記的規則仍列入
results_table 的線性 baseline（§67.7）。

  關係型   ŷ_j = Σ_u B[u,j]·z^US_u + w₁·z^TW_j + w₂·（同產業其他股票的 z^TW 平均）
                 + w₃·（其他產業股票的 z^TW 平均）
           B 每檔自由、帶 ridge λ；w 三個共用、不懲罰。解的是聯合最小平方的精確解：
           先把 B 消掉、w 解 3x3 系統、再回代 B（與登記時的交替最小平方逐位元相同）。
           沒有同業的 9 檔，同業平均記為 0。
  兩區塊   美股 + 台股的 ridge，兩個區塊各自一個強度，台股 50x50 係數自由。

共同協定（與線性-z 相同）：特徵 = 美股 30 + 台股 50 的昨日 log return，以訓練窗逐欄標準化；
目標 = 逐日橫截面 z 的 y；無截距；超參以 val 平均 IC 選。網格從 config 的
baselines.relational_linear 讀，產業別取 universe 的 industry 欄。

--confirm 重跑 §67.4 登記的判準與描述項。模型都是確定性的，重跑只是重現登記當時的數字，
不是新的一次 test 檢驗。

只讀資料；--save 才寫入 runs/ 與 runs_f2/。

用法：
    .venv/bin/python scripts/relational_linear.py             # 兩折的選擇與 val / test IC
    .venv/bin/python scripts/relational_linear.py --save      # 另寫預測檔供 results_table.py 使用
    .venv/bin/python scripts/relational_linear.py --confirm   # 重現 §67.5 的判準與描述項
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import tw_lag_channel as T                                    # noqa: E402
from src.dataset.config import load_universe                  # noqa: E402

FOLDS = tuple(T.FOLDS)
# 預測檔的位置與字尾；results_table.py 以字尾找檔（第一折 runs/**，第二折 runs_f2/）
OUT = {"第一折": ROOT / "runs" / "linear", "第二折": ROOT / "runs_f2"}
SLUG = {"relational": "lin_relational", "twoblock": "lin_twoblock"}
NAME = {"relational": "關係型", "twoblock": "兩區塊", "linear_z": "線性-z", "ktw": "KTW+",
        "tw_relational": "關係型台股通道", "channel": "通道 ④", "main": "MAGNET 主 arm", "p10b": "P10b"}
WINDOWS = ("全部", "500", "250")                              # --confirm 的訓練窗（描述項）


# ---------------------------------------------------------------- 設定與關係
def grids(cfg_path: str) -> dict:
    g = ((yaml.safe_load(open(ROOT / cfg_path)) or {}).get("baselines") or {}).get("relational_linear")
    if not g or not {"relational_lambda", "twoblock_alpha"} <= set(g):
        raise ValueError(f"{cfg_path} 缺少 baselines.relational_linear（請在 configs 設定）")
    return {k: [float(x) for x in g[k]] for k in ("relational_lambda", "twoblock_alpha")}


def relations(codes) -> tuple[list, list]:
    """每檔的同產業夥伴（不含自己）與其他產業股票的索引。"""
    ind = load_universe("tw50").industry
    lab = [ind[c] for c in codes]
    peers = [[k for k in range(len(lab)) if k != j and lab[k] == lab[j]] for j in range(len(lab))]
    others = [[k for k in range(len(lab)) if lab[k] != lab[j]] for j in range(len(lab))]
    return peers, others


# ---------------------------------------------------------------- 模型
def standardize(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [(X - mu) / sd for X in (Xtr, *others)]


def rel_feats(ZT, peers, others):
    """台股昨日報酬（標準化後）-> [自身, 同產業平均, 其他產業平均]，各 [T, n]。"""
    peer = np.column_stack([ZT[:, p].mean(1) if p else np.zeros(len(ZT)) for p in peers])
    oth = np.column_stack([ZT[:, o].mean(1) for o in others])
    return [ZT, peer, oth]


def fit_relational(ZU, R, Tz, lam):
    """min ||Tz − ZU·B − Σ w_r R_r||² + λ||B||² 的精確解。"""
    K = np.linalg.solve(ZU.T @ ZU + lam * np.eye(ZU.shape[1]), ZU.T)

    def S(X):
        return X - ZU @ (K @ X)
    SR, ST = [S(r) for r in R], S(Tz)
    M = np.array([[np.sum(a * b) for b in SR] for a in R])
    w = np.linalg.solve(M, np.array([np.sum(a * ST) for a in R]))
    return K @ (Tz - sum(wr * r for wr, r in zip(w, R))), w


def mean_ic(P, Y) -> float:
    return float(np.nanmean(T.ic_ser(P, Y)))


def relational(Z, Tz, yva, nu, lams, peers, others):
    """Z：{split: 標準化後的 [T, nu + n]}，含 train 與 val。回傳 (val IC, 選擇, {split: 預測})。"""
    R = {sp: rel_feats(z[:, nu:], peers, others) for sp, z in Z.items()}
    best = None
    for lam in lams:
        B, w = fit_relational(Z["train"][:, :nu], R["train"], Tz, lam)
        P = {sp: Z[sp][:, :nu] @ B + sum(wr * r for wr, r in zip(w, R[sp])) for sp in Z if sp != "train"}
        v = mean_ic(P["val"], yva)
        if best is None or v > best[0]:
            best = (v, dict(lam=lam, w=[float(x) for x in w]), P)
    return best


def twoblock(Z, Tz, yva, nu, alphas):
    G, H = Z["train"].T @ Z["train"], Z["train"].T @ Tz
    nt = Z["train"].shape[1] - nu
    best = None
    for aU in alphas:
        for aT in alphas:
            C = np.linalg.solve(G + np.diag(np.r_[np.full(nu, aU), np.full(nt, aT)]), H)
            P = {sp: Z[sp] @ C for sp in Z if sp != "train"}
            v = mean_ic(P["val"], yva)
            if best is None or v > best[0]:
                best = (v, dict(aU=aU, aT=aT), P)
    return best


def linear_z(Z, Tz, yva):
    """線性-z 的協定（共用一個 alpha，網格同 tw_lag_channel.ridge）；--confirm 截短訓練窗時用。"""
    s, V = np.linalg.eigh(Z["train"].T @ Z["train"])
    VtXtT = V.T @ (Z["train"].T @ Tz)
    best = None
    for a in T.ALPHAS:
        C = V @ (VtXtT / (s[:, None] + a))
        P = {sp: Z[sp] @ C for sp in Z if sp != "train"}
        v = mean_ic(P["val"], yva)
        if best is None or v > best[0]:
            best = (v, dict(alpha=float(a)), P)
    return best


def tw_relational(Z, Tz, yva, nu, peers, others):
    """只有台股的關係型通道：3 個權重的 OLS，沒有超參。"""
    R = {sp: rel_feats(z[:, nu:], peers, others) for sp, z in Z.items()}
    w = np.linalg.lstsq(np.column_stack([r.ravel() for r in R["train"]]), Tz.ravel(), rcond=None)[0]
    P = {sp: sum(wr * r for wr, r in zip(w, R[sp])) for sp in Z if sp != "train"}
    return mean_ic(P["val"], yva), dict(w=[float(x) for x in w]), P


# ---------------------------------------------------------------- 輸出
def save(fold, key, D, P, hp, val_ic, s, stamp):
    d = OUT[fold] / f"{stamp}_{SLUG[key]}"
    (d / "predictions").mkdir(parents=True)
    n = len(D["codes"])
    pd.DataFrame([(D["d_test"][t], D["codes"][j], P[t, j], D["y_test"][t, j])
                  for t in range(len(D["y_test"])) for j in range(n)],
                 columns=["target_date", "ticker", "y_hat", "y"]
                 ).to_csv(d / "predictions" / "test_predictions.csv", index=False)
    json.dump({"slug": d.name, "tag": SLUG[key], "status": "FINISHED", "model": key, "fold": fold,
               "hyperparameters": hp, "val_IC": val_ic,
               "test_metrics": {"IC": s["IC"], "ICIR": s["ICIR"], "RankIC": s["RIC"]},
               "proposal": "§67"}, open(d / "meta.json", "w"), indent=2, ensure_ascii=False)
    print(f"    -> {d.relative_to(ROOT)}")


def pooled(per):
    d = np.concatenate(per)
    d = d[np.isfinite(d)]
    return float(d.mean()), float(T.hac_p(d)), int(len(d))


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def report(ser):
    """§67.4 登記的判準與描述項（ser：(折, 模型, 訓練窗) -> 逐日 test IC）。"""
    def diff(a, b, W="全部"):
        per = [ser[(f, a, W)] - ser[(f, b, W)] for f in FOLDS]
        m, p, n = pooled(per)
        return dict(d=[float(np.nanmean(x)) for x in per], p_fold=[float(T.hac_p(x)) for x in per],
                    pooled=m, p=p, n=n)

    def line(lab, r, extra=""):
        print(f"  {lab}：第一折 {r['d'][0]:+.4f}（p {r['p_fold'][0]:.3f}）  第二折 {r['d'][1]:+.4f}"
              f"（p {r['p_fold'][1]:.3f}）  串接 {r['pooled']:+.4f}，HAC p {r['p']:.4f}，n={r['n']}{extra}")

    H = diff("relational", "twoblock")
    ok = H["d"][0] > 0 and H["d"][1] > 0 and H["p"] < 0.05
    print("\n主判準：關係型 − 兩區塊（兩折都 > 0，且串接 HAC p < 0.05）")
    line("差", H, f"  -> {'通過' if ok else '未通過'}")
    S = {"S1 關係型 − 線性-z": diff("relational", "linear_z"), "S2 關係型 − KTW+": diff("relational", "ktw"),
         "S3 關係型台股通道 − 通道 ④": diff("tw_relational", "channel")}
    print("次要（三個一起做 Holm）")
    for (k, r), h in zip(S.items(), holm(np.array([r["p"] for r in S.values()]))):
        ok = r["d"][0] > 0 and r["d"][1] > 0 and h < 0.05
        line(k, r, f"，Holm {h:.4f}  -> {'通過' if ok else '未通過'}")
    print("描述（不計入判準）")
    for W in WINDOWS[1:]:
        for b in ("twoblock", "linear_z"):
            line(f"關係型 − {NAME[b]}，訓練窗 {W}", diff("relational", b, W))
    for b in ("main", "p10b"):
        line(f"關係型 − {NAME[b]}（種子平均）", diff("relational", b))


def main() -> None:
    ap = argparse.ArgumentParser(description="關係型線性與兩區塊 ridge（proposal §67）")
    ap.add_argument("--save", action="store_true", help="寫預測檔到 runs/linear/ 與 runs_f2/")
    ap.add_argument("--confirm", action="store_true", help="重現 §67.4 登記的判準與描述項")
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M")
    ser = {}
    for f, c in T.FOLDS.items():
        D = T.load_fold(c["cfg"], c["f2"])
        g = grids(c["cfg"])
        peers, others = relations(D["codes"])
        nu = D["us_train"].shape[1]
        X = {sp: np.hstack([D[f"us_{sp}"], D[f"tw_{sp}"]]) for sp in ("train", "val", "test")}
        n = len(D["y_train"])
        print(f"[{f}] train {n} / val {len(D['y_val'])} / test {len(D['y_test'])} 天；"
              f"有同業 {sum(map(bool, peers))} 檔")
        for W in (WINDOWS if a.confirm else WINDOWS[:1]):
            sl = slice(n - (n if W == "全部" else int(W)), n)
            Z = dict(zip(("train", "val", "test"), standardize(X["train"][sl], X["val"], X["test"])))
            Tz = T.zrow(D["y_train"][sl])
            fits = {"relational": relational(Z, Tz, D["y_val"], nu, g["relational_lambda"], peers, others),
                    "twoblock": twoblock(Z, Tz, D["y_val"], nu, g["twoblock_alpha"])}
            if a.confirm and W != "全部":
                fits["linear_z"] = linear_z(Z, Tz, D["y_val"])
            for k, (v, hp, P) in fits.items():
                s = T.summ(P["test"], D["y_test"])
                ser[(f, k, W)] = s["ser"]
                print(f"  訓練窗 {W:>4s} {NAME[k]}：val IC {v:+.4f}  test IC {s['IC']:+.4f} / ICIR {s['ICIR']:.4f}"
                      f" / RankIC {s['RIC']:+.4f}  {hp}")
                if a.save and W == "全部":
                    save(f, k, D, P["test"], hp, v, s, stamp)
            if a.confirm and W == "全部":
                _, hp, P = tw_relational(Z, Tz, D["y_val"], nu, peers, others)
                ser[(f, "tw_relational", W)] = T.summ(P["test"], D["y_test"])["ser"]
        if a.confirm:
            ser[(f, "linear_z", "全部")] = T.summ(T.ridge(D, "US30+TW50", "zscore", intercept=False)["test"],
                                                 D["y_test"])["ser"]
            ser[(f, "channel", "全部")] = T.summ(T.channel(D)["test"], D["y_test"])["ser"]
            H, _ = T.preds(c["ktw"], "test", D["d_test"], D["codes"])
            ser[(f, "ktw", "全部")] = T.summ(H[0], D["y_test"])["ser"]
            for k in ("main", "p10b"):
                H, _ = T.preds(c[k], "test", D["d_test"], D["codes"])
                ser[(f, k, "全部")] = np.nanmean(np.stack([T.ic_ser(h, D["y_test"]) for h in H]), 0)
    if a.confirm:
        report(ser)


if __name__ == "__main__":
    main()
