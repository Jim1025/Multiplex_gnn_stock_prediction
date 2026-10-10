"""
tw_lag_channel.py — 凍結的線性通道的封閉解：④（proposal §62，P10）與 P12 的寬路徑（§63）

    z_k(t)  = (r_k(t−1) − μ_k) / σ_k          昨日 log_return，μ、σ 取訓練窗
    ĉ_j(t)  = Σ_k C[k,j] · z_k(t) + b_j       C：[p, n]，列 = 昨日的來源 k，欄 = 目標 j

輸入由 cfg 的 inputs 決定：
    l2     ④：50 檔台股（p = 50，C 為方陣）
    l1_l2  P12 的寬路徑：30 檔美股 + 50 檔台股（p = 80），與 KTW+ 的設計矩陣相同；
           台股區塊放在後面，所以台股 j 自身的欄位是 p − n + j

C 是 per-target ridge 的封閉解，50 個目標共用一個 alpha，alpha 以 val 的逐日橫截面
平均 IC 選。演算法與 scripts/tw_lag_channel.py 的 channel()（事前登記的版本）相同，
訓練開始前解一次、之後凍結；它是 numpy 運算，不消耗 torch 的隨機數，所以同一顆種子下
GNN 的初始權重與不開 ④ 時完全相同。

為什麼是封閉解而不是讓 SGD 學（§62.6）：per-target 讀出層（§35.1 E2）與模型聯合訓練時
沒有探針那層在 val 上選的正則化，種子 sd 由 0.0050 翻到 0.0111（過擬合）。
這裡把那個有正則化的估計器原封不動放進模型。

所有選項從 cfg（model.tw_lag_channel）讀，不在這裡設預設值。
"""

from __future__ import annotations

import numpy as np


def collect_lag_xy(ds, ret_idx: int, inputs: str) -> tuple[np.ndarray, np.ndarray]:
    """MultiplexDataset -> (昨日 log_return [T, p], 目標 y [T, n])。

    昨日報酬取 x_seq_L1 / x_seq_L2 的最後一步（嚴格早於 target_date 的最後一列，見資料集的
    look-ahead 守護），與模型 forward 裡通道吃的是同一組張量。inputs 為 l2 或 l1_l2。
    """
    if inputs not in ("l2", "l1_l2"):
        raise ValueError(f"model.tw_lag_channel.inputs 需為 l2 或 l1_l2，得到 {inputs!r}")
    xs, ys = [], []
    for i in range(len(ds)):
        s = ds[i]
        x = s["x_seq_L2"][-1, :, ret_idx].numpy()
        if inputs == "l1_l2":
            x = np.concatenate([s["x_seq_L1"][-1, :, ret_idx].numpy(), x])
        xs.append(x)
        ys.append(s["y"].numpy())
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _row_standardize(Y: np.ndarray) -> np.ndarray:
    """逐日橫截面 z-score（與 scripts/factor_vs_graph.py 的 zscore_transform 同式）。"""
    R = Y - Y.mean(1, keepdims=True)
    sd = R.std(1, keepdims=True)
    return R / np.where(sd < 1e-12, 1.0, sd)


def _mean_daily_ic(P: np.ndarray, Y: np.ndarray) -> float:
    v = []
    for t in range(len(Y)):
        a, b = P[t], Y[t]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        v.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(v)) if v else float("nan")


def fit_closed_form(x_tr: np.ndarray, y_tr: np.ndarray,
                    x_va: np.ndarray, y_va: np.ndarray, cfg: dict) -> dict:
    """解出通道的 μ、σ、C、b 與選中的 alpha。x 為 [T, p]、y 為 [T, n]，C 為 [p, n]。

    cfg 必須有：target（raw | zscore）、intercept（bool）、diagonal（bool）、
    ridge_log10_alpha（[log10 下界, log10 上界, 點數]）。
    """
    for k in ("target", "intercept", "diagonal", "ridge_log10_alpha"):
        if k not in cfg:
            raise ValueError(f"model.tw_lag_channel 缺少 {k!r}（請在 configs/base.yaml 設定）")
    target = str(cfg["target"])
    if target not in ("raw", "zscore"):
        raise ValueError(f"model.tw_lag_channel.target 需為 raw 或 zscore，得到 {target!r}")
    lo, hi, num = cfg["ridge_log10_alpha"]
    alphas = np.logspace(float(lo), float(hi), int(num))

    p, n = x_tr.shape[1], y_tr.shape[1]
    off = p - n                                   # 台股區塊在輸入裡的起點（l2 為 0、l1_l2 為 30）
    mu, sd = x_tr.mean(0), x_tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z_tr, Z_va = (x_tr - mu) / sd, (x_va - mu) / sd
    T = _row_standardize(y_tr) if target == "zscore" else y_tr
    Tm = T.mean(0)

    # 每個 alpha 的係數；diagonal=False 時逐目標拿掉自身那一欄（該係數恆為 0）
    paths = np.zeros((len(alphas), p, n))
    if cfg["diagonal"]:
        s, V = np.linalg.eigh(Z_tr.T @ Z_tr)
        VtXtT = V.T @ (Z_tr.T @ (T - Tm))
        for k, a in enumerate(alphas):
            paths[k] = V @ (VtXtT / (s[:, None] + a))
    else:
        for j in range(n):
            cols = np.delete(np.arange(p), off + j)
            Zj = Z_tr[:, cols]
            s, V = np.linalg.eigh(Zj.T @ Zj)
            VtXtT = V.T @ (Zj.T @ (T[:, [j]] - Tm[j]))
            for k, a in enumerate(alphas):
                paths[k][cols, j] = (V @ (VtXtT / (s[:, None] + a)))[:, 0]

    b = Tm if cfg["intercept"] else np.zeros(n)
    vic = [_mean_daily_ic(Z_va @ paths[k] + b, y_va) for k in range(len(alphas))]
    k = int(np.nanargmax(vic))
    return {"mu": mu, "sd": sd, "C": paths[k], "b": b,
            "alpha": float(alphas[k]), "val_ic": float(vic[k])}
