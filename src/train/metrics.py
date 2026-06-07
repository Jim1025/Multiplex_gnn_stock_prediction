"""
metrics.py — M4 評估指標
Corresponds to IMPLEMENTATION_SPEC §6 (evaluation metrics)

核心指標：
  - cross_sectional_ic(y_hat, y, method): 單一時間點 t 的橫截面 IC
        Pearson 相關係數（method="pearson"）或 Spearman 相關係數（method="spearman"）
  - aggregate_ic(...):    跨多個時間點聚合 → IC / ICIR / RankIC / RankICIR
  - regression_metrics(): MSE / MAE / RMSE 攤平計算

警告（SPEC §6）：
  n=7 stocks 太小 → 單日 IC 噪音極大。
  解法：早期停止監控的是「整 epoch 上所有日子的 IC 平均」，而非單日 IC。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# 內部工具：純 numpy 的 Pearson / Spearman
# ---------------------------------------------------------------------------

def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson 相關係數（純 numpy，避免 scipy 依賴）。"""
    if a.size < 2 or b.size < 2:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 相關係數 = 排名後的 Pearson。"""
    if a.size < 2 or b.size < 2:
        return float("nan")
    # argsort 兩次得到 rank
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson_corr(ra, rb)


# ---------------------------------------------------------------------------
# 單一時間點 IC
# ---------------------------------------------------------------------------

def cross_sectional_ic(
    y_hat: Tensor | np.ndarray,
    y:     Tensor | np.ndarray,
    method: str = "pearson",
) -> float:
    """
    Cross-sectional Information Coefficient at single time t.

    Args:
        y_hat : [n] 預測值（n=7 stocks）
        y     : [n] 真實值
        method: "pearson" → IC；"spearman" → RankIC

    Returns:
        float（可能為 NaN，當 std=0 或 n<2）
    """
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    y_hat = np.asarray(y_hat).ravel()
    y     = np.asarray(y).ravel()
    assert y_hat.shape == y.shape, f"shape mismatch: {y_hat.shape} vs {y.shape}"

    # 過濾 NaN（任一邊有 NaN 整對丟）
    valid = np.isfinite(y_hat) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    a, b = y_hat[valid], y[valid]

    if method == "pearson":
        return _pearson_corr(a, b)
    if method == "spearman":
        return _spearman_corr(a, b)
    raise ValueError(f"未知 method: {method!r}（pearson|spearman）")


# ---------------------------------------------------------------------------
# 聚合 IC
# ---------------------------------------------------------------------------

def aggregate_ic(
    daily_y_hats: list[Tensor] | list[np.ndarray],
    daily_ys:     list[Tensor] | list[np.ndarray],
) -> dict:
    """
    跨多個時間點聚合 IC 與 ICIR。

    Args:
        daily_y_hats : list of [n] 預測，長度 = #(time-points)
        daily_ys     : list of [n] 真實

    Returns:
        {
            "IC":      float,   每日 Pearson IC 的平均
            "ICIR":    float,   mean(IC) / std(IC)；std=0 時為 NaN
            "RankIC":  float,   Spearman 版本平均
            "RankICIR":float,
            "daily_IC":     list[float],
            "daily_RankIC": list[float],
        }
    """
    daily_ic   = []
    daily_rank = []
    for yh, y in zip(daily_y_hats, daily_ys):
        daily_ic.append(cross_sectional_ic(yh, y, method="pearson"))
        daily_rank.append(cross_sectional_ic(yh, y, method="spearman"))

    arr_ic   = np.asarray([x for x in daily_ic   if not math.isnan(x)], dtype=np.float64)
    arr_rank = np.asarray([x for x in daily_rank if not math.isnan(x)], dtype=np.float64)

    def _icir(arr: np.ndarray) -> float:
        if arr.size < 2:
            return float("nan")
        std = arr.std(ddof=1)
        if std < 1e-12:
            return float("nan")
        return float(arr.mean() / std)

    return {
        "IC":         float(arr_ic.mean())   if arr_ic.size   > 0 else float("nan"),
        "ICIR":       _icir(arr_ic),
        "RankIC":     float(arr_rank.mean()) if arr_rank.size > 0 else float("nan"),
        "RankICIR":   _icir(arr_rank),
        "daily_IC":     daily_ic,
        "daily_RankIC": daily_rank,
    }


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(
    y_hat: Tensor | np.ndarray,
    y:     Tensor | np.ndarray,
) -> dict:
    """
    跨整個 split 攤平計算 MSE / MAE / RMSE。

    Args:
        y_hat, y: [N_total] 或 [N_days, n]，會 .ravel()

    Returns:
        {"MSE": float, "MAE": float, "RMSE": float}
    """
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    y_hat = np.asarray(y_hat).ravel().astype(np.float64)
    y     = np.asarray(y).ravel().astype(np.float64)

    valid = np.isfinite(y_hat) & np.isfinite(y)
    if valid.sum() == 0:
        return {"MSE": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}

    diff = y_hat[valid] - y[valid]
    mse = float((diff * diff).mean())
    mae = float(np.abs(diff).mean())
    rmse = float(math.sqrt(mse))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}
