"""
metrics.py — M4 評估指標
Corresponds to IMPLEMENTATION_SPEC §6 (evaluation metrics)

核心指標：
  - cross_sectional_ic(y_hat, y, method): 單一時間點 t 的橫截面 IC
        Pearson 相關係數（method="pearson"）或 Spearman 相關係數（method="spearman"）
  - aggregate_ic(...):    跨多個時間點聚合 → IC / ICIR / RankIC / RankICIR
  - regression_metrics(): MSE / MAE / RMSE / R2 / R2_zero 攤平計算
  - long_short_metrics(): 每日等權多空組合的 Sharpe / PnL（pre-cost 診斷）
  - rank_bucket_returns(): 各預測名次的平均實現報酬（訊號集中在哪一端）

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


def _avg_rank(a: np.ndarray) -> np.ndarray:
    """平手值取平均名次的 0-based 名次（純 numpy，等同 scipy rankdata "average"）。

    2026-09-01 取代 `np.argsort(np.argsort(a))`。舊寫法對平手值給的是
    **任意但相異**的名次，兩個後果：

      1. 常數輸入會被拆成一個任意排列（實測 [0, 26, 27, 28, ...]），
         於是 Spearman 回傳「ticker 順序 vs 目標」的相關而**不是 NaN**——
         `RankIC` 因此偵測不到預測塌縮，甚至捏造出數字（proposal §44.7）。
      2. 一般平手值也拿到相異名次，嚴格說已經不是 Spearman。

    無平手時本函式回傳 0, 1, ..., n-1，**與舊寫法逐位元相同**，
    所以既有健康 run 的數值不變。常數輸入則全部得到 (n-1)/2，
    變異為 0，由 `_pearson_corr` 的 denom 保護回傳 NaN——正確行為。
    """
    n = a.size
    order = np.argsort(a, kind="mergesort")
    srt = a[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 相關係數 = 平均名次後的 Pearson。"""
    if a.size < 2 or b.size < 2:
        return float("nan")
    return _pearson_corr(_avg_rank(a), _avg_rank(b))


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

# 有效天數門檻：可算出 IC 的日數低於總日數的這個比例時，聚合值回 NaN。
#
# 為什麼需要：`aggregate_ic` 原本把 NaN 日濾掉之後直接平均，沒有下限。
# 實測後果——`tw50_rnorm_s7` 的 val 有 247 天，其中 **246 天預測塌縮成常數**
# （IC 無定義），`best_val_IC = 0.171085` 是**剩下那一天**的值，
# 而它高於本專案任何健康 arm 的 val IC（約 0.158）。也就是說：
# 一個完全塌縮的模型，在模型選擇的指標上看起來是史上最好的一個。
# detach 版更誇張，val IC 0.2542 / 0.2809（proposal §44.7、§44.8.4）。
#
# 回 NaN 之後，train.py 的 `improved = (not isnan(monitor_val)) and ...`
# 會直接讓該 epoch 不被選為 best checkpoint——三個修正裡的第三個因此免費取得。
# 健康 run 的有效天數是 100%，數值完全不受影響。
MIN_VALID_FRAC = 0.5


def aggregate_ic(
    daily_y_hats: list[Tensor] | list[np.ndarray],
    daily_ys:     list[Tensor] | list[np.ndarray],
    min_valid_frac: float = MIN_VALID_FRAC,
) -> dict:
    """
    跨多個時間點聚合 IC 與 ICIR。

    Args:
        daily_y_hats : list of [n] 預測，長度 = #(time-points)
        daily_ys     : list of [n] 真實

    Returns:
        {
            "IC":      float,   每日 Pearson IC 的平均；有效天數 < min_valid_frac
                                時為 NaN（見 MIN_VALID_FRAC 的說明）
            "ICIR":    float,   mean(IC) / std(IC)；std=0 時為 NaN
            "n_days":  int,     總天數
            "n_valid_IC" / "n_valid_RankIC": int，可算出該指標的天數
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

    n_days = len(daily_ic)
    floor  = min_valid_frac * n_days

    def _mean(arr: np.ndarray) -> float:
        # 有效天數不足時回 NaN，而不是拿少數幾天的平均當成整段的成績
        if arr.size == 0 or arr.size < floor:
            return float("nan")
        return float(arr.mean())

    return {
        "IC":         _mean(arr_ic),
        "ICIR":       _icir(arr_ic) if arr_ic.size   >= floor else float("nan"),
        "RankIC":     _mean(arr_rank),
        "RankICIR":   _icir(arr_rank) if arr_rank.size >= floor else float("nan"),
        "n_days":       n_days,
        "n_valid_IC":   int(arr_ic.size),
        "n_valid_RankIC": int(arr_rank.size),
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
    跨整個 split 攤平計算 MSE / MAE / RMSE / R2 / R2_zero。

    R2      : 1 - SS_res / SS_tot，SS_tot 以「樣本均值」為基準（標準定義）。
              可為負值——代表模型比「永遠猜均值」還差。
    R2_zero : 1 - SS_res / Σy²，以「零預測」為基準。日頻報酬的樣本均值本身
              噪音極大，故資產定價文獻常改用零基準（Gu, Kelly & Xiu 2020）。

    兩者皆為 level R²：衡量報酬「數值」的可解釋變異，
    與橫截面「排序」能力（IC）是不同的量，不可互相推導。
    本研究預期兩者皆 ≈ 0，正是「日頻可預測的訊號只在次序、不在數值」的證據。

    Args:
        y_hat, y: [N_total] 或 [N_days, n]，會 .ravel()

    另回傳 dispersion = std(y_hat) / std(y)：預測離散度相對實際報酬的比值。
    此值遠小於 1 代表 prediction collapse——模型退化成近乎常數預測。
    IC 是尺度不變量，看不出這件事；MSE 反而「獎勵」塌縮，
    故必須獨立監測。

    Returns:
        {"MSE", "MAE", "RMSE", "R2", "R2_zero", "dispersion"}
    """
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    y_hat = np.asarray(y_hat).ravel().astype(np.float64)
    y     = np.asarray(y).ravel().astype(np.float64)

    nan = float("nan")
    valid = np.isfinite(y_hat) & np.isfinite(y)
    if valid.sum() == 0:
        return {"MSE": nan, "MAE": nan, "RMSE": nan,
                "R2": nan, "R2_zero": nan, "dispersion": nan}

    yv   = y[valid]
    yhv  = y_hat[valid]
    diff = yhv - yv
    std_y = float(yv.std())

    ss_res  = float((diff * diff).sum())
    ss_tot  = float(((yv - yv.mean()) ** 2).sum())
    ss_zero = float((yv * yv).sum())

    mse  = float((diff * diff).mean())
    mae  = float(np.abs(diff).mean())
    rmse = float(math.sqrt(mse))
    return {
        "MSE":     mse,
        "MAE":     mae,
        "RMSE":    rmse,
        "R2":      1.0 - ss_res / ss_tot  if ss_tot  > 1e-30 else nan,
        "R2_zero": 1.0 - ss_res / ss_zero if ss_zero > 1e-30 else nan,
        "dispersion": float(yhv.std()) / std_y if std_y > 1e-30 else nan,
    }


# ---------------------------------------------------------------------------
# Portfolio metrics（pre-cost 診斷用）
# ---------------------------------------------------------------------------

def long_short_metrics(
    daily_y_hats: list[Tensor] | list[np.ndarray],
    daily_ys:     list[Tensor] | list[np.ndarray],
    n_side:           int,
    periods_per_year: int,
) -> dict:
    """
    每日等權、金額中性的多空組合：做多預測最高的 n_side 檔、做空最低的 n_side 檔。

    存在理由：IC 是統計量，Sharpe 是同一組預測的經濟讀法。跨市場文獻
    （Gao et al. 2022、Liu et al. 2026）皆報 Sharpe，此處補齊可比性。

    重要限制（報告時必須明講）：
      - 無交易成本、無市場衝擊、無流動性限制、無放空限制
      - 屬 pre-cost 診斷，不是可實作的 alpha
      - y 為 log return，組合報酬以成分 log return 平均近似
        （日頻量級下誤差 O(σ²/2)，可忽略，但非嚴格等式）

    Args:
        daily_y_hats : list of [n]，每個元素為一天的橫截面預測
        daily_ys     : list of [n]，對應的實現報酬
        n_side       : 多空各取幾檔（需 2 * n_side <= k）
        periods_per_year : 年化因子（台股約 246-252 交易日）

    Returns:
        {"Sharpe", "mean_daily_pnl", "std_daily_pnl", "hit_rate",
         "cum_log_return", "ann_return", "n_days", "daily_pnl"}
    """
    if n_side < 1:
        raise ValueError(f"n_side 需 >= 1，當前為 {n_side}")

    nan = float("nan")
    daily_pnl: list[float] = []

    for yh, yy in zip(daily_y_hats, daily_ys):
        if isinstance(yh, torch.Tensor):
            yh = yh.detach().cpu().numpy()
        if isinstance(yy, torch.Tensor):
            yy = yy.detach().cpu().numpy()
        yh = np.asarray(yh).ravel().astype(np.float64)
        yy = np.asarray(yy).ravel().astype(np.float64)

        valid = np.isfinite(yh) & np.isfinite(yy)
        if valid.sum() < 2 * n_side:
            continue
        a, b = yh[valid], yy[valid]

        order = np.argsort(-a, kind="stable")        # 由高到低
        long_leg  = b[order[:n_side]].mean()
        short_leg = b[order[-n_side:]].mean()
        daily_pnl.append(float(long_leg - short_leg))

    arr = np.asarray(daily_pnl, dtype=np.float64)
    if arr.size == 0:
        return {
            "Sharpe": nan, "mean_daily_pnl": nan, "std_daily_pnl": nan,
            "hit_rate": nan, "cum_log_return": nan, "ann_return": nan,
            "n_days": 0, "daily_pnl": [],
        }

    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if arr.size > 1 else nan
    sharpe = (
        mean / std * math.sqrt(periods_per_year)
        if (not math.isnan(std) and std > 1e-12) else nan
    )
    return {
        "Sharpe":         sharpe,
        "mean_daily_pnl": mean,
        "std_daily_pnl":  std,
        "hit_rate":       float((arr > 0).mean()),
        "cum_log_return": float(arr.sum()),
        "ann_return":     mean * periods_per_year,
        "n_days":         int(arr.size),
        "daily_pnl":      daily_pnl,
    }


def rank_bucket_returns(
    daily_y_hats: list[Tensor] | list[np.ndarray],
    daily_ys:     list[Tensor] | list[np.ndarray],
) -> dict:
    """
    各「預測名次」對應的平均實現報酬。名次 0 = 當日預測最高。

    k 很小時（本研究 k=7）逐名次比 quantile 分組更直接：一天只有 7 檔，
    分位桶等同逐名次，且逐名次能看出訊號是否集中在頭尾而非單調遞減。
    universe 擴充後再改為 quantile 分組。

    Returns:
        {"by_rank": list[float]（長度 = 最大 cross-section 寬度）,
         "n_days_per_rank": list[int],
         "n_days": int}
    """
    sums:   list[float] = []
    counts: list[int]   = []
    n_days = 0

    for yh, yy in zip(daily_y_hats, daily_ys):
        if isinstance(yh, torch.Tensor):
            yh = yh.detach().cpu().numpy()
        if isinstance(yy, torch.Tensor):
            yy = yy.detach().cpu().numpy()
        yh = np.asarray(yh).ravel().astype(np.float64)
        yy = np.asarray(yy).ravel().astype(np.float64)

        valid = np.isfinite(yh) & np.isfinite(yy)
        if valid.sum() < 2:
            continue
        a, b = yh[valid], yy[valid]
        order = np.argsort(-a, kind="stable")
        n_days += 1

        while len(sums) < order.size:
            sums.append(0.0)
            counts.append(0)
        for r, idx in enumerate(order):
            sums[r]   += float(b[idx])
            counts[r] += 1

    by_rank = [
        (s / c) if c > 0 else float("nan")
        for s, c in zip(sums, counts)
    ]
    return {"by_rank": by_rank, "n_days_per_rank": counts, "n_days": n_days}
