"""
visualize_run.py — MAGNET 訓練實驗視覺化
Corresponds to IMPLEMENTATION_SPEC §6 (evaluation) 的展示層

從 mlruns/ 與 runs/ 讀取訓練結果，產出一組 PNG 圖：

  單一 run 分析（--run-id 接受 slug / hash / 前綴 / "latest"）：
    訓練曲線：
      01_loss_curves.png             train/val 各分量損失曲線（每 epoch）
      02_ic_curves.png               val_IC / val_RankIC + best epoch 標記
    預測診斷（test set）：
      03_test_daily_ic.png           每日 IC 時序 + 分佈直方圖
      04_test_scatter.png            y_hat vs y 散點 + 分佈疊圖
      05_per_ticker_ic.png           每檔 TW 個股 IC bar chart
      06_per_ticker_timeseries.png   每檔個股 y vs ŷ 時序對照（4×2 grid）
      07_direction_accuracy.png      方向預測命中率 vs 50% baseline
      08_rank_bucket_returns.png     按 ŷ rank 分桶看實際 y 分佈（單調性測試）
      09_long_short_equity.png       假想 long-short portfolio 累積報酬曲線

  跨 run 對照（--compare 或 --all）：
    runs/comparison/99_compare_runs.png  所有 run 的 test IC/RankIC/ICIR/MSE bar chart

用法：
    python scripts/visualize_run.py                       # 視覺化最新 run（INDEX.csv 最後一行）
    python scripts/visualize_run.py --run-id 20260607_1634_full   # 用 slug
    python scripts/visualize_run.py --run-id 94d49ee2     # 用 hash 前綴亦可
    python scripts/visualize_run.py --run-id 20260607     # slug 前綴
    python scripts/visualize_run.py --compare             # 跨 run 對照
    python scripts/visualize_run.py --all                 # 為所有 run 生圖 + 對照

輸出佈局（圖片直接寫進該 run 的資料夾）：
    runs/<slug>/figures/01_loss_curves.png …
    runs/comparison/99_compare_runs.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# MLflow 3.x file-store opt-out（與 train.py 同步）
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import matplotlib

matplotlib.use("Agg")   # 不開窗，直接寫檔
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------------------------
# 全域樣式
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi":     120,
    "savefig.dpi":    150,
    "savefig.bbox":   "tight",
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":    ["sans-serif"],
    # 中文支援（macOS）
    "font.sans-serif": ["PingFang TC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


# ---------------------------------------------------------------------------
# MLflow 工具
# ---------------------------------------------------------------------------

def _client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def list_runs(client: MlflowClient, experiment_name: str = "MAGNET_M4_baseline") -> list:
    """列出某 experiment 下所有 run，按開始時間排序（新→舊）。"""
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return []
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=100,
    )
    return runs


def get_metric_history(client: MlflowClient, run_id: str, key: str) -> tuple[list[int], list[float]]:
    """回傳 (steps, values)；不存在則回傳 ([], [])。"""
    try:
        history = client.get_metric_history(run_id, key)
    except Exception:
        return [], []
    history = sorted(history, key=lambda m: m.step)
    return [m.step for m in history], [m.value for m in history]


# ---------------------------------------------------------------------------
# 指標計算（本地簡化版，避免依賴 src.train.metrics 的 torch 路徑）
# ---------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson(ra, rb)


def daily_ic_series(pred_df: pd.DataFrame) -> pd.DataFrame:
    """從 predictions df 算每日 IC / RankIC，回傳 [date, IC, RankIC]。"""
    rows = []
    for date, g in pred_df.groupby("target_date", sort=True):
        yh = g["y_hat"].to_numpy(dtype=np.float64)
        y  = g["y"].to_numpy(dtype=np.float64)
        rows.append({
            "target_date": date,
            "IC":     _pearson(yh, y),
            "RankIC": _spearman(yh, y),
        })
    return pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)


def per_ticker_ic(pred_df: pd.DataFrame) -> pd.DataFrame:
    """每檔 ticker 跨整個 test 期間的 Pearson 相關。"""
    rows = []
    for ticker, g in pred_df.groupby("ticker"):
        yh = g["y_hat"].to_numpy(dtype=np.float64)
        y  = g["y"].to_numpy(dtype=np.float64)
        rows.append({
            "ticker":    ticker,
            "IC":        _pearson(yh, y),
            "RankIC":    _spearman(yh, y),
            "n_samples": len(g),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# slug → 顯示用短標籤
# ---------------------------------------------------------------------------

def short_tag_from_slug(slug_or_tag: str) -> str:
    """從完整 slug 取出簡短 phase 識別碼，用於圖表 title/legend 顯示。

    範例：
      '20260607_1634_full'                          → 'full'
      '20260608_2354_opt_p2_variance_penalty'       → 'opt_p2'
      '20260608_1200_opt_p1_loss_reweight'          → 'opt_p1'
      '20260609_0900_opt_p3_lr_sched_icir'          → 'opt_p3'
      '20260607_1500_sanity'                        → 'sanity'

    規則：
      1. 去掉 YYYYMMDD_HHMM 前綴
      2. 若剩下部分是 ['opt', 'pN', ...] 形式 → 保留前兩段 (opt_pN)
      3. 否則 → 只保留第一段

    註：runs/<slug>/ 目錄結構維持完整 slug，僅圖內顯示用此短標籤。
    """
    if not slug_or_tag:
        return "run"
    parts = slug_or_tag.split("_")
    # 去掉日期時間前綴 YYYYMMDD_HHMM
    if len(parts) >= 3 and len(parts[0]) == 8 and parts[0].isdigit():
        parts = parts[2:]
    if not parts:
        return slug_or_tag
    # 形如 'opt_pN_…' → 保留 opt_pN
    if (len(parts) >= 2 and parts[0] == "opt"
            and len(parts[1]) >= 2 and parts[1][0] == "p"
            and parts[1][1:].isdigit()):
        return f"{parts[0]}_{parts[1]}"
    # 其他 tag → 只保留第一段（如 'full', 'sanity'…）
    return parts[0]


# ---------------------------------------------------------------------------
# 繪圖：單一 run
# ---------------------------------------------------------------------------

def plot_loss_curves(client: MlflowClient, run_id: str, out_path: Path, run_tag: str) -> None:
    """train/val 各分量 loss 曲線（每 epoch）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── train losses
    ax = axes[0]
    for key, label, color in [
        ("train/loss_total", "total",  "#1f77b4"),
        ("train/loss_mse",   "mse",    "#ff7f0e"),
        ("train/loss_rank",  "rank",   "#2ca02c"),
        ("train/loss_align", "align",  "#d62728"),
    ]:
        steps, vals = get_metric_history(client, run_id, key)
        if steps:
            ax.plot(steps, vals, label=label, color=color, linewidth=1.8)
    ax.set_title(f"Train Loss — run={run_tag}")
    ax.set_xlabel("Epoch  (1 epoch = full train pass)")
    ax.set_ylabel("Loss  (per-sample mean, unitless)")
    ax.legend(loc="best", fontsize=9)
    ax.set_yscale("log")

    # ── val losses（與左圖對稱：同樣 4 個 loss 分量）
    ax = axes[1]
    for key, label, color in [
        ("val/loss_total", "total", "#1f77b4"),
        ("val/loss_mse",   "mse",   "#ff7f0e"),
        ("val/loss_rank",  "rank",  "#2ca02c"),
        ("val/loss_align", "align", "#d62728"),
    ]:
        steps, vals = get_metric_history(client, run_id, key)
        if steps:
            ax.plot(steps, vals, label=label, color=color, linewidth=1.8)
    ax.set_title(f"Val Loss — run={run_tag}")
    ax.set_xlabel("Epoch  (1 epoch = full train pass)")
    ax.set_ylabel("Loss  (per-sample mean, unitless)")
    ax.legend(loc="best", fontsize=9)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_ic_curves(client: MlflowClient, run_id: str, out_path: Path, run_tag: str) -> None:
    """val_IC / val_RankIC 每 epoch 曲線 + best epoch 標記。"""
    fig, ax = plt.subplots(figsize=(10, 4.8))

    steps_ic,   vals_ic   = get_metric_history(client, run_id, "val/IC")
    steps_rank, vals_rank = get_metric_history(client, run_id, "val/RankIC")

    if steps_ic:
        ax.plot(steps_ic, vals_ic, label="val/IC",
                color="#1f77b4", linewidth=2.0, marker="o", markersize=4)
    if steps_rank:
        ax.plot(steps_rank, vals_rank, label="val/RankIC",
                color="#2ca02c", linewidth=2.0, marker="s", markersize=4)

    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)

    # best epoch 標記（用 nanargmax 避開 NaN/inf 干擾）
    if steps_ic and vals_ic:
        arr = np.asarray(vals_ic, dtype=np.float64)
        if np.isfinite(arr).any():
            best_idx = int(np.nanargmax(np.where(np.isfinite(arr), arr, -np.inf)))
            best_ep  = steps_ic[best_idx]
            best_val = float(arr[best_idx])
        else:
            best_idx = None
    else:
        best_idx = None
    if best_idx is not None:
        ax.axvline(best_ep, color="red", linewidth=1.2, linestyle=":", alpha=0.7)
        ax.scatter([best_ep], [best_val], color="red", s=90, zorder=5,
                   label=f"best val_IC={best_val:.4f} @ epoch {best_ep}")

    ax.set_title(f"Validation IC / RankIC — run={run_tag}")
    ax.set_xlabel("Epoch  (1 epoch = full train pass)")
    ax.set_ylabel("Correlation  (unitless, [-1, 1])")
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_test_daily_ic(pred_df: pd.DataFrame, out_path: Path, run_tag: str) -> None:
    """test 集每日 IC 時序 + 直方圖（兩 panel）。"""
    daily = daily_ic_series(pred_df)
    ic_vals = daily["IC"].dropna().to_numpy()
    rank_vals = daily["RankIC"].dropna().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── 時序
    ax = axes[0]
    ax.plot(pd.to_datetime(daily["target_date"]), daily["IC"],
            label="daily IC", color="#1f77b4", linewidth=1.2, alpha=0.85)
    ax.plot(pd.to_datetime(daily["target_date"]), daily["RankIC"],
            label="daily RankIC", color="#2ca02c", linewidth=1.0, alpha=0.6)
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    if ic_vals.size > 0:
        ax.axhline(ic_vals.mean(), color="red", linewidth=1.2,
                   label=f"mean IC = {ic_vals.mean():.4f}")
    ax.set_title(f"Test Daily IC — run={run_tag}")
    ax.set_xlabel("Date (test period)")
    ax.set_ylabel("Daily cross-section IC  (unitless, [-1, 1])")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best", fontsize=9)

    # ── 直方圖
    ax = axes[1]
    if ic_vals.size > 0:
        ax.hist(ic_vals, bins=30, color="#1f77b4", alpha=0.55, label="IC", edgecolor="black", linewidth=0.5)
        ax.axvline(ic_vals.mean(), color="red", linewidth=1.5,
                   label=f"mean = {ic_vals.mean():.4f}\nstd = {ic_vals.std(ddof=1):.4f}")
    ax.axvline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title(f"Distribution of Daily IC (n={ic_vals.size} days)")
    ax.set_xlabel("Daily IC  (unitless, [-1, 1])")
    ax.set_ylabel("Frequency  (count of days)")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_test_scatter(pred_df: pd.DataFrame, out_path: Path, run_tag: str) -> None:
    """y_hat vs y 散點（含 45° 線、Pearson 相關）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    yh = pred_df["y_hat"].to_numpy(dtype=np.float64)
    y  = pred_df["y"].to_numpy(dtype=np.float64)
    r  = _pearson(yh, y)

    # ── pooled
    ax = axes[0]
    ax.scatter(y, yh, s=8, alpha=0.35, color="#1f77b4", edgecolors="none")
    lim = max(abs(y).max(), abs(yh).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="red", linewidth=1.2, linestyle="--",
            label="y = ŷ (45°)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Pooled — Pearson r={r:.4f}, n={len(y)}")
    ax.set_xlabel("True log_return  y  (unitless)")
    ax.set_ylabel("Predicted log_return  ŷ  (unitless)")
    ax.legend(loc="best", fontsize=9)

    # ── 預測分佈 vs 真實分佈
    ax = axes[1]
    bins = np.linspace(-lim, lim, 60)
    ax.hist(y,  bins=bins, alpha=0.55, color="#1f77b4", label="y (true)",      edgecolor="black", linewidth=0.3)
    ax.hist(yh, bins=bins, alpha=0.55, color="#ff7f0e", label="ŷ (predicted)", edgecolor="black", linewidth=0.3)
    ax.set_title("Distribution: y vs ŷ")
    ax.set_xlabel("log_return  (unitless)")
    ax.set_ylabel("Frequency  (count of samples)")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"Test — run={run_tag}", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_per_ticker_ic(pred_df: pd.DataFrame, out_path: Path, run_tag: str) -> None:
    """每檔 TW ticker 的 IC / RankIC bar chart。"""
    tdf = per_ticker_ic(pred_df).sort_values("IC", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(tdf))
    w = 0.38
    ax.bar(x - w/2, tdf["IC"],     width=w, color="#1f77b4", label="IC (Pearson)")
    ax.bar(x + w/2, tdf["RankIC"], width=w, color="#2ca02c", label="RankIC (Spearman)")
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tdf["ticker"], rotation=0)
    ax.set_xlabel("TW ticker code")
    ax.set_ylabel("Correlation  (unitless, [-1, 1])")
    ax.set_title(f"Per-Ticker IC on Test Set — run={run_tag}")
    ax.legend(loc="best", fontsize=10)

    # 在每根棒上標數值
    for xi, ic, rk in zip(x, tdf["IC"], tdf["RankIC"]):
        ax.text(xi - w/2, ic, f"{ic:+.3f}", ha="center",
                va="bottom" if ic >= 0 else "top", fontsize=8)
        ax.text(xi + w/2, rk, f"{rk:+.3f}", ha="center",
                va="bottom" if rk >= 0 else "top", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 繪圖：預測深度分析（06-09）
# ---------------------------------------------------------------------------

def plot_per_ticker_timeseries(pred_df: pd.DataFrame, out_path: Path, run_tag: str) -> None:
    """每檔 TW ticker 的 y vs ŷ 時序對照（4×2 grid）。

    揭露：模型在哪些時段跟得上、哪些段崩盤。
    """
    tickers = sorted(pred_df["ticker"].unique())
    n = len(tickers)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.4 * nrows), sharex=True)
    axes = np.atleast_2d(axes).flatten()

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        g = pred_df[pred_df["ticker"] == ticker].sort_values("target_date")
        dates = pd.to_datetime(g["target_date"])
        y     = g["y"].to_numpy()
        yh    = g["y_hat"].to_numpy()
        ic    = _pearson(yh, y)

        ax.plot(dates, y,  label="y (true)",      color="#1f77b4",
                linewidth=0.9, alpha=0.7)
        ax.plot(dates, yh, label="ŷ (predicted)", color="#ff7f0e",
                linewidth=1.1, alpha=0.95)
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(f"{ticker}  IC={ic:+.3f}", fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    # 隱藏多餘的子圖
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # 共用 x/y 軸標籤（每張子圖個別補，避免 suptitle 擋到）
    for ax in axes[:n]:
        ax.set_xlabel("Date (test period)", fontsize=8)
        ax.set_ylabel("log_return", fontsize=8)

    fig.suptitle(f"Per-Ticker Predictions vs Truth (Test Set) — run={run_tag}",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_direction_accuracy(pred_df: pd.DataFrame, out_path: Path, run_tag: str) -> None:
    """方向預測命中率 bar chart（per-ticker + overall），含 50% baseline。

    揭露：排除「全猜 0 但 MSE 看似小」的塌縮模式；
    方向預測對交易直接相關。
    """
    rows = []
    for ticker, g in pred_df.groupby("ticker"):
        y  = g["y"].to_numpy()
        yh = g["y_hat"].to_numpy()
        # 排除 y 接近 0 的樣本（無方向可言）
        mask = np.abs(y) > 1e-6
        if mask.sum() == 0:
            hit = float("nan")
        else:
            hit = float((np.sign(yh[mask]) == np.sign(y[mask])).mean())
        rows.append({"ticker": ticker, "hit_rate": hit, "n": int(mask.sum())})
    df = pd.DataFrame(rows).sort_values("hit_rate", ascending=False)

    # overall（pooled）
    y  = pred_df["y"].to_numpy()
    yh = pred_df["y_hat"].to_numpy()
    mask = np.abs(y) > 1e-6
    overall = float((np.sign(yh[mask]) == np.sign(y[mask])).mean()) if mask.sum() else float("nan")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(df))
    colors = ["#2ca02c" if v >= 0.5 else "#d62728" for v in df["hit_rate"]]
    ax.bar(x, df["hit_rate"], color=colors, alpha=0.85)
    ax.axhline(0.5, color="gray", linewidth=1.2, linestyle="--",
               label="50% random baseline")
    ax.axhline(overall, color="black", linewidth=1.0, linestyle=":",
               label=f"overall = {overall:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["ticker"], rotation=0)
    ax.set_ylim(0, max(0.7, df["hit_rate"].max() * 1.15 if not df["hit_rate"].isna().all() else 0.7))
    ax.set_xlabel("TW ticker code")
    ax.set_ylabel("Direction hit rate  (proportion, [0, 1])")
    ax.set_title(f"Direction Prediction Accuracy — run={run_tag}  "
                 f"(excludes |y|<1e-6 days)")
    ax.legend(loc="best", fontsize=9)

    for xi, v, n in zip(x, df["hit_rate"], df["n"]):
        if np.isfinite(v):
            ax.text(xi, v, f"{v:.3f}\n(n={n})", ha="center",
                    va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_rank_bucket_returns(pred_df: pd.DataFrame, out_path: Path, run_tag: str) -> None:
    """排名分桶分析：每日依 ŷ 排序，看 rank 1..n 的平均實際 y。

    揭露：模型最看好的個股是否真的漲最多（單調性 = 排序力）。
    強模型應呈現 rank 1 > rank 2 > ... > rank n 的單調遞減。
    """
    n_per_day = pred_df.groupby("target_date").size().mode().iloc[0]  # 通常 = 7

    bucket_rows = []
    for date, g in pred_df.groupby("target_date"):
        # rank 1 = 預測 y_hat 最高
        ranked = g.sort_values("y_hat", ascending=False).reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            bucket_rows.append({
                "rank": rank,
                "y":    row["y"],
                "y_hat": row["y_hat"],
            })
    bdf = pd.DataFrame(bucket_rows)
    stats = bdf.groupby("rank").agg(
        mean_y=("y", "mean"),
        sem_y=("y", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        mean_y_hat=("y_hat", "mean"),
        n=("y", "size"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── 左：rank bucket 平均實際報酬（含 95% CI）
    ax = axes[0]
    x = stats["rank"].to_numpy()
    ax.bar(x, stats["mean_y"], yerr=1.96 * stats["sem_y"],
           color="#1f77b4", alpha=0.85, capsize=4,
           label="mean realized y (±95% CI)")
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xlabel("Rank by ŷ within day  (1 = highest predicted, 7 = lowest)")
    ax.set_ylabel("Mean realized log_return  (unitless)")
    ax.set_title("Realized return by predicted rank")
    ax.legend(loc="best", fontsize=9)
    for xi, v in zip(x, stats["mean_y"]):
        ax.text(xi, v, f"{v:+.4f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)

    # ── 右：mean ŷ 與 mean y 兩條線比較
    ax = axes[1]
    ax.plot(x, stats["mean_y_hat"], "o-", color="#ff7f0e", linewidth=2,
            markersize=7, label="mean ŷ (predicted)")
    ax.plot(x, stats["mean_y"],     "s-", color="#1f77b4", linewidth=2,
            markersize=7, label="mean y (realized)")
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xlabel("Rank by ŷ within day  (1 = highest predicted, 7 = lowest)")
    ax.set_ylabel("Mean log_return  (unitless)")
    ax.set_title("Predicted vs Realized by Rank")
    ax.legend(loc="best", fontsize=9)

    # ── 計算 top-bottom spread 與 monotonicity（Spearman rank vs rank）
    # 註：rank=1 代表「ŷ 最高」（finance 慣例），完美模型 rank↑ 時 mean_y↓
    #     → 原始 Spearman(rank, mean_y) 在完美模型 = -1
    # 為了讓讀者直觀「正值 = 好」，額外提供 directional_monotonicity = -mono
    top    = stats["mean_y"].iloc[0]
    bottom = stats["mean_y"].iloc[-1]
    spread = top - bottom
    mono   = _spearman(stats["rank"].to_numpy(), stats["mean_y"].to_numpy())
    dir_mono = -mono   # 衍生指標：+1 = 完美單調預測，-1 = 完全反向

    # 主標題 + 副標題（兩行）
    fig.suptitle(f"Rank Bucket Analysis — run={run_tag}",
                 fontsize=12, y=1.04)
    fig.text(0.5, 0.97,
             f"top-bottom spread = {spread:+.5f}    "
             f"directional_monotonicity = {dir_mono:+.3f}  "
             f"(raw Spearman = {mono:+.3f}; +1 = perfect)",
             ha="center", va="bottom", fontsize=11, color="#444444")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_long_short_equity(pred_df: pd.DataFrame, out_path: Path, run_tag: str) -> None:
    """假想 long-short portfolio 累積報酬曲線。

    每日：long 預測最高的 1 檔、short 預測最低的 1 檔（等權重）。
    對照：「全部 long」被動策略基準。

    揭露：預測信號拿去交易能不能賺；但 n=7 + 246 天樣本太小，
    僅供參考用、不可解讀為實盤可行。
    """
    daily_rows = []
    for date, g in pred_df.groupby("target_date"):
        ranked_pred = g.sort_values("y_hat", ascending=False)
        top    = ranked_pred.iloc[0]
        bottom = ranked_pred.iloc[-1]
        passive_mean = g["y"].mean()
        # long top, short bottom（等權 0.5 / -0.5 → 淨曝險 0）
        ls_ret = top["y"] - bottom["y"]

        # ★ Oracle（上帝視角）：當天買入實際漲最多、做空實際跌最多
        #   這是給定 n=7 cross-section 下「事後最強」的 long-short 上限基準
        ranked_true = g.sort_values("y", ascending=False)
        oracle_ret = ranked_true.iloc[0]["y"] - ranked_true.iloc[-1]["y"]

        daily_rows.append({
            "target_date": date,
            "long_short":  ls_ret,
            "passive":     passive_mean,
            "long_only_top": top["y"],
            "oracle_ls":   oracle_ret,
        })
    ddf = pd.DataFrame(daily_rows).sort_values("target_date").reset_index(drop=True)
    ddf["target_date"] = pd.to_datetime(ddf["target_date"])

    # 累積報酬：對 log_return 直接 cumsum（近似累積對數報酬）
    ddf["cum_ls"]      = ddf["long_short"].cumsum()
    ddf["cum_passive"] = ddf["passive"].cumsum()
    ddf["cum_long_top"] = ddf["long_only_top"].cumsum()
    ddf["cum_oracle"]  = ddf["oracle_ls"].cumsum()

    # 統計
    def _stats(daily: np.ndarray) -> dict:
        if daily.size == 0:
            return {"total": float("nan"), "sharpe": float("nan"), "maxdd": float("nan")}
        mu  = daily.mean()
        sig = daily.std(ddof=1)
        sharpe_ann = (mu / sig * np.sqrt(252)) if sig > 1e-12 else float("nan")
        cum = np.cumsum(daily)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        return {"total": cum[-1], "sharpe": sharpe_ann, "maxdd": dd.min()}

    s_ls      = _stats(ddf["long_short"].to_numpy())
    s_passive = _stats(ddf["passive"].to_numpy())
    s_oracle  = _stats(ddf["oracle_ls"].to_numpy())

    # 模型相對 oracle 的「捕獲率」：累積 ls 報酬 / 累積 oracle 報酬
    capture = (s_ls["total"] / s_oracle["total"]) if abs(s_oracle["total"]) > 1e-9 else float("nan")

    fig, ax = plt.subplots(figsize=(12, 5))
    # Oracle 上界（金色虛線）
    ax.plot(ddf["target_date"], ddf["cum_oracle"],
            label=f"ORACLE long-best short-worst  "
                  f"(Σ={s_oracle['total']:+.3f}, Sharpe={s_oracle['sharpe']:+.2f}) "
                  f"— upper bound",
            color="#FFD700", linewidth=2.2, linestyle="--", alpha=0.9)
    ax.plot(ddf["target_date"], ddf["cum_ls"],
            label=f"model long-top short-bottom  "
                  f"(Σ={s_ls['total']:+.3f}, Sharpe={s_ls['sharpe']:+.2f}, "
                  f"MaxDD={s_ls['maxdd']:+.3f})",
            color="#1f77b4", linewidth=2.0)
    ax.plot(ddf["target_date"], ddf["cum_passive"],
            label=f"passive (mean of all 7)  "
                  f"(Σ={s_passive['total']:+.3f}, Sharpe={s_passive['sharpe']:+.2f})",
            color="gray", linewidth=1.5, linestyle="--", alpha=0.85)
    ax.plot(ddf["target_date"], ddf["cum_long_top"],
            label="model long-only top-1",
            color="#2ca02c", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)

    ax.set_title(
        f"Hypothetical Long-Short Cumulative Return — run={run_tag}\n"
        f"n=7 stocks × {len(ddf)} days; oracle capture rate = "
        f"{(capture * 100):.1f}%; diagnostic only — not tradable",
        fontsize=11,
    )
    ax.set_xlabel("Date (test period)")
    ax.set_ylabel("Cumulative log_return  (sum of daily log_return)")
    ax.legend(loc="best", fontsize=9)
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 繪圖：跨 run 對照
# ---------------------------------------------------------------------------

def plot_compare_runs(
    client: MlflowClient,
    runs: list,
    out_path: Path,
    top_k: int = 8,
) -> None:
    """跨 run 對照：test IC / RankIC / ICIR / MSE。優先用 INDEX.csv 取得 slug。

    擴展性策略（避免 N 個 run 後圖表崩潰）：
      - 過濾：當 run 數 > top_k 時，只保留
        (a) baseline（label='full'）
        (b) 當前 winner（test_IC 最高那個）
        (c) test_IC top-K 的 run
        其餘以 suptitle 註記 `+ N more runs hidden (max IC=X)`。
      - Winner 高亮：所有 4 個子圖中 winner 的 bar 用金色 + 黑邊。

    Args:
        top_k: 保留 top-K by test_IC，預設 8。可透過 CLI `--top-k` 覆寫。
    """
    index = load_index()
    by_id = {row["run_id"]: row for row in index}
    index_order = {row["run_id"]: i for i, row in enumerate(index)}

    summary = []
    for run in runs:
        m = run.data.metrics
        if "test/IC" not in m:
            continue
        row = by_id.get(run.info.run_id, {})
        slug = row.get("slug") or run.data.tags.get("magnet.slug") \
                or run.data.tags.get("mlflow.runName", run.info.run_id[:8])
        summary.append({
            "label":       short_tag_from_slug(slug),
            "full_slug":   slug,
            "order":       index_order.get(run.info.run_id, 1e9),
            "tag":         row.get("tag") or run.data.tags.get("mlflow.runName", ""),
            "test_IC":     m.get("test/IC",     float("nan")),
            "test_RankIC": m.get("test/RankIC", float("nan")),
            "test_ICIR":   m.get("test/ICIR",   float("nan")),
            "test_MSE":    m.get("test/MSE",    float("nan")),
            "best_val_IC": m.get("best_val_IC", float("nan")),
        })

    if not summary:
        print("[compare] 沒有可比較的 run（缺 test/IC）")
        return

    full_df = pd.DataFrame(summary)

    # 找 winner（test_IC 最高，NaN 排除）
    valid = full_df[full_df["test_IC"].notna()]
    winner_label = full_df.loc[valid["test_IC"].idxmax(), "label"] if not valid.empty else None

    # 過濾：top-K + baseline + winner（≤ top_k 時不過濾）
    if len(full_df) <= top_k:
        df = full_df.sort_values("order").reset_index(drop=True)
        n_hidden = 0
        hidden_max_ic = float("nan")
    else:
        keep_labels: set[str] = set()
        if "full" in full_df["label"].values:
            keep_labels.add("full")
        if winner_label is not None:
            keep_labels.add(winner_label)
        keep_labels |= set(valid.nlargest(top_k, "test_IC")["label"])

        df = full_df[full_df["label"].isin(keep_labels)] \
                   .sort_values("order").reset_index(drop=True)
        hidden_df = full_df[~full_df["label"].isin(keep_labels)]
        n_hidden = len(hidden_df)
        hidden_valid = hidden_df[hidden_df["test_IC"].notna()]
        hidden_max_ic = float(hidden_valid["test_IC"].max()) if not hidden_valid.empty else float("nan")

    print("\n[compare] 摘要表（顯示順序 = 訓練時間先後）：")
    print(df[["label", "full_slug", "test_IC", "test_RankIC",
              "test_ICIR", "test_MSE", "best_val_IC"]].to_string(index=False))
    if n_hidden:
        print(f"[compare] 已隱藏 {n_hidden} 個 run（hidden max test_IC = {hidden_max_ic:+.4f}）")
    if winner_label:
        print(f"[compare] winner（test_IC 最高）= {winner_label}（金色高亮）")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    is_winner = [lbl == winner_label for lbl in labels]
    winner_color = "#E2C000"   # gold

    for ax, key, title, ylabel, default_color in [
        (axes[0, 0], "test_IC",     "Test IC (Pearson)",
         "Information correlation",        "#1f77b4"),
        (axes[0, 1], "test_RankIC", "Test RankIC (Spearman)",
         "Rank information correlation",   "#2ca02c"),
        (axes[1, 0], "test_ICIR",   "Test ICIR",
         "IC / std(IC)",                   "#ff7f0e"),
        (axes[1, 1], "test_MSE",    "Test MSE (lower better)",
         "MSE",                            "#d62728"),
    ]:
        vals = df[key].to_numpy()
        bar_colors = [winner_color if w else default_color for w in is_winner]
        ax.bar(x, vals, color=bar_colors, alpha=0.9,)
        ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:+.4f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)

    # 主標題 + 副標題（隱藏數量 + winner）
    fig.suptitle("MAGNET — Cross-Run Comparison (Test Set)",
                 fontsize=12, y=1.02)
    subtitle_parts = []
    if winner_label:
        subtitle_parts.append(f"winner = {winner_label} (gold)")
    if n_hidden:
        subtitle_parts.append(f"+ {n_hidden} runs hidden (max test_IC = {hidden_max_ic:+.4f})")
    if subtitle_parts:
        fig.text(0.5, 0.985, "  •  ".join(subtitle_parts),
                 ha="center", va="bottom", fontsize=10, color="#444444")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# INDEX.csv 載入 + slug 解析
# ---------------------------------------------------------------------------

RUNS_ROOT = Path("runs")


def load_index() -> list[dict]:
    """讀 runs/INDEX.csv，回傳 list[dict]（按 start_time 順序）。"""
    path = RUNS_ROOT / "INDEX.csv"
    if not path.exists():
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def resolve_run(ident: Optional[str], index: list[dict],
                runs: list) -> Optional[tuple[str, str, "mlflow.entities.Run"]]:
    """
    把使用者輸入的識別子解析成 (slug, run_id, mlflow_run)。

    支援格式：
      - None / "latest"        → INDEX.csv 最後一行
      - 完整 slug              → INDEX.csv 完全比對
      - slug 前綴（如 20260607）→ INDEX.csv 起頭比對，多筆取最新
      - tag 名（如 "full"）    → INDEX.csv 該 tag 的最新一行
      - hash / hash 前綴       → 直接比對 run_id
    """
    runs_by_id = {r.info.run_id: r for r in runs}

    # Index 起手沒東西時直接 fallback 到 MLflow
    if not index:
        if ident is None or ident == "latest":
            return _wrap(runs[0]) if runs else None
        for r in runs:
            if r.info.run_id.startswith(ident):
                return _wrap(r)
        return None

    # ── latest ─────────────────────────────────────────────────
    if ident is None or ident == "latest":
        row = index[-1]
        return _resolve_row(row, runs_by_id)

    # ── 完整 slug ──────────────────────────────────────────────
    for row in index:
        if row["slug"] == ident:
            return _resolve_row(row, runs_by_id)

    # ── slug 前綴
    matches = [r for r in index if r["slug"].startswith(ident)]
    if matches:
        return _resolve_row(matches[-1], runs_by_id)   # 取最新

    # ── tag 名（精確）
    tag_matches = [r for r in index if r["tag"] == ident]
    if tag_matches:
        return _resolve_row(tag_matches[-1], runs_by_id)

    # ── hash / hash 前綴
    for row in index:
        if row["run_id"].startswith(ident):
            return _resolve_row(row, runs_by_id)
    for run in runs:
        if run.info.run_id.startswith(ident):
            return _wrap(run)

    return None


def _resolve_row(row: dict, runs_by_id: dict) -> Optional[tuple[str, str, "mlflow.entities.Run"]]:
    rid = row["run_id"]
    run = runs_by_id.get(rid)
    if run is None:
        print(f"[WARN] INDEX.csv 中的 run_id={rid[:8]}... 在 MLflow 內找不到", file=sys.stderr)
        return None
    return row["slug"], rid, run


def _wrap(run) -> tuple[str, str, "mlflow.entities.Run"]:
    """MLflow run 沒有 slug 時的 fallback。"""
    slug = run.data.tags.get("magnet.slug", run.info.run_id[:12])
    return slug, run.info.run_id, run


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _find_predictions_csv(slug: str, run_id: str, kind: str = "test") -> Optional[Path]:
    """優先找 runs/<slug>/predictions/，否則 fallback 到 mlruns artifacts。"""
    candidates = [
        RUNS_ROOT / slug / "predictions" / f"{kind}_predictions.csv",
    ]
    # mlruns artifacts（M4 階段的舊命名格式）
    for exp_dir in Path("mlruns").glob("*/"):
        candidates.append(exp_dir / run_id / "artifacts" / "predictions" / f"{kind}_predictions.csv")
        candidates.append(exp_dir / run_id / "artifacts" / "predictions" / f"{run_id}_{kind}_predictions.csv")

    for c in candidates:
        if c.exists():
            return c
    return None


def visualize_single_run(
    client: MlflowClient,
    slug: str,
    run_id: str,
    run: "mlflow.entities.Run",
) -> None:
    short_tag = run.data.tags.get("mlflow.runName", run_id[:8])
    # runs/<slug>/ 目錄保留完整 slug；圖標題改用短標籤（opt_pN / full / sanity…）
    run_tag = short_tag_from_slug(slug)
    out_dir = RUNS_ROOT / slug / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[viz] slug={slug}  tag={short_tag}  display={run_tag}  → {out_dir}/")

    # ── 1. loss curves
    plot_loss_curves(client, run_id, out_dir / "01_loss_curves.png", run_tag)
    print("  01_loss_curves.png")

    # ── 2. IC curves
    plot_ic_curves(client, run_id, out_dir / "02_ic_curves.png", run_tag)
    print("  02_ic_curves.png")

    # ── 3-5. 需要 predictions CSV
    test_csv = _find_predictions_csv(slug, run_id, "test")
    if test_csv is None:
        print(f"  [WARN] 找不到 {slug} 的 test predictions CSV，跳過 03-05")
        return
    pred_df = pd.read_csv(test_csv)

    plot_test_daily_ic(pred_df, out_dir / "03_test_daily_ic.png", run_tag)
    print("  03_test_daily_ic.png")

    plot_test_scatter(pred_df, out_dir / "04_test_scatter.png", run_tag)
    print("  04_test_scatter.png")

    plot_per_ticker_ic(pred_df, out_dir / "05_per_ticker_ic.png", run_tag)
    print("  05_per_ticker_ic.png")

    plot_per_ticker_timeseries(pred_df, out_dir / "06_per_ticker_timeseries.png", run_tag)
    print("  06_per_ticker_timeseries.png")

    plot_direction_accuracy(pred_df, out_dir / "07_direction_accuracy.png", run_tag)
    print("  07_direction_accuracy.png")

    plot_rank_bucket_returns(pred_df, out_dir / "08_rank_bucket_returns.png", run_tag)
    print("  08_rank_bucket_returns.png")

    plot_long_short_equity(pred_df, out_dir / "09_long_short_equity.png", run_tag)
    print("  09_long_short_equity.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="MAGNET 訓練結果視覺化")
    parser.add_argument("--run-id", type=str, default=None,
                        help="識別子，吃 slug / hash / 前綴 / tag 名（預設 latest）")
    parser.add_argument("--all", action="store_true",
                        help="為所有 run 生圖 + 跨 run 對照")
    parser.add_argument("--compare", action="store_true",
                        help="只跑跨 run 對照（不生單 run 圖）")
    parser.add_argument("--experiment", type=str, default="MAGNET_M4_baseline",
                        help="MLflow experiment 名稱")
    parser.add_argument("--tracking-uri", type=str, default="file:./mlruns",
                        help="MLflow tracking URI（預設 file:./mlruns）")
    parser.add_argument("--top-k", type=int, default=8,
                        help="跨 run 對照圖最多顯示幾個 run（按 test_IC 排序；"
                             "baseline 'full' 與 winner 永遠保留）")
    args = parser.parse_args()

    client = _client(args.tracking_uri)
    runs = list_runs(client, args.experiment)
    if not runs:
        print(f"[FAIL] Experiment '{args.experiment}' 下沒有任何 run", file=sys.stderr)
        return 1

    index = load_index()

    # ── 決定要處理哪些 run（list of (slug, run_id, mlflow_run)）
    targets: list[tuple[str, str, "mlflow.entities.Run"]] = []
    if args.all or args.compare:
        if index:
            # 從 INDEX.csv 拉，順序穩定
            runs_by_id = {r.info.run_id: r for r in runs}
            for row in index:
                tup = _resolve_row(row, runs_by_id)
                if tup:
                    targets.append(tup)
        else:
            targets = [_wrap(r) for r in runs]
    elif args.run_id:
        tup = resolve_run(args.run_id, index, runs)
        if tup is None:
            print(f"[FAIL] 找不到 run：{args.run_id}", file=sys.stderr)
            print(f"   可用 slug 列表：")
            for r in index:
                print(f"     {r['slug']}  ({r['tag']}, run_id={r['run_id'][:8]}...)")
            return 1
        targets = [tup]
    else:
        tup = resolve_run("latest", index, runs)
        if tup is None:
            print("[FAIL] 無可用 run", file=sys.stderr)
            return 1
        targets = [tup]

    # ── 單 run 視覺化
    if not args.compare:
        for slug, rid, run in targets:
            visualize_single_run(client, slug, rid, run)

    # ── 跨 run 對照
    if args.all or args.compare or len(targets) > 1:
        cmp_dir = RUNS_ROOT / "comparison"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        plot_compare_runs(client, runs, cmp_dir / "99_compare_runs.png",
                          top_k=args.top_k)
        print(f"\n[viz] cross-run comparison → {cmp_dir}/99_compare_runs.png")

    print(f"\n完成。圖片寫入 runs/<slug>/figures/ 與 runs/comparison/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
