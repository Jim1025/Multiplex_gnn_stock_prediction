"""
graph_eda.py — Multiplex 圖結構 EDA 視覺化分析
=================================================
對 graph_builder.py 產出的 1643 張圖快照進行完整的探索性資料分析。

產出圖表可直接作為論文素材：
  ① L1/L2 邊密度時間序列（市場相關性如何隨時間變動）
  ② 平均相關係數熱圖（哪些股票對相關性最高）
  ③ 邊數分布直方圖（是否有極端日期邊數異常）
  ④ 不同閾值的敏感性分析（0.2 / 0.3 / 0.4 / 0.5）
  ⑤ 節點度數分布（哪些股票在圖中是「中心節點」）
  ⑥ L1 vs L2 邊密度散點圖（兩市場同步性）
  ⑦ 圖結構統計摘要表（論文用）

執行方式：
    python notebooks/graph_eda.py

    或在 Jupyter Notebook 中逐段複製執行。

輸出：
    docs/figures/graph_eda/  下的 PNG 圖檔
    docs/figures/graph_eda/graph_structure_summary.csv
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# 嘗試設定中文字型（macOS 與 Linux）
try:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC",
                                       "Microsoft JhengHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# 全域圖表風格
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
})


# ════════════════════════════════════════════════════════════
# 設定
# ════════════════════════════════════════════════════════════

METADATA_PATH = "data/graphs/snapshots/graph_metadata.csv"
SNAPSHOT_DIR  = "data/graphs/snapshots"
FEATURES_ADR  = "data/features/adr"
FEATURES_TW   = "data/features/tw"
FIG_DIR       = "docs/figures/graph_eda"

# 從 config 載入配對
def _load_pair_map():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from src.dataset import config as cfg
        if hasattr(cfg, "get_pair_dict") and callable(cfg.get_pair_dict):
            return cfg.get_pair_dict()
        pm = cfg.PAIR_MAP
        first_val = next(iter(pm.values()))
        if isinstance(first_val, dict):
            return {k: v["tw"] for k, v in pm.items()}
        return dict(pm)
    except ImportError:
        return {
            "TSM": "2330", "UMC": "2303", "ASX": "3711",
            "CHT": "2412", "IMOS": "8150", "AUOTY": "2409", "HNHPF": "2317",
        }


PAIR_MAP = _load_pair_map()
ADR_TICKERS = list(PAIR_MAP.keys())
TW_CODES    = list(PAIR_MAP.values())
N_NODES     = len(ADR_TICKERS)
MAX_DIRECTED_EDGES = N_NODES * (N_NODES - 1)     # 42
MAX_UNDIRECTED     = N_NODES * (N_NODES - 1) // 2  # 21

# 公司名稱對照（用於圖表標籤）
COMPANY_NAMES = {
    "TSM": "台積電", "UMC": "聯電", "ASX": "日月光",
    "CHT": "中華電信", "IMOS": "南茂", "AUOTY": "友達", "HNHPF": "鴻海",
    "2330": "台積電", "2303": "聯電", "3711": "日月光",
    "2412": "中華電信", "8150": "南茂", "2409": "友達", "2317": "鴻海",
}


# ════════════════════════════════════════════════════════════
# 載入資料
# ════════════════════════════════════════════════════════════

def load_metadata() -> pd.DataFrame:
    """載入 graph_metadata.csv 並篩選成功建構的快照。"""
    df = pd.read_csv(METADATA_PATH)
    df["target_date"] = pd.to_datetime(df["target_date"])
    # 排除暖機期跳過的快照
    df = df[df["n_l1_nodes"] > 0].copy()
    df = df.set_index("target_date").sort_index()
    # 計算無向邊密度
    df["l1_undirected"] = df["n_l1_edges"] / 2
    df["l2_undirected"] = df["n_l2_edges"] / 2
    df["l1_density_pct"] = df["l1_undirected"] / MAX_UNDIRECTED * 100
    df["l2_density_pct"] = df["l2_undirected"] / MAX_UNDIRECTED * 100
    return df


def load_correlation_matrices(corr_window: int = 60) -> dict:
    """
    用 features CSV 計算全期間的滾動相關矩陣，用於熱圖。
    回傳 {"adr": mean_corr_matrix, "tw": mean_corr_matrix}
    """
    result = {}
    for layer, tickers, feat_dir in [
        ("adr", ADR_TICKERS, FEATURES_ADR),
        ("tw",  TW_CODES,    FEATURES_TW),
    ]:
        series = {}
        for tk in tickers:
            path = Path(feat_dir) / f"{tk}.csv"
            if path.exists():
                df = pd.read_csv(path, index_col="Date", parse_dates=True)
                series[tk] = df["log_return"]
        if len(series) < 2:
            continue
        combined = pd.DataFrame(series).dropna()
        # 全期間平均相關矩陣
        mean_corr = combined.corr(method="pearson")
        result[layer] = mean_corr
    return result


# ════════════════════════════════════════════════════════════
# 圖表 1：L1/L2 邊密度時間序列
# ════════════════════════════════════════════════════════════

def plot_edge_density_timeseries(df: pd.DataFrame, fig_dir: str):
    """L1 與 L2 邊密度隨時間的變化（含 30 日移動平均）。"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, col, label, color in [
        (axes[0], "l1_density_pct", "L1 ADR Graph", "#2196F3"),
        (axes[1], "l2_density_pct", "L2 TW Graph",  "#FF9800"),
    ]:
        ax.plot(df.index, df[col], alpha=0.3, color=color, linewidth=0.5)
        ma = df[col].rolling(30, min_periods=1).mean()
        ax.plot(df.index, ma, color=color, linewidth=2,
                label=f"{label} (30-day MA)")
        ax.set_ylabel("Edge Density (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # 標記重大事件
        events = {
            "2020-03": "COVID-19",
            "2022-02": "Russia-Ukraine",
            "2023-01": "ChatGPT Boom",
        }
        for date_str, event in events.items():
            try:
                d = pd.Timestamp(date_str)
                if df.index.min() <= d <= df.index.max():
                    ax.axvline(d, color="gray", linestyle="--", alpha=0.5)
                    ax.text(d, ax.get_ylim()[1] * 0.95, f" {event}",
                            fontsize=8, color="gray", va="top")
            except Exception:
                pass

    axes[1].set_xlabel("Date")
    fig.suptitle("Edge Density Over Time (L1 ADR vs L2 TW)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "01_edge_density_timeseries.png")
    plt.close()
    print("  01_edge_density_timeseries.png")


# ════════════════════════════════════════════════════════════
# 圖表 2：平均相關係數熱圖
# ════════════════════════════════════════════════════════════

def plot_correlation_heatmaps(corr_mats: dict, fig_dir: str):
    """L1 ADR 與 L2 TW 的全期間平均 Pearson 相關係數矩陣。"""
    # 用 gridspec 為 colorbar 預留獨立空間，避免與熱圖重疊
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])
    axes_list = [ax1, ax2]

    im = None
    for ax, (layer, corr_df), title in zip(
        axes_list,
        corr_mats.items(),
        ["L1 ADR Correlation", "L2 TW Correlation"],
    ):
        tickers = corr_df.columns.tolist()
        labels = tickers   # 純英文 ticker，不含中文公司名
        mat = corr_df.values

        im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # 在每格顯示數值
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                color = "white" if abs(mat[i, j]) > 0.6 else "black"
                ax.text(j, i, f"{mat[i,j]:.2f}",
                        ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, cax=cbar_ax, label="Pearson ρ")
    fig.suptitle("Full-Period Mean Correlation Matrix",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(Path(fig_dir) / "02_correlation_heatmap.png")
    plt.close()
    print("  02_correlation_heatmap.png")


# ════════════════════════════════════════════════════════════
# 圖表 3：邊數分布直方圖
# ════════════════════════════════════════════════════════════

def plot_edge_count_distribution(df: pd.DataFrame, fig_dir: str):
    """L1 / L2 邊數分布（直方圖 + 統計量）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, label, color in [
        (axes[0], "l1_undirected", "L1 ADR", "#2196F3"),
        (axes[1], "l2_undirected", "L2 TW",  "#FF9800"),
    ]:
        data = df[col]
        ax.hist(data, bins=MAX_UNDIRECTED + 1, range=(0, MAX_UNDIRECTED),
                color=color, alpha=0.7, edgecolor="white")
        ax.axvline(data.mean(), color="red", linestyle="--",
                   label=f"Mean={data.mean():.1f}")
        ax.axvline(data.median(), color="green", linestyle="-.",
                   label=f"Median={data.median():.1f}")
        ax.set_xlabel("Number of Undirected Edges")
        ax.set_ylabel("Frequency (trading days)")
        ax.set_title(f"{label} Edge Count Distribution")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 統計量文字
        stats_text = (
            f"Min={int(data.min())}  Max={int(data.max())}\n"
            f"Std={data.std():.1f}  IQR=[{data.quantile(0.25):.0f}, {data.quantile(0.75):.0f}]"
        )
        ax.text(0.97, 0.85, stats_text, transform=ax.transAxes,
                fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Edge Count Distribution (Undirected Pairs)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "03_edge_count_distribution.png")
    plt.close()
    print("  03_edge_count_distribution.png")


# ════════════════════════════════════════════════════════════
# 圖表 4：不同閾值的敏感性分析
# ════════════════════════════════════════════════════════════

def plot_threshold_sensitivity(fig_dir: str):
    """
    用 features CSV 重新計算不同閾值下的平均邊數。
    模擬 graph_builder 在不同 threshold 下的行為。
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    corr_window = 60

    # 載入所有 log_return 序列
    layer_data = {}
    for layer, tickers, feat_dir in [
        ("L1 ADR", ADR_TICKERS, FEATURES_ADR),
        ("L2 TW",  TW_CODES,    FEATURES_TW),
    ]:
        series = {}
        for tk in tickers:
            path = Path(feat_dir) / f"{tk}.csv"
            if path.exists():
                df = pd.read_csv(path, index_col="Date", parse_dates=True)
                series[tk] = df["log_return"]
        if len(series) >= 2:
            layer_data[layer] = pd.DataFrame(series).dropna()

    if not layer_data:
        print("  [WARN] 無法載入 features CSV，跳過閾值敏感性分析")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"L1 ADR": "#2196F3", "L2 TW": "#FF9800"}

    for layer, combined in layer_data.items():
        n = len(combined.columns)
        max_pairs = n * (n - 1) // 2
        avg_edges = []

        for thr in thresholds:
            edge_counts = []
            for start in range(0, len(combined) - corr_window, corr_window // 2):
                window = combined.iloc[start:start + corr_window]
                if len(window) < corr_window // 2:
                    continue
                corr_mat = window.corr().values
                # 計算無向邊數
                count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(corr_mat[i, j]) > thr:
                            count += 1
                edge_counts.append(count)
            avg = np.mean(edge_counts) if edge_counts else 0
            avg_edges.append(avg)

        ax.plot(thresholds, avg_edges, "o-", color=colors[layer],
                linewidth=2, markersize=6, label=layer)
        # 在每個點標數字
        for thr, avg in zip(thresholds, avg_edges):
            ax.annotate(f"{avg:.1f}", (thr, avg),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8)

    ax.axhline(MAX_UNDIRECTED, color="gray", linestyle=":", alpha=0.5,
               label=f"Max possible ({MAX_UNDIRECTED})")
    ax.axvline(0.3, color="red", linestyle="--", alpha=0.5,
               label="Current threshold (0.3)")
    ax.set_xlabel("Correlation Threshold |ρ|")
    ax.set_ylabel("Average Undirected Edge Count")
    ax.set_title("Threshold Sensitivity Analysis",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "04_threshold_sensitivity.png")
    plt.close()
    print("  04_threshold_sensitivity.png")


# ════════════════════════════════════════════════════════════
# 圖表 5：節點度數分布
# ════════════════════════════════════════════════════════════

def plot_node_degree_distribution(fig_dir: str):
    """
    從快照中抽樣計算每個節點的平均度數。
    度數高的節點 = 圖中的「中心節點」，對 GAT 影響最大。
    """
    import torch

    snap_dir = Path(SNAPSHOT_DIR)
    files = sorted(snap_dir.glob("graph_*.pt"))
    if not files:
        print("  [WARN] 找不到快照，跳過節點度數分析")
        return

    # 抽樣 100 張快照
    rng = np.random.default_rng(seed=42)
    sample = rng.choice(files, size=min(100, len(files)), replace=False)

    # 累積每個節點的度數
    l1_degrees = np.zeros((len(sample), N_NODES))
    l2_degrees = np.zeros((len(sample), N_NODES))

    for idx, path in enumerate(sample):
        data = torch.load(path, weights_only=False)

        # L1
        ei = data["adr", "corr", "adr"].edge_index
        for i in range(N_NODES):
            l1_degrees[idx, i] = int((ei[0] == i).sum())

        # L2
        ei = data["tw", "corr", "tw"].edge_index
        for i in range(N_NODES):
            l2_degrees[idx, i] = int((ei[0] == i).sum())

    # 平均度數
    l1_mean = l1_degrees.mean(axis=0)
    l2_mean = l2_degrees.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, degrees, tickers, label, color in [
        (axes[0], l1_mean, ADR_TICKERS, "L1 ADR", "#2196F3"),
        (axes[1], l2_mean, TW_CODES,    "L2 TW",  "#FF9800"),
    ]:
        labels = tickers   # 純英文 ticker，不含中文公司名
        bars = ax.bar(range(N_NODES), degrees, color=color, alpha=0.8,
                      edgecolor="white")
        ax.set_xticks(range(N_NODES))
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Average Degree")
        ax.set_title(f"{label} Node Degree", fontsize=12, fontweight="bold")
        ax.set_ylim(0, N_NODES - 1 + 0.5)

        # 在每根 bar 上標數字
        for bar, deg in zip(bars, degrees):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{deg:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Average Node Degree Distribution (sampled 100 snapshots)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "05_node_degree_distribution.png")
    plt.close()
    print("  05_node_degree_distribution.png")


# ════════════════════════════════════════════════════════════
# 圖表 6：L1 vs L2 邊密度散點圖
# ════════════════════════════════════════════════════════════

def plot_l1_vs_l2_scatter(df: pd.DataFrame, fig_dir: str):
    """L1 與 L2 邊密度的逐日散點圖（兩市場同步性）。"""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(df["l1_density_pct"], df["l2_density_pct"],
               alpha=0.15, s=10, color="#4CAF50")

    # 迴歸線
    from numpy.polynomial.polynomial import polyfit
    b, m = polyfit(df["l1_density_pct"], df["l2_density_pct"], 1)
    x_line = np.linspace(0, 100, 100)
    ax.plot(x_line, b + m * x_line, "r--", linewidth=2, label=f"OLS (slope={m:.2f})")

    # 相關係數
    corr = df["l1_density_pct"].corr(df["l2_density_pct"])
    ax.text(0.05, 0.95, f"Pearson ρ = {corr:.3f}",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("L1 ADR Edge Density (%)")
    ax.set_ylabel("L2 TW Edge Density (%)")
    ax.set_title("L1 vs L2 Edge Density (Daily)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.plot([0, 100], [0, 100], "k:", alpha=0.3, label="y = x")
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "06_l1_vs_l2_scatter.png")
    plt.close()
    print("  06_l1_vs_l2_scatter.png")


# ════════════════════════════════════════════════════════════
# 圖表 7：圖結構統計摘要表
# ════════════════════════════════════════════════════════════

def generate_summary_table(df: pd.DataFrame, fig_dir: str):
    """產出論文用的統計摘要表（CSV + 終端機輸出）。"""
    summary = {
        "Metric": [
            "Number of nodes per layer",
            "Number of snapshots",
            "Correlation window (trading days)",
            "Correlation threshold |ρ|",
            "L1 ADR: mean directed edges",
            "L1 ADR: mean undirected pairs",
            "L1 ADR: mean density (%)",
            "L1 ADR: min/max undirected pairs",
            "L2 TW: mean directed edges",
            "L2 TW: mean undirected pairs",
            "L2 TW: mean density (%)",
            "L2 TW: min/max undirected pairs",
            "A12 cross-layer edges",
            "L1-L2 density correlation (ρ)",
            "Snapshots with imputed nodes",
            "Snapshots with long-gap nodes",
        ],
        "Value": [
            str(N_NODES),
            str(len(df)),
            "60",
            "0.3",
            f"{df['n_l1_edges'].mean():.1f}",
            f"{df['l1_undirected'].mean():.1f}",
            f"{df['l1_density_pct'].mean():.1f}",
            f"{int(df['l1_undirected'].min())} / {int(df['l1_undirected'].max())}",
            f"{df['n_l2_edges'].mean():.1f}",
            f"{df['l2_undirected'].mean():.1f}",
            f"{df['l2_density_pct'].mean():.1f}",
            f"{int(df['l2_undirected'].min())} / {int(df['l2_undirected'].max())}",
            str(N_NODES),
            f"{df['l1_density_pct'].corr(df['l2_density_pct']):.3f}",
            str(int(df["has_imputed"].sum())) if "has_imputed" in df.columns else "N/A",
            str(int(df["has_long_gap"].sum())) if "has_long_gap" in df.columns else "N/A",
        ],
    }

    summary_df = pd.DataFrame(summary)
    csv_path = Path(fig_dir) / "graph_structure_summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("Graph Structure Summary (論文用)")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(f"  {row['Metric']:<45} {row['Value']}")
    print(f"\n  摘要表 → {csv_path}")


# ════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Multiplex Graph Structure EDA")
    print("=" * 60)

    # 建立輸出目錄
    os.makedirs(FIG_DIR, exist_ok=True)

    # 載入 metadata
    if not Path(METADATA_PATH).exists():
        print(f"[FAIL] 找不到 {METADATA_PATH}")
        print(f"   請先執行 graph_builder.py 產出圖快照")
        return 1

    df = load_metadata()
    print(f"\n載入 {len(df)} 張快照的 metadata")
    print(f"  日期範圍：{df.index.min().date()} ~ {df.index.max().date()}")
    print(f"  L1 平均邊數：{df['n_l1_edges'].mean():.1f}（有向）/ "
          f"{df['l1_undirected'].mean():.1f}（無向）")
    print(f"  L2 平均邊數：{df['n_l2_edges'].mean():.1f}（有向）/ "
          f"{df['l2_undirected'].mean():.1f}（無向）")

    # ── 圖表 1：邊密度時間序列 ──────────────────────────────
    print(f"\n產出圖表至 {FIG_DIR}/")
    plot_edge_density_timeseries(df, FIG_DIR)

    # ── 圖表 2：相關係數熱圖 ────────────────────────────────
    corr_mats = load_correlation_matrices()
    if corr_mats:
        plot_correlation_heatmaps(corr_mats, FIG_DIR)
    else:
        print("  [WARN] 無法載入 features CSV，跳過相關係數熱圖")

    # ── 圖表 3：邊數分布 ────────────────────────────────────
    plot_edge_count_distribution(df, FIG_DIR)

    # ── 圖表 4：閾值敏感性 ──────────────────────────────────
    plot_threshold_sensitivity(FIG_DIR)

    # ── 圖表 5：節點度數 ────────────────────────────────────
    plot_node_degree_distribution(FIG_DIR)

    # ── 圖表 6：L1 vs L2 散點 ──────────────────────────────
    plot_l1_vs_l2_scatter(df, FIG_DIR)

    # ── 圖表 7：統計摘要表 ──────────────────────────────────
    generate_summary_table(df, FIG_DIR)

    print("\n" + "=" * 60)
    print("EDA 完成")
    print(f"   共 6 張圖表 + 1 份摘要表 → {FIG_DIR}/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())