"""
visualize_graph_snapshot.py — Multiplex GNN 圖結構視覺化（proposal 風格）
==========================================================================
模仿論文 proposal 中的 3D 透視圖風格：
  - 綠色 L1 ADR Graph 在上層（略向右偏移）
  - 橘色 L2 TW Graph 在下層（略向左偏移）
  - A12 跨層虛線連接兩層對應節點
  - 邊的粗細 ∝ |ρ|（相關係數）

執行方式：
    python notebooks/visualize_graph_snapshot.py
    python notebooks/visualize_graph_snapshot.py --date 2024-01-15
    python notebooks/visualize_graph_snapshot.py --no-a12    # 不畫跨層線

輸出：
    docs/figures/graph_eda/multiplex_graph_snapshot.png
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


# ════════════════════════════════════════════════════════════
# 設定
# ════════════════════════════════════════════════════════════

SNAPSHOT_DIR = "data/graphs/snapshots"
FIG_DIR      = "docs/figures/graph_eda"

PAIR_MAP_ORDERED = [
    ("TSM",   "2330", "TSMC"),
    ("UMC",   "2303", "UMC"),
    ("ASX",   "3711", "ASE"),
    ("CHT",   "2412", "CHT"),
    ("IMOS",  "8150", "ChipMOS"),
    ("AUOTY", "2409", "AUO"),
    ("HNHPF", "2317", "Foxconn"),
]

N_NODES = len(PAIR_MAP_ORDERED)

# 顏色（對齊 proposal 風格）
COLOR_L1_NODE  = "#4CAF50"   # 綠色節點
COLOR_L1_BG    = "#C8E6C9"   # 綠色背景
COLOR_L1_EDGE  = "#2E7D32"   # 深綠邊

COLOR_L2_NODE  = "#FF9800"   # 橘色節點
COLOR_L2_BG    = "#FFE0B2"   # 橘色背景
COLOR_L2_EDGE  = "#E65100"   # 深橘邊

COLOR_A12      = "#757575"   # 灰色跨層線
COLOR_TEXT     = "#FFFFFF"   # 節點內白字


# ════════════════════════════════════════════════════════════
# 載入快照
# ════════════════════════════════════════════════════════════

def load_snapshot(date_str=None):
    snap_dir = Path(SNAPSHOT_DIR)
    files = sorted(snap_dir.glob("graph_*.pt"))
    if not files:
        raise FileNotFoundError(f"找不到快照於 {SNAPSHOT_DIR}")

    if date_str:
        target_file = snap_dir / f"graph_{date_str}.pt"
        if not target_file.exists():
            raise FileNotFoundError(f"找不到 {target_file}")
        return torch.load(target_file, weights_only=False), date_str

    # 自動挑選邊數適中的快照
    rng = np.random.default_rng(seed=42)
    mid = len(files) // 4
    candidates = files[mid : mid + len(files) // 2]
    sample = rng.choice(candidates, size=min(50, len(candidates)), replace=False)

    best_file, best_score = None, float("inf")
    for f in sample:
        data = torch.load(f, weights_only=False)
        n_l1 = data["adr", "corr", "adr"].edge_index.shape[1]
        n_l2 = data["tw",  "corr", "tw"].edge_index.shape[1]
        score = abs((n_l1 + n_l2) - 45)
        if score < best_score:
            best_score = score
            best_file = f

    data = torch.load(best_file, weights_only=False)
    return data, best_file.stem.replace("graph_", "")


# ════════════════════════════════════════════════════════════
# 節點佈局（矩形網格 + 3D 透視偏移）
# ════════════════════════════════════════════════════════════

def _compute_grid_positions():
    """
    計算 7 個節點的矩形佈局位置。
    排列方式（模仿 proposal）：
        上排：node0   node1   node2
        下排：node3   node4   node5   node6
    """
    # 上排 3 個，下排 4 個
    top_row    = [(0.5, 1.0), (2.0, 1.0), (3.5, 1.0)]
    bottom_row = [(-0.2, 0.0), (1.15, 0.0), (2.5, 0.0), (3.85, 0.0)]

    positions = top_row + bottom_row  # 共 7 個
    return np.array(positions)


def _apply_perspective(positions, offset_x, offset_y, skew_x=0.0, skew_y=0.0):
    """對節點位置施加平移 + 透視傾斜。"""
    result = positions.copy()
    # 透視傾斜（讓平面有 3D 感）
    for i in range(len(result)):
        x, y = result[i]
        result[i, 0] = x + offset_x + y * skew_x
        result[i, 1] = y + offset_y + x * skew_y
    return result


# ════════════════════════════════════════════════════════════
# 繪圖
# ════════════════════════════════════════════════════════════

def draw_multiplex_graph(data, date_str, out_path, show_a12=True):

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect("equal")
    ax.axis("off")

    # ── 節點位置 ──────────────────────────────────────────
    base_pos = _compute_grid_positions()

    # L1（上層）：向右上偏移，略帶透視
    l1_pos = _apply_perspective(base_pos,
                                offset_x=1.5, offset_y=4.5,
                                skew_x=0.15, skew_y=0.0)
    # L2（下層）：向左下偏移，略帶透視
    l2_pos = _apply_perspective(base_pos,
                                offset_x=0.0, offset_y=0.0,
                                skew_x=0.15, skew_y=0.0)

    # ── 背景平行四邊形 ────────────────────────────────────
    def _draw_layer_bg(ax, positions, color, label, label_pos="left"):
        xs = positions[:, 0]
        ys = positions[:, 1]
        pad = 0.8
        # 四角（含透視傾斜效果）
        corners = np.array([
            [xs.min() - pad,        ys.min() - pad * 0.6],
            [xs.max() + pad,        ys.min() - pad * 0.6],
            [xs.max() + pad + 0.4,  ys.max() + pad * 0.8],
            [xs.min() - pad + 0.4,  ys.max() + pad * 0.8],
        ])
        polygon = plt.Polygon(corners, facecolor=color, edgecolor="none",
                               alpha=0.5, zorder=0)
        ax.add_patch(polygon)

        # 層標籤
        if label_pos == "left":
            lx = corners[3, 0] - 0.2
            ly = (corners[3, 1] + corners[0, 1]) / 2
            ax.text(lx, ly, label, fontsize=14, fontweight="bold",
                    color="#333333", rotation=90, ha="right", va="center")
        else:
            lx = corners[1, 0] + 0.5
            ly = (corners[1, 1] + corners[2, 1]) / 2
            ax.text(lx, ly, label, fontsize=14, fontweight="bold",
                    color="#333333", rotation=90, ha="left", va="center")

    _draw_layer_bg(ax, l1_pos, COLOR_L1_BG, "L1: US ADR", label_pos="left")
    _draw_layer_bg(ax, l2_pos, COLOR_L2_BG, "L2: TWSE",   label_pos="left")

    # ── 繪製邊的通用函式 ──────────────────────────────────
    def _draw_edges(edge_index, edge_attr, positions, color):
        drawn = set()
        n_undirected = 0
        for k in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, k]), int(edge_index[1, k])
            key = (min(src, dst), max(src, dst))
            if key in drawn:
                continue
            drawn.add(key)
            n_undirected += 1

            w = float(edge_attr[k, 0]) if edge_attr is not None else 0.5
            lw = 1.0 + w * 5       # 線寬 1 ~ 6
            alpha = 0.3 + w * 0.5   # 透明度 0.3 ~ 0.8

            ax.plot([positions[src][0], positions[dst][0]],
                    [positions[src][1], positions[dst][1]],
                    color=color, linewidth=lw, alpha=alpha,
                    solid_capstyle="round", zorder=1)
        return n_undirected

    # L1 邊
    l1_ei = data["adr", "corr", "adr"].edge_index
    l1_ea = data["adr", "corr", "adr"].edge_attr
    n_l1 = _draw_edges(l1_ei, l1_ea, l1_pos, COLOR_L1_EDGE)

    # L2 邊
    l2_ei = data["tw", "corr", "tw"].edge_index
    l2_ea = data["tw", "corr", "tw"].edge_attr
    n_l2 = _draw_edges(l2_ei, l2_ea, l2_pos, COLOR_L2_EDGE)

    # ── A12 跨層連接 ─────────────────────────────────────
    if show_a12:
        a12_ei = data["adr", "cross", "tw"].edge_index
        for k in range(a12_ei.shape[1]):
            src = int(a12_ei[0, k])
            dst = int(a12_ei[1, k])
            ax.plot([l1_pos[src][0], l2_pos[dst][0]],
                    [l1_pos[src][1], l2_pos[dst][1]],
                    color=COLOR_A12, linewidth=1.0, linestyle="--",
                    alpha=0.4, zorder=1)

    # ── 繪製節點 ─────────────────────────────────────────
    node_radius = 0.35

    for i in range(N_NODES):
        adr_ticker = PAIR_MAP_ORDERED[i][0]
        tw_code    = PAIR_MAP_ORDERED[i][1]

        # L1 節點（綠色）
        circle_l1 = plt.Circle(l1_pos[i], node_radius,
                                facecolor=COLOR_L1_NODE,
                                edgecolor="#388E3C", linewidth=2, zorder=3)
        ax.add_patch(circle_l1)
        ax.text(l1_pos[i][0], l1_pos[i][1], adr_ticker,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=COLOR_TEXT, zorder=4)

        # L2 節點（橘色）
        circle_l2 = plt.Circle(l2_pos[i], node_radius,
                                facecolor=COLOR_L2_NODE,
                                edgecolor="#E65100", linewidth=2, zorder=3)
        ax.add_patch(circle_l2)
        ax.text(l2_pos[i][0], l2_pos[i][1], tw_code,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=COLOR_TEXT, zorder=4)

    # ── 圖例 ─────────────────────────────────────────────
    max_pairs = N_NODES * (N_NODES - 1) // 2
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_L1_NODE,
               markeredgecolor="#388E3C", markersize=14,
               label=f"L1 ADR Node (n={N_NODES})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_L2_NODE,
               markeredgecolor="#E65100", markersize=14,
               label=f"L2 TW Node (n={N_NODES})"),
        Line2D([0], [0], color=COLOR_L1_EDGE, linewidth=3,
               label=f"L1 Corr. Edge ({n_l1}/{max_pairs})"),
        Line2D([0], [0], color=COLOR_L2_EDGE, linewidth=3,
               label=f"L2 Corr. Edge ({n_l2}/{max_pairs})"),
    ]
    if show_a12:
        legend_elements.append(
            Line2D([0], [0], color=COLOR_A12, linewidth=1.5, linestyle="--",
                   label=f"A12 Cross-layer ({N_NODES} diag.)")
        )

    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=10, frameon=True, fancybox=True,
              edgecolor="gray", framealpha=0.9)

    # ── 標題 ─────────────────────────────────────────────
    target_date  = data.target_date
    window_start = data.window_start
    window_end   = data.window_end

    title_line1 = "Multiplex GNN Graph Snapshot"
    title_line2 = (
        f"Target: {target_date}    "
        f"Window: [{window_start}, {window_end}]    "
        f"|ρ| > 0.3"
    )
    ax.set_title(f"{title_line1}\n{title_line2}",
                 fontsize=14, fontweight="bold", pad=15)

    # ── 邊粗細說明 ────────────────────────────────────────
    ax.text(0.02, 0.02,
            "Edge width ∝ |ρ|  (thicker = stronger correlation)",
            transform=ax.transAxes, fontsize=9, ha="left", va="bottom",
            color="gray", style="italic")

    # ── 統計資訊 ─────────────────────────────────────────
    stats = (
        f"L1 density: {n_l1/max_pairs*100:.0f}%  "
        f"L2 density: {n_l2/max_pairs*100:.0f}%  "
        f"Features: 9-dim"
    )
    ax.text(0.98, 0.02, stats, transform=ax.transAxes,
            fontsize=9, ha="right", va="bottom", color="gray")

    # ── 自動調整視窗 ─────────────────────────────────────
    all_pos = np.vstack([l1_pos, l2_pos])
    margin = 1.5
    ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin + 0.5)

    plt.tight_layout()
    os.makedirs(Path(out_path).parent, exist_ok=True)
    plt.savefig(out_path, dpi=200, facecolor="white", edgecolor="none")
    plt.close()
    print(f"✅ {out_path}")
    print(f"   Date: {target_date}")
    print(f"   L1 edges: {n_l1}/{max_pairs}  L2 edges: {n_l2}/{max_pairs}")


# ════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multiplex GNN 圖結構視覺化（proposal 風格）")
    parser.add_argument("--date", default=None,
                        help="指定日期（如 2024-01-15），不指定則自動挑選")
    parser.add_argument("--out", default=None,
                        help="輸出路徑")
    parser.add_argument("--no-a12", action="store_true",
                        help="不畫 A12 跨層連線（只展示 intra-layer 邊）")
    args = parser.parse_args()

    out_path = args.out or str(Path(FIG_DIR) / "multiplex_graph_snapshot.png")
    data, date_str = load_snapshot(args.date)
    draw_multiplex_graph(data, date_str, out_path, show_a12=not args.no_a12)


if __name__ == "__main__":
    main()