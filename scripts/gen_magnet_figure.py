"""
gen_magnet_figure.py — 產生 MAGNET 架構圖（MEIG 風格）。

風格參照：水平流向、虛線分區、資料庫圓柱、鄰接矩陣熱圖、堆疊卡片、
傾斜的多層圖平面、空心塊狀箭頭、浮動斜體註解。

三張矩陣熱圖是**真實資料**，不是示意：
    A_1(t), A_2(t)  取 --date 當天快照的 |rho|（> tau 才成邊）
    B               取 --run 的 checkpoint 中 weak_beta * weak_mask
所以圖上的數字與 docs/results_table.md、docs/roadmap_stage_a.md 同源；
換 run 或換日期只要重跑本腳本，圖會跟著更新。

    python scripts/gen_magnet_figure.py
    python scripts/gen_magnet_figure.py --date 2025-04-07
"""
from __future__ import annotations

import argparse
import base64
import io
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models._universe import universe_from_cfg  # noqa: E402

W, H = 1790, 1010

INK = "#1f2933"
GREY = "#5b6672"
LINE = "#7d8a99"
DASH = "#2b2b2b"
BLUE_F, BLUE_S = "#d7e5f7", "#9fb8d8"
AMB_F, AMB_S = "#fbe4a8", "#b45309"
GRN_F, GRN_S = "#cfe3cf", "#4d7c4d"
RED = "#a3282d"
IDENT = "#f0a23c"

s: list[str] = []
add = s.append


# ══════════════════════════ 繪圖基元 ══════════════════════════
def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, t, size=17, fill=INK, anchor="middle", weight="400",
         style="normal"):
    add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" '
        f'font-style="{style}">{esc(t)}</text>')


def rect(x, y, w, h, fill="#ffffff", stroke=INK, sw=1.5, rx=0):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def region(x, y, w, h, title, tsize=19):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="none" '
        f'stroke="{DASH}" stroke-width="2" stroke-dasharray="10,7"/>')
    text(x + 22, y + 36, title, tsize, INK, "start", "700")


def block_arrow(x, y, length=42):
    add(f'<g transform="translate({x},{y})">'
        f'<path d="M0,-7 L{length-18},-7 L{length-18},-14 L{length},0 '
        f'L{length-18},14 L{length-18},7 L0,7 Z" '
        f'fill="#ffffff" stroke="{LINE}" stroke-width="1.4"/></g>')


def stack(x, y, w, h, n=3, off=7, inner=None):
    for k in range(n - 1, 0, -1):
        add(f'<rect x="{x + k * off}" y="{y - k * off}" width="{w}" '
            f'height="{h}" fill="#ffffff" stroke="{LINE}" stroke-width="1.3"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" '
        f'stroke="{LINE}" stroke-width="1.6"/>')
    if inner == "graph":
        rnd = random.Random(7)
        pts = [(x + 20 + rnd.random() * (w - 40), y + 18 + rnd.random() * (h - 36))
               for _ in range(5)]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                if (a + b) % 2 == 0:
                    add(f'<line x1="{pts[a][0]:.1f}" y1="{pts[a][1]:.1f}" '
                        f'x2="{pts[b][0]:.1f}" y2="{pts[b][1]:.1f}" '
                        f'stroke="#9bbf9b" stroke-width="1.2"/>')
        for px, py in pts:
            add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="#bcdcbc" '
                f'stroke="#5f9160" stroke-width="1.2"/>')
    elif inner == "seq":
        for k in range(4):
            yy = y + 16 + k * (h - 30) / 3
            add(f'<line x1="{x + 16}" y1="{yy:.1f}" x2="{x + w - 16}" '
                f'y2="{yy:.1f}" stroke="#c3ccd6" stroke-width="2.6"/>')


def cylinder(cx, top, w, h, label1, label2):
    rx, ry = w / 2, 13
    add(f'<path d="M{cx-rx},{top+ry} v{h-2*ry} a{rx},{ry} 0 0 0 {2*rx},0 '
        f'v{-(h-2*ry)}" fill="#fdfdfd" stroke="{INK}" stroke-width="1.6"/>')
    add(f'<ellipse cx="{cx}" cy="{top+ry}" rx="{rx}" ry="{ry}" fill="#ffffff" '
        f'stroke="{INK}" stroke-width="1.6"/>')
    text(cx, top + h / 2 + 4, label1, 17, INK, "middle", "700")
    text(cx, top + h / 2 + 25, label2, 14, GREY)


def api_chip(x, y):
    add(f'<g transform="translate({x},{y})">'
        f'<circle cx="13" cy="13" r="12" fill="#f5c451" stroke="#b8860b" '
        f'stroke-width="1.4"/>'
        f'<circle cx="13" cy="13" r="4.6" fill="#ffffff" stroke="#b8860b" '
        f'stroke-width="1.2"/></g>')
    text(x + 13, y + 42, "API", 12, GREY)


def heat(x, y, w, h, arr: np.ndarray, title=None, tsize=16):
    """把真實矩陣以 base64 PNG 內嵌；pixelated 放大，保留每一格。"""
    im = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
        f'preserveAspectRatio="none" image-rendering="pixelated" '
        f'href="data:image/png;base64,{b64}" '
        f'xlink:href="data:image/png;base64,{b64}"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" '
        f'stroke="{BLUE_S}" stroke-width="1.8"/>')
    if title:
        text(x + w / 2, y - 12, title, tsize, INK, "middle", "700")


def plane(x, y, w, h, skew, fill, stroke, nodes, edges, ncol, nstroke, labels):
    add(f'<path d="M{x+skew},{y} L{x+w+skew},{y} L{x+w},{y+h} L{x},{y+h} Z" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4" opacity="0.85"/>')
    pos = [(x + u * w + (1 - v) * skew, y + v * h) for (u, v) in nodes]
    for a, b in edges:
        add(f'<line x1="{pos[a][0]:.1f}" y1="{pos[a][1]:.1f}" '
            f'x2="{pos[b][0]:.1f}" y2="{pos[b][1]:.1f}" stroke="{nstroke}" '
            f'stroke-width="1.3" opacity="0.75"/>')
    for i, (px, py) in enumerate(pos):
        add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="9" fill="{ncol}" '
            f'stroke="{nstroke}" stroke-width="1.3"/>')
        text(px, py + 3.5, labels[i], 9, "#ffffff", "middle", "700")
    return pos


# ══════════════════════════ 真實資料 ══════════════════════════
def load_real(run_dir: Path, date: str) -> dict:
    cfg_path = run_dir / "config_snapshot.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    u = universe_from_cfg(cfg)
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt",
                                     map_location="cpu",
                                     weights_only=False)["model_state_dict"])
    model.eval()

    ds = MultiplexDataset(snapshot_dir=str(ROOT / cfg["data"]["snapshot_dir"]),
                          features_dir=str(ROOT / cfg["data"]["features_dir"]),
                          T=cfg["model"]["lstm"]["T_history"], split="test",
                          config_path=str(cfg_path))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=multiplex_collate, num_workers=0)
    batch = None
    for b in loader:
        if b["target_date"][0] == date:
            batch = b
            break
    if batch is None:
        raise SystemExit(f"test split 中找不到 {date}")

    def dense(ei, ea, n):
        A = np.zeros((n, n), dtype=np.float32)
        e = ei[0].numpy() if isinstance(ei, list) else ei.numpy()
        w = (ea[0].numpy() if isinstance(ea, list) else ea.numpy()).reshape(-1)
        A[e[0], e[1]] = w
        return A

    A1 = dense(batch["edge_index_L1"], batch["edge_attr_L1"], u.n_l1)
    A2 = dense(batch["edge_index_L2"], batch["edge_attr_L2"], u.n_l2)
    B = (model.weak_beta * model.weak_mask).detach().numpy()
    pair = [(i, j) for j, i in enumerate(u.pair_index) if i >= 0]
    return {"A1": A1, "A2": A2, "B": B, "pair": pair, "u": u,
            "n_edge_1": int((A1 > 0).sum()), "n_edge_2": int((A2 > 0).sum()),
            "tau": 0.3}


def rgb_adj(A: np.ndarray) -> np.ndarray:
    """無邊 = 近白；有邊時 |rho| 由淺到深藍。"""
    out = np.empty(A.shape + (3,), dtype=np.float32)
    out[...] = np.array([242, 246, 251], dtype=np.float32)
    m = A > 0
    if m.any():
        v = A[m]
        lo, hi = 0.3, max(float(v.max()), 0.31)
        t = np.clip((v - lo) / (hi - lo), 0, 1)[:, None]
        c0 = np.array([190, 214, 238], dtype=np.float32)
        c1 = np.array([26, 71, 118], dtype=np.float32)
        out[m] = c0 * (1 - t) + c1 * t
    np.fill_diagonal(out[..., 0], 255.0)
    np.fill_diagonal(out[..., 1], 255.0)
    np.fill_diagonal(out[..., 2], 255.0)
    return out


def rgb_beta(B: np.ndarray, pair) -> np.ndarray:
    """B 用發散色階（負紅 / 零白 / 正藍）；恆等邊位置塗橘。"""
    out = np.empty(B.shape + (3,), dtype=np.float32)
    a = float(np.abs(B).max()) or 1.0
    t = np.clip(B / a, -1, 1)
    white = np.array([250, 251, 253], dtype=np.float32)
    pos = np.array([21, 82, 143], dtype=np.float32)
    neg = np.array([160, 45, 40], dtype=np.float32)
    tp = np.clip(t, 0, 1)[..., None]
    tn = np.clip(-t, 0, 1)[..., None]
    out[...] = white * (1 - tp - tn) + pos * tp + neg * tn
    for i, j in pair:
        out[i, j] = np.array([240, 162, 60], dtype=np.float32)
    return out


# ══════════════════════════ 版面 ══════════════════════════
def build(d: dict, date: str) -> str:
    s.clear()
    add(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'font-family="\'Helvetica Neue\', Arial, sans-serif">')
    mk = "".join(
        f'<marker id="{n}" markerWidth="10" markerHeight="10" refX="7" '
        f'refY="3.4" orient="auto"><path d="M0,0 L7,3.4 L0,6.8 Z" '
        f'fill="{c}"/></marker>'
        for n, c in [("tri", LINE), ("triA", AMB_S), ("triR", RED),
                     ("triB", "#2f5fa8")])
    add(f"<defs>{mk}</defs>")
    add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # ───── 左：資料來源 ─────
    api_chip(20, 268)
    cylinder(106, 250, 122, 100, "US OHLCV", "30 tickers")
    api_chip(20, 618)
    cylinder(106, 600, 122, 100, "TW OHLCV", "50 tickers")
    text(106, 748, "Yahoo Finance, daily", 14, GREY, "middle", "400", "italic")
    text(106, 770, "2019-01 to 2025-12", 14, GREY, "middle", "400", "italic")

    # ───── 特徵與圖建構層 ─────
    rect(176, 226, 46, 500, GRN_F, GRN_S, 1.8, rx=9)
    add(f'<text transform="translate(205,476) rotate(-90)" font-size="18" '
        f'fill="#22452a" text-anchor="middle" font-weight="700">'
        f'Feature &amp; Graph Construction Layer</text>')
    text(16, 812, "9 indicators, 3 used", 14, GREY, "start")
    text(16, 834, f"edge iff |rho| > {d['tau']}, 60-day window", 14, GREY, "start")
    block_arrow(230, 300, 28)
    block_arrow(230, 650, 28)

    # ───── 上：Phase 1 / L1 ─────
    region(244, 22, 700, 292, "Phase 1 — L1: US ADR market layer")
    rect(262, 162, 110, 68, "#ffffff", INK, 1.5, rx=7)
    text(317, 190, "x₁ᵢ(t)", 18, INK, "middle", "700")
    text(317, 212, "3 features", 14, GREY)
    block_arrow(380, 196, 38)
    stack(426, 156, 112, 82, 3, 7, inner="seq")
    text(482, 268, "Shared LSTM", 16, INK, "middle", "700")
    text(482, 290, "H = 64", 14, GREY)
    block_arrow(550, 196, 34)
    heat(608, 38, 88, 88, rgb_adj(d["A1"]))
    text(708, 74, "A₁(t)", 16, INK, "start", "700")
    text(708, 96, f"{d['n_edge_1']} edges", 14, GREY, "start")
    add(f'<path d="M652,130 L652,162" stroke="{DASH}" stroke-width="1.5" '
        f'stroke-dasharray="5,4" marker-end="url(#tri)"/>')
    stack(596, 170, 112, 82, 3, 7, inner="graph")
    text(652, 282, "GATv2", 16, INK, "middle", "700")
    text(652, 304, "1 layer, 4 heads", 14, GREY)
    block_arrow(720, 196, 34)
    rect(768, 158, 168, 76, "#ffffff", INK, 1.5, rx=7)
    text(852, 182, "Type projection", 16, INK, "middle", "700")
    text(852, 203, "Linear → GELU → LN", 13.5, GREY)
    text(852, 225, "29 dims", 15, INK, "middle", "700")

    # ───── 下：Phase 1 / L2 ─────
    DY = 654
    region(244, 22 + DY, 700, 292, "Phase 1 — L2: Taiwan market layer")
    rect(262, 162 + DY, 110, 68, "#ffffff", INK, 1.5, rx=7)
    text(317, 190 + DY, "x₂ⱼ(t)", 18, INK, "middle", "700")
    text(317, 212 + DY, "3 features", 14, GREY)
    block_arrow(380, 196 + DY, 38)
    stack(426, 156 + DY, 112, 82, 3, 7, inner="seq")
    text(482, 268 + DY, "Shared LSTM", 16, INK, "middle", "700")
    text(482, 290 + DY, "same weights", 14, GREY)
    block_arrow(550, 196 + DY, 34)
    heat(608, 38 + DY, 88, 88, rgb_adj(d["A2"]))
    text(708, 74 + DY, "A₂(t)", 16, INK, "start", "700")
    text(708, 96 + DY, f"{d['n_edge_2']} edges", 14, GREY, "start")
    add(f'<path d="M652,{130+DY} L652,{162+DY}" stroke="{DASH}" '
        f'stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#tri)"/>')
    stack(596, 170 + DY, 112, 82, 3, 7, inner="graph")
    text(652, 282 + DY, "GATv2", 16, INK, "middle", "700")
    text(652, 304 + DY, "independent weights", 14, GREY)
    block_arrow(720, 196 + DY, 34)
    rect(768, 158 + DY, 168, 76, "#ffffff", INK, 1.5, rx=7)
    text(852, 182 + DY, "Type projection", 16, INK, "middle", "700")
    text(852, 203 + DY, "Linear → GELU → LN", 13.5, GREY)
    text(852, 225 + DY, "32 dims", 15, INK, "middle", "700")

    # 共用 LSTM
    add(f'<path d="M482,238 C556,420 556,600 482,{156+DY}" fill="none" '
        f'stroke="#9aa5b1" stroke-width="1.4" stroke-dasharray="6,5"/>')
    text(566, 500, "shared", 14, GREY, "middle", "400", "italic")
    text(566, 520, "LSTM weights", 14, GREY, "middle", "400", "italic")

    # ───── 中：多層圖示意 ─────
    p_us = plane(254, 356, 196, 84, 42, "#dff0df", "#7aa77a",
                 [(.10, .25), (.34, .64), (.55, .18), (.75, .66), (.93, .30)],
                 [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)], "#3f9c4f", "#2c6e3a",
                 ["TSM", "UMC", "ASX", "AMD", "CHT"])
    p_tw = plane(254, 520, 196, 84, 42, "#fde3cf", "#d79a6a",
                 [(.10, .25), (.30, .66), (.52, .20), (.73, .66), (.91, .30)],
                 [(0, 1), (1, 3), (2, 3), (3, 4), (0, 2)], "#e07b39", "#a8542a",
                 ["2330", "2303", "3711", "2454", "2412"])
    for k, kind in [(0, "id"), (1, "id"), (2, "id"), (4, "id"), (3, "cand")]:
        col, dsh, wd = ((INK, "6,0", 1.9) if kind == "id"
                        else ("#9aa5b1", "4,5", 1.2))
        add(f'<line x1="{p_us[k][0]:.1f}" y1="{p_us[k][1]:.1f}" '
            f'x2="{p_tw[k][0]:.1f}" y2="{p_tw[k][1]:.1f}" stroke="{col}" '
            f'stroke-width="{wd}" stroke-dasharray="{dsh}"/>')
    text(352, 634, "7 identity edges (solid)", 14, GREY, "middle", "400",
         "italic")
    text(352, 656, "43 unpaired TW nodes (dashed)", 14, GREY, "middle", "400",
         "italic")

    # ───── 中：raw skip + concat ─────
    rect(506, 366, 256, 108, AMB_F, AMB_S, 2.2, rx=9)
    text(634, 396, "Raw skip", 19, "#7c3f06", "middle", "700")
    text(634, 420, "BN_F(x₁) → Linear(3→3), no bias", 13.5, "#7c4a09")
    text(634, 440, "feature-wise norm, across nodes", 13.5, "#7c4a09")
    text(634, 464, "3 dims", 15, "#7c3f06", "middle", "700")
    add(f'<path d="M317,230 L317,330 L494,330 L494,378 L500,378" fill="none" '
        f'stroke="{AMB_S}" stroke-width="2.2" stroke-dasharray="8,5" '
        f'marker-end="url(#triA)"/>')
    rect(800, 366, 136, 108, "#ffffff", AMB_S, 2.0, rx=9)
    text(868, 400, "concat", 18, INK, "middle", "700")
    text(868, 426, "d′ = 29 + 3", 14, GREY)
    text(868, 452, "= 32", 17, INK, "middle", "700")
    add(f'<path d="M886,236 L886,358" fill="none" stroke="{LINE}" '
        f'stroke-width="1.7" marker-end="url(#tri)"/>')
    add(f'<path d="M764,420 L794,420" fill="none" stroke="{AMB_S}" '
        f'stroke-width="2.2" marker-end="url(#triA)"/>')
    text(868, 506, "+0.0754", 15, RED, "middle", "700", "italic")
    text(868, 528, "(+0.0448 without skip)", 13.5, RED, "middle", "400",
         "italic")

    # ───── 右中：Phase 2 ─────
    region(962, 222, 470, 710, "Phase 2 — Two-tier coupling & gated fusion")
    heat(988, 316, 250, 150, rgb_beta(d["B"], d["pair"]))
    text(1113, 306, "B ⊙ M   (30 US × 50 TW)", 17, INK, "middle", "700")
    for i, (cx, lab) in enumerate([
            (IDENT, "identity edge, fixed = 1  (7)"),
            ("#15528f", "candidate B > 0"),
            ("#a02d28", "candidate B < 0"),
    ]):
        yy = 496 + i * 24
        add(f'<rect x="988" y="{yy-12}" width="16" height="16" fill="{cx}" '
            f'stroke="{BLUE_S}" stroke-width="0.9"/>')
        text(1014, yy + 1, lab, 14, GREY, "start")
    text(988, 592, "1,493 candidate edges, 100% non-zero", 14, RED, "start",
         "400", "italic")
    text(988, 614, "|B| max 0.018, median 0.003, λ = 0", 14, RED, "start",
         "400", "italic")

    rect(988, 644, 420, 106, BLUE_F, "#5b7fa6", 1.8, rx=9)
    text(1198, 672, "Cross-market aggregation", 17, "#173d61", "middle", "700")
    text(1198, 698, "ĥ₁ⱼ = h₁ₚ₍ⱼ₎ + Σᵢ B_eff[i,j] · h₁ᵢ", 16, "#173d61")
    text(1198, 720, "identity term absent for the 43 unpaired j", 13.5,
         "#3d5f7f")
    text(1198, 740, "candidate share 12% paired / 7% unpaired", 13.5, "#3d5f7f")

    rect(988, 762, 420, 140, AMB_F, AMB_S, 2.2, rx=9)
    text(1198, 792, "Gated fusion", 19, "#7c3f06", "middle", "700")
    text(1198, 818, "gⱼ = σ(W_g[ĥ₁ⱼ ; h₂ⱼ] + b_g) ∈ [0,1]³²", 15, "#7c4a09")
    text(1198, 842, "h_f = g ⊙ ĥ₁ + (1 − g) ⊙ h₂", 16, "#7c4a09")
    text(1198, 866, "one valve for all cross-market flow", 13.5, "#8a5a12",
         "middle", "400", "italic")
    text(1198, 888, "measured flat: 0.5018, node sd 0.0005", 13.5, RED,
         "middle", "400", "italic")
    add(f'<path d="M1198,750 L1198,756" fill="none" stroke="{LINE}" '
        f'stroke-width="1.7" marker-end="url(#tri)"/>')

    # concat -> 耦合（紅）；L2 -> gate（藍）
    add(f'<path d="M940,420 L960,420 L960,392 L982,392" fill="none" '
        f'stroke="{RED}" stroke-width="2" stroke-dasharray="9,5" '
        f'marker-end="url(#triR)"/>')
    add(f'<path d="M940,{196+DY} L960,{196+DY} L960,832 L982,832" fill="none" '
        f'stroke="#2f5fa8" stroke-width="2" stroke-dasharray="9,5" '
        f'marker-end="url(#triB)"/>')

    # ───── 右：Phase 3 ─────
    region(1464, 222, 306, 710, "Phase 3 — Rank-oriented prediction", 17)
    stack(1550, 300, 130, 92, 3, 7, inner="seq")
    text(1615, 424, "Prediction head", 16, INK, "middle", "700")
    text(1615, 446, "32 → 64 → 1, ReLU", 14, GREY)
    add(f'<path d="M1408,832 L1440,832 L1440,346 L1544,346" fill="none" '
        f'stroke="{LINE}" stroke-width="1.7" marker-end="url(#tri)"/>')
    add(f'<path d="M1615,452 L1615,494" fill="none" stroke="{LINE}" '
        f'stroke-width="1.7" marker-end="url(#tri)"/>')
    rect(1488, 502, 260, 132, "#ffffff", INK, 1.6, rx=9)
    text(1618, 532, "ŷⱼ(t+1)", 19, INK, "middle", "700")
    text(1618, 558, "daily ranking of the 50 TW stocks", 14, GREY)
    text(1618, 580, "scored by cross-sectional IC", 14, GREY)
    text(1618, 610, "IC = +0.0439  (10 seeds)", 16, RED, "middle", "700")
    add(f'<path d="M1618,640 L1618,676" fill="none" stroke="{LINE}" '
        f'stroke-width="1.7" marker-end="url(#tri)"/>')
    rect(1488, 684, 260, 140, BLUE_F, "#5b7fa6", 1.8, rx=9)
    text(1618, 714, "Combined loss", 17, "#173d61", "middle", "700")
    text(1618, 742, "L = L_MSE", 15, "#173d61")
    text(1618, 764, "+ 0.5·L_rank + 0.1·L_var", 15, "#173d61")
    text(1618, 790, "gradient shares 27 / 71 / 2 %", 13.5, "#3d5f7f")
    text(1618, 812, "L_align removed after ablation", 13.5, "#3d5f7f")
    text(1618, 878, "walk-forward split, no shuffling", 13.5, GREY, "middle",
         "400", "italic")
    text(1618, 900, "early stopping on validation IC", 13.5, GREY, "middle",
         "400", "italic")

    # ───── 探針註解 ─────
    text(317, 148, "+0.0874", 15, RED, "middle", "700", "italic")
    text(1615, 478, "+0.0439", 15, RED, "middle", "700", "italic")
    text(852, 288, "Within-market information", 14.5, GREY, "middle", "400",
         "italic")
    text(852, 288 + DY, "Within-market information", 14.5, GREY, "middle",
         "400", "italic")
    text(1198, 636, "Cross-market information", 14.5, GREY, "middle", "400",
         "italic")

    # ───── 圖例 ─────
    add(f'<rect x="970" y="950" width="22" height="15" fill="{AMB_F}" '
        f'stroke="{AMB_S}" stroke-width="1.6"/>')
    text(1002, 963, "new in MAGNET-v2", 14.5, GREY, "start")
    add(f'<line x1="1166" y1="958" x2="1206" y2="958" stroke="#2f5fa8" '
        f'stroke-width="2" stroke-dasharray="9,5"/>')
    text(1216, 963, "TW state enters the gate directly", 14.5, GREY, "start")
    add(f'<line x1="1466" y1="958" x2="1506" y2="958" stroke="{RED}" '
        f'stroke-width="2" stroke-dasharray="9,5"/>')
    text(1516, 963, "coupling input", 14.5, GREY, "start")
    text(970, 992, "+0.0874", 14.5, RED, "start", "700", "italic")
    text(1038, 992, f"= test IC a per-target ridge probe extracts there.  "
                    f"A₁/A₂: real {date} snapshot.  B: trained checkpoint.",
         14.5, GREY, "start")

    add("</svg>")
    return "\n".join(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="產生 MAGNET 架構圖")
    ap.add_argument("--run", default="runs/20260816_1601_tw50_T1F3bnl1_s42")
    ap.add_argument("--date", default="2025-04-10")
    ap.add_argument("--out", default="docs/figures/magnet_flow_v4.svg")
    a = ap.parse_args()

    d = load_real(ROOT / a.run, a.date)
    out = ROOT / a.out
    out.write_text(build(d, a.date))
    print(f"wrote {out}")
    print(f"  A₁ {d['n_edge_1']} edges / A₂ {d['n_edge_2']} edges @ {a.date}")
    print(f"  B  |max| {np.abs(d['B']).max():.5f}, "
          f"non-zero {(np.abs(d['B']) > 1e-8).sum()} / {d['B'].size}")


if __name__ == "__main__":
    main()
