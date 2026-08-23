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

W, H = 1850, 1090

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
# 教授建議 4「減少圖標」：拿掉 API 圓片、資料庫圓柱、堆疊卡片與卡片內的
# 假節點塗鴉。留下來的圖形只有三種——矩形、真實矩陣熱圖、雙層平面示意——
# 每一種都在傳遞圖上讀得到的資訊，沒有純裝飾的元素。
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


def box(x, y, w, h, lines, fill="#ffffff", stroke=INK, sw=1.6, rx=7,
        head_size=16.5, body_size=13.5, head_fill=None, body_fill=None):
    """一個方塊 = 標題 + 若干行說明。所有文字都畫在框內，不靠圖說解碼。"""
    rect(x, y, w, h, fill, stroke, sw, rx)
    cx = x + w / 2
    n = len(lines)
    total = head_size + 6 + (n - 1) * (body_size + 6)
    yy = y + (h - total) / 2 + head_size
    text(cx, yy, lines[0], head_size, head_fill or INK, "middle", "700")
    for ln in lines[1:]:
        yy += body_size + 6
        text(cx, yy, ln, body_size, body_fill or GREY)


def region(x, y, w, h, title, tsize=19):
    """教授建議 3：Phase 標題畫在虛線框『外面』，不與框內元件爭空間。"""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="none" '
        f'stroke="{DASH}" stroke-width="2" stroke-dasharray="10,7"/>')
    text(x, y - 11, title, tsize, INK, "start", "700")


def arrow(x1, y1, x2, y2, colour=LINE, sw=1.8, marker="tri", dash=None,
          label=None, lsize=13, lgap=9, lanchor="middle", lx=None, ly=None):
    """一律細線箭頭；label 直接寫在線旁，說明這條線在搬什麼。"""
    da = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{colour}" '
        f'stroke-width="{sw}"{da} marker-end="url(#{marker})"/>')
    if label:
        mx = lx if lx is not None else (x1 + x2) / 2
        my = ly if ly is not None else (y1 + y2) / 2 - lgap
        text(mx, my, label, lsize, GREY, lanchor, "400", "italic")


def elbow(pts, colour=LINE, sw=1.8, marker="tri", dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(pts))
    add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{sw}"{da} '
        f'marker-end="url(#{marker})"/>')


def probe(x, y, value, note):
    """線性探針讀數。教授建議 5：數字旁一定要有它是什麼的說明。"""
    text(x, y, f"linear probe   test IC {value}", 14, RED, "middle", "700",
         "italic")
    text(x, y + 19, note, 13, RED, "middle", "400", "italic")


def heat(x, y, w, h, arr: np.ndarray, title=None, tsize=16, sub=None,
         rowlab=None, collab=None):
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
        text(x + w / 2, y - (26 if sub else 12), title, tsize, INK, "middle",
             "700")
    if sub:
        text(x + w / 2, y - 9, sub, 13, GREY)
    if collab:
        text(x + w / 2, y + h + 17, collab, 13, GREY)
    if rowlab:
        add(f'<text transform="translate({x-10},{y+h/2}) rotate(-90)" '
            f'font-size="13" fill="{GREY}" text-anchor="middle">'
            f'{esc(rowlab)}</text>')


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
    DY = 654

    # ───── 左：資料來源與特徵建構 ─────
    box(28, 196, 168, 78, ["US daily OHLCV", "30 tickers"])
    box(28, 196 + DY, 168, 78, ["TW daily OHLCV", "50 tickers"])
    text(112, 976, "Yahoo Finance", 13, GREY, "middle", "400", "italic")
    text(112, 996, "2019-01 to 2025-12", 13, GREY, "middle", "400", "italic")

    rect(214, 196, 42, 732, GRN_F, GRN_S, 1.8, rx=9)
    add(f'<text transform="translate(237,562) rotate(-90)" font-size="18" '
        f'fill="#22452a" text-anchor="middle" font-weight="700">'
        f'Feature and graph construction</text>')
    for yy in (234, 234 + DY):
        arrow(198, yy, 210, yy)
        arrow(258, yy, 270, yy)
    text(28, 1030, "9 technical indicators computed, 3 fed to the model", 13.5,
         GREY, "start", "400", "italic")
    text(28, 1052, f"edge drawn iff |correlation| > {d['tau']}, "
                   f"60-day window ending at t-1", 13.5, GREY, "start", "400",
         "italic")

    # ───── Phase 1 / L1（美股層）─────
    # 每個方塊的文字都放得下框寬；維度標註畫在方塊上緣之上，不壓到框線。
    region(262, 54, 690, 300, "Phase 1 — US market layer")
    box(272, 200, 112, 68, ["US features", "day t, 3 each"])
    arrow(386, 234, 420, 234, label="30 x 3", ly=186)
    box(424, 194, 150, 80, ["Shared LSTM", "hidden 64"])
    text(499, 292, "same weights as the TW layer", 13, GREY, "middle", "400",
         "italic")
    arrow(576, 234, 610, 234, label="30 x 64", ly=186)
    heat(633, 74, 92, 92, rgb_adj(d["A1"]))
    text(737, 94, "US correlation graph", 14, INK, "start", "700")
    text(737, 114, f"day t, {d['n_edge_1']} directed edges", 13, GREY, "start")
    text(737, 134, "rebuilt every day", 13, GREY, "start")
    text(737, 154, "attention also reads |corr|", 13, GREY, "start")
    arrow(679, 168, 679, 190, DASH, 1.5, "tri", "5,4")
    box(614, 194, 130, 80, ["GATv2", "1 layer, 1 head"])
    arrow(746, 234, 780, 234, label="30 x 64", ly=186)
    box(784, 194, 158, 80,
        ["Type projection", "linear, GELU, LayerNorm", "output 29 dims"],
        body_size=13)
    text(863, 322, "Within-market information", 14.5, GREY, "middle", "400",
         "italic")
    probe(330, 150, "+0.0874", "raw features, before the encoder")

    # ───── Phase 1 / L2（台股層）─────
    region(262, 54 + DY, 690, 300, "Phase 1 — Taiwan market layer")
    box(272, 200 + DY, 112, 68, ["TW features", "day t, 3 each"])
    arrow(386, 234 + DY, 420, 234 + DY, label="50 x 3", ly=186 + DY)
    box(424, 194 + DY, 150, 80, ["Shared LSTM", "hidden 64"])
    text(499, 292 + DY, "same weights as the US layer", 13, GREY, "middle",
         "400", "italic")
    arrow(576, 234 + DY, 610, 234 + DY, label="50 x 64", ly=186 + DY)
    heat(633, 74 + DY, 92, 92, rgb_adj(d["A2"]))
    text(737, 94 + DY, "TW correlation graph", 14, INK, "start", "700")
    text(737, 114 + DY, f"day t, {d['n_edge_2']} directed edges", 13, GREY,
         "start")
    text(737, 134 + DY, "same rule as the US layer", 13, GREY, "start")
    text(737, 154 + DY, "its own GAT weights", 13, GREY, "start")
    arrow(679, 168 + DY, 679, 190 + DY, DASH, 1.5, "tri", "5,4")
    box(614, 194 + DY, 130, 80, ["GATv2", "1 layer, 1 head"])
    arrow(746, 234 + DY, 780, 234 + DY, label="50 x 64", ly=186 + DY)
    box(784, 194 + DY, 158, 80,
        ["Type projection", "linear, GELU, LayerNorm", "output 32 dims"],
        body_size=13)
    text(863, 322 + DY, "Within-market information", 14.5, GREY, "middle",
         "400", "italic")

    # ───── 中：跨層接線示意 ─────
    text(276, 386, "Cross-layer wiring", 15, INK, "start", "700")
    p_us = plane(272, 402, 200, 80, 42, "#dff0df", "#7aa77a",
                 [(.10, .25), (.34, .64), (.55, .18), (.75, .66), (.93, .30)],
                 [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)], "#3f9c4f", "#2c6e3a",
                 ["TSM", "UMC", "ASX", "AMD", "CHT"])
    p_tw = plane(272, 544, 200, 80, 42, "#fde3cf", "#d79a6a",
                 [(.10, .25), (.30, .66), (.52, .20), (.73, .66), (.91, .30)],
                 [(0, 1), (1, 3), (2, 3), (3, 4), (0, 2)], "#e07b39", "#a8542a",
                 ["2330", "2303", "3711", "2454", "2412"])
    for k, kind in [(0, "id"), (1, "id"), (2, "id"), (4, "id"), (3, "cand")]:
        col, dsh, wd = ((INK, "6,0", 1.9) if kind == "id"
                        else ("#9aa5b1", "4,5", 1.2))
        add(f'<line x1="{p_us[k][0]:.1f}" y1="{p_us[k][1]:.1f}" '
            f'x2="{p_tw[k][0]:.1f}" y2="{p_tw[k][1]:.1f}" stroke="{col}" '
            f'stroke-width="{wd}" stroke-dasharray="{dsh}"/>')
    text(276, 648, "solid: the 7 ADR pairs, weight fixed at 1", 13.5, GREY,
         "start", "400", "italic")
    text(276, 668, "dashed: the other 43 TW stocks, learned weights", 13.5,
         GREY, "start", "400", "italic")

    # ───── 中：raw skip 與拼接 ─────
    box(516, 400, 264, 108,
        ["Raw skip", "each feature normalised across nodes",
         "then a 3 to 3 linear map", "output 3 dims"],
        AMB_F, AMB_S, 2.2, 9, 18, 13.5, "#7c3f06", "#7c4a09")
    elbow([(328, 270), (328, 356), (504, 356), (504, 454), (510, 454)],
          AMB_S, 2.2, "triA", "8,5")
    arrow(782, 454, 789, 454, AMB_S, 2.2, "triA")
    box(793, 400, 140, 108, ["concat", "29 + 3", "32 dims"], "#ffffff",
        AMB_S, 2.0, 9, 18, 14)
    arrow(863, 276, 863, 392)
    probe(863, 542, "+0.0754", "at the coupling point, with the skip")
    text(863, 580, "without the skip it is +0.0448", 13, RED, "middle", "400",
         "italic")

    # ───── Phase 2 ─────
    X2, CX2 = 1008, 1246
    region(X2, 250, 476, 700, "Phase 2 — Cross-market coupling and fusion")
    heat(X2 + 28, 372, 250, 150, rgb_beta(d["B"], d["pair"]),
         title="Learned coupling weights",
         sub="every US stock may reach every TW stock",
         rowlab="30 US stocks", collab="50 TW stocks")
    for i, (cx, lab) in enumerate([
            (IDENT, "the 7 ADR pairs — a separate path, weight fixed at 1"),
            ("#15528f", "learned weight, positive"),
            ("#a02d28", "learned weight, negative"),
    ]):
        yy = 556 + i * 24
        add(f'<rect x="{X2+28}" y="{yy-12}" width="16" height="16" '
            f'fill="{cx}" stroke="{BLUE_S}" stroke-width="0.9"/>')
        text(X2 + 54, yy + 1, lab, 13.5, GREY, "start")
    text(X2 + 28, 644, "1,493 candidate edges, none pruned to zero", 13.5, RED,
         "start", "400", "italic")
    text(X2 + 28, 666,
         "largest weight 0.018, median 0.003 — sparsity penalty off",
         13.5, RED, "start", "400", "italic")

    box(X2 + 28, 686, 424, 100,
        ["Cross-market aggregation",
         "each TW stock adds its ADR partner, if it has one,",
         "plus a weighted sum over all 30 US stocks",
         "the 43 unpaired stocks get only the weighted sum"],
        BLUE_F, "#5b7fa6", 1.8, 9, 17, 13.5, "#173d61", "#3d5f7f")
    arrow(CX2, 788, CX2, 804)
    box(X2 + 28, 812, 424, 130,
        ["Gated fusion", "one valve per dimension decides how much",
         "US state to mix into each TW stock",
         "measured flat: 0.502, spread across stocks 0.0005"],
        AMB_F, AMB_S, 2.2, 9, 19, 13.5, "#7c3f06", "#7c4a09")

    elbow([(937, 432), (978, 432), (978, 306), (X2 + 22, 306)], RED, 2, "triR",
          "9,5")
    text(978, 296, "30 x 32", 13, RED, "middle", "400", "italic")
    elbow([(946, 234 + DY), (978, 234 + DY), (978, 876), (X2 + 22, 876)],
          "#2f5fa8", 2, "triB", "9,5")
    text(978, 916, "50 x 32", 13, "#2f5fa8", "middle", "400", "italic")

    # ───── Phase 3 ─────
    X3, CX3 = 1510, 1663
    region(X3, 250, 306, 700, "Phase 3 — Ranking", 19)
    elbow([(X2 + 452, 876), (1494, 876), (1494, 380), (X3 + 72, 380)])
    box(X3 + 76, 340, 152, 84, ["Prediction head", "32 to 64 to 1", "one ReLU"])
    text(CX3, 452, "model output   test IC +0.0475", 14, RED, "middle", "700",
         "italic")
    text(CX3, 471, "63% of what the coupling point holds", 13, RED, "middle",
         "400", "italic")
    arrow(CX3, 484, CX3, 512)
    box(X3 + 26, 520, 252, 122,
        ["Predicted return", "one number per TW stock, day t+1",
         "ranked across the 50 stocks",
         "scored by cross-sectional IC"])
    text(CX3, 664, "246 test days, 10 seeds", 15, RED, "middle", "700")
    arrow(CX3, 682, CX3, 706)
    box(X3 + 26, 714, 252, 132,
        ["Training objective", "squared error, weight 1",
         "ranking loss, weight 0.5", "variance floor, weight 0.1",
         "gradient shares 27 / 71 / 2 percent"],
        BLUE_F, "#5b7fa6", 1.8, 9, 17, 13.5, "#173d61", "#3d5f7f")
    text(CX3, 880, "walk-forward split, never shuffled", 13.5, GREY, "middle",
         "400", "italic")
    text(CX3, 902, "model chosen on validation IC", 13.5, GREY, "middle",
         "400", "italic")

    # ───── 圖例 ─────
    add(f'<rect x="1040" y="1016" width="22" height="15" fill="{AMB_F}" '
        f'stroke="{AMB_S}" stroke-width="1.6"/>')
    text(1072, 1029, "added in MAGNET-v2", 14, GREY, "start")
    add(f'<line x1="1240" y1="1024" x2="1280" y2="1024" stroke="{RED}" '
        f'stroke-width="2" stroke-dasharray="9,5"/>')
    text(1290, 1029, "US side entering the coupling", 14, GREY, "start")
    add(f'<line x1="1520" y1="1024" x2="1560" y2="1024" stroke="#2f5fa8" '
        f'stroke-width="2" stroke-dasharray="9,5"/>')
    text(1570, 1029, "TW side entering the gate", 14, GREY, "start")
    text(1040, 1056, f"Correlation graphs are the real {date} snapshot; "
                     f"coupling weights are the trained checkpoint.", 13.5,
         GREY, "start", "400", "italic")

    add("</svg>")
    return "\n".join(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="產生 MAGNET 架構圖")
    ap.add_argument("--run", default="runs/20260816_1601_tw50_T1F3bnl1_s42")
    ap.add_argument("--date", default="2025-04-10")
    ap.add_argument("--out", default="docs/figures/magnet_flow_v5.svg")
    a = ap.parse_args()

    d = load_real(ROOT / a.run, a.date)
    out = ROOT / a.out
    out.write_text(build(d, a.date))
    print(f"wrote {out}")
    print(f"  A1 {d['n_edge_1']} edges / A2 {d['n_edge_2']} edges @ {a.date}")
    print(f"  B  |max| {np.abs(d['B']).max():.5f}, "
          f"non-zero {(np.abs(d['B']) > 1e-8).sum()} / {d['B'].size}")


if __name__ == "__main__":
    main()
