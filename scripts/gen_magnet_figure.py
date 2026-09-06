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

W, H = 1800, 900          # 規則 6：寬高比 2:1

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


# ══════════════════════════ 繪圖基元 ══════════════════════════
# 規則 2：只保留熱圖、圓柱、堆疊卡片、平面示意四種圖形，
#         而且**全圖只有一種箭頭樣式**（細實線 + 單一箭頭標記）。
def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, t, size=16, fill=INK, anchor="middle", weight="400",
         style="normal"):
    add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" '
        f'font-style="{style}">{esc(t)}</text>')


def rect(x, y, w, h, fill="#ffffff", stroke=INK, sw=1.5, rx=0):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def box(x, y, w, h, lines, fill="#ffffff", stroke=INK, sw=1.6, rx=6,
        hs=15.5, bs=12.5, hf=None, bf=None):
    rect(x, y, w, h, fill, stroke, sw, rx)
    cx, n = x + w / 2, len(lines)
    total = hs + 5 + (n - 1) * (bs + 5)
    yy = y + (h - total) / 2 + hs
    text(cx, yy, lines[0], hs, hf or INK, "middle", "700")
    for ln in lines[1:]:
        yy += bs + 5
        text(cx, yy, ln, bs, bf or GREY)


def region(x, y, w, h, title, tsize=17):
    """規則 1：Phase 標題畫在虛線框外面。"""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" '
        f'stroke="{DASH}" stroke-width="1.8" stroke-dasharray="9,6"/>')
    text(x, y - 10, title, tsize, INK, "start", "700")


def arrow(pts, label=None, lx=None, ly=None):
    """全圖唯一的箭頭樣式。pts 為折線的頂點串。"""
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}"
                 for i, (x, y) in enumerate(pts))
    add(f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="1.7" '
        f'marker-end="url(#tri)"/>')
    if label:
        mx = lx if lx is not None else (pts[0][0] + pts[-1][0]) / 2
        my = ly if ly is not None else (pts[0][1] + pts[-1][1]) / 2 - 8
        text(mx, my, label, 12, GREY, "middle", "400", "italic")


def line(pts):
    """無箭頭的匯流線。匯流排不是資料流箭頭，故不掛 marker，
    「單一箭頭樣式」的規則不受影響。"""
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}"
                 for i, (x, y) in enumerate(pts))
    add(f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="1.7"/>')


def oplus(cx, cy, r=15):
    """加總節點。全圖唯一破例的第五種圖形。

    理由：教授指出 Phase 2 畫成三條上下相連的 Path，會被讀成順序關係。
    三項實際上是同時從同一個輸入算出、相加。三線收斂到一個節點是
    「合併」的通用語法，不可能被讀成「然後」——這個語意重要到值得
    破例。替代方案（三線交會但不畫圓）視覺重量不足。"""
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" '
        f'stroke="{INK}" stroke-width="1.8"/>')
    k = r * 0.52
    add(f'<path d="M{cx-k},{cy} H{cx+k} M{cx},{cy-k} V{cy+k}" fill="none" '
        f'stroke="{INK}" stroke-width="1.8"/>')


def cylinder(cx, top, w, h, label1, label2):
    rx, ry = w / 2, 11
    add(f'<path d="M{cx-rx},{top+ry} v{h-2*ry} a{rx},{ry} 0 0 0 {2*rx},0 '
        f'v{-(h-2*ry)}" fill="#fdfdfd" stroke="{INK}" stroke-width="1.6"/>')
    add(f'<ellipse cx="{cx}" cy="{top+ry}" rx="{rx}" ry="{ry}" fill="#ffffff" '
        f'stroke="{INK}" stroke-width="1.6"/>')
    text(cx, top + h / 2 + 3, label1, 15, INK, "middle", "700")
    text(cx, top + h / 2 + 22, label2, 12.5, GREY)


def stack(x, y, w, h, lines, n=3, off=6):
    """堆疊卡片：代表「每張日快照各跑一次」。"""
    for k in range(n - 1, 0, -1):
        add(f'<rect x="{x + k * off}" y="{y - k * off}" width="{w}" '
            f'height="{h}" rx="4" fill="#ffffff" stroke="{LINE}" '
            f'stroke-width="1.2"/>')
    box(x, y, w, h, lines)


def heat(x, y, w, h, arr, title=None, sub=None, rowlab=None, collab=None):
    im = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO(); im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
        f'preserveAspectRatio="none" image-rendering="pixelated" '
        f'href="data:image/png;base64,{b64}" '
        f'xlink:href="data:image/png;base64,{b64}"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" '
        f'stroke="{BLUE_S}" stroke-width="1.5"/>')
    if title:
        text(x + w / 2, y - (22 if sub else 9), title, 14, INK, "middle", "700")
    if sub:
        text(x + w / 2, y - 7, sub, 11.5, GREY)
    if collab:
        text(x + w / 2, y + h + 15, collab, 11.5, GREY)
    if rowlab:
        add(f'<text transform="translate({x-9},{y+h/2}) rotate(-90)" '
            f'font-size="11.5" fill="{GREY}" text-anchor="middle">'
            f'{esc(rowlab)}</text>')


def plane(x, y, w, h, skew, fill, stroke, nodes, edges, ncol, nstroke, labels):
    add(f'<path d="M{x+skew},{y} L{x+w+skew},{y} L{x+w},{y+h} L{x},{y+h} Z" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.3" opacity="0.85"/>')
    pos = [(x + u * w + (1 - v) * skew, y + v * h) for (u, v) in nodes]
    for a, b in edges:
        add(f'<line x1="{pos[a][0]:.1f}" y1="{pos[a][1]:.1f}" '
            f'x2="{pos[b][0]:.1f}" y2="{pos[b][1]:.1f}" stroke="{nstroke}" '
            f'stroke-width="1.2" opacity="0.75"/>')
    for i, (px, py) in enumerate(pos):
        add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="{ncol}" '
            f'stroke="{nstroke}" stroke-width="1.2"/>')
        text(px, py + 3, labels[i], 8, "#ffffff", "middle", "700")
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




# ══════════════════════════ 版面（2:1）══════════════════════════
# 定案風格（2026-08-25）：
#   1 Phase 標題畫在虛線框外面
#   2 全圖只有一種箭頭樣式；保留熱圖、圓柱、堆疊卡片、平面示意四種圖形
#   3 維度 / 節點數 / 邊數直接標在圖上
#   4 不放算式，耦合路徑一律白話描述
#   5 只畫架構，不放任何 IC 或探針讀數
#   6 寬高比 2:1
#   Phase 1 是**單一個框**，內含美股列、跨層接線、台股列。
def build(d: dict, date: str) -> str:
    s.clear()
    add(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'font-family="\'Helvetica Neue\', Arial, sans-serif">')
    add(f'<defs><marker id="tri" markerWidth="9" markerHeight="9" refX="6.5" '
        f'refY="3" orient="auto"><path d="M0,0 L6.5,3 L0,6 Z" '
        f'fill="{LINE}"/></marker></defs>')
    add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    DY = 430

    # ───── 左：資料來源 ─────
    cylinder(96, 258, 118, 86, "US OHLCV", "30 tickers")
    cylinder(96, 273 + DY, 118, 86, "TW OHLCV", "50 tickers")
    text(96, 390, "Yahoo Finance", 11.5, GREY, "middle", "400", "italic")
    text(96, 406, "2019-01 to 2025-12", 11.5, GREY, "middle", "400", "italic")

    rect(170, 128, 34, 672, GRN_F, GRN_S, 1.6, rx=7)
    add(f'<text transform="translate(187,459) rotate(-90)" font-size="15" '
        f'fill="#22452a" text-anchor="middle" font-weight="700">'
        f'Feature and graph construction</text>')
    for yy in (301, 316 + DY):
        arrow([(158, yy), (166, yy)])
        arrow([(206, yy), (222, yy)])
    text(24, 818, "9 technical indicators computed, 3 fed to the model",
         12, GREY, "start", "400", "italic")
    text(24, 836, f"edge drawn iff |correlation| > {d['tau']}, "
                  f"60-day window ending at t-1", 12, GREY, "start", "400",
         "italic")

    # ───── Phase 1：單一個框，含兩個市場列與中間的跨層接線 ─────
    region(226, 128, 720, 672, "Phase 1 — Dual-market encoding")
    for tag, dy, n, A, ekey, note in [
            ("US", 0, 30, d["A1"], "n_edge_1", "shared LSTM weights"),
            ("TW", DY, 50, d["A2"], "n_edge_2", "same LSTM weights"),
    ]:
        box(238, 268 + dy, 112, 66, [f"{tag} features", "day t", "3 per stock"])
        arrow([(352, 301 + dy), (378, 301 + dy)], f"{n} x 3", ly=259 + dy)
        box(380, 268 + dy, 116, 66,
            ["Input norm", "per feature,", "across stocks"], AMB_F, AMB_S, 1.8)
        arrow([(498, 301 + dy), (524, 301 + dy)])
        stack(526, 268 + dy, 120, 66, ["Shared LSTM", "hidden 64"])
        text(586, 354 + dy, note, 11.5, GREY, "middle", "400", "italic")
        arrow([(654, 301 + dy), (680, 301 + dy)], f"{n} x 64", ly=259 + dy)
        heat(700, 174 + dy, 60, 60, rgb_adj(A))
        text(772, 194 + dy, f"{tag} correlation graph", 13, INK, "start", "700")
        text(772, 212 + dy, f"{d[ekey]} directed edges", 11.5, GREY, "start")
        text(772, 229 + dy, "rebuilt every day", 11.5, GREY, "start")
        arrow([(730, 238 + dy), (730, 266 + dy)])
        stack(682, 268 + dy, 118, 66, ["GATv2", "1 layer, 1 head"])
        arrow([(808, 301 + dy), (828, 301 + dy)])
        box(830, 268 + dy, 116, 66, ["Type projection", "to 32 dims"])
        # 標籤縮短為不含 "32,"：維度在左邊的 Type projection 框裡已經寫了。
        # Phase 1 與 Phase 2 的虛線框只隔 38px，任何較長的標籤都會壓到框線；
        # Gated fusion 下移到 y=652 之後這一點才顯出來。
        arrow([(948, 301 + dy), (982, 301 + dy)],
              "to Phase 2" if dy == 0 else "to fusion",
              lx=964, ly=259 + dy)

    # ───── Phase 1 中段：跨層接線 ─────
    text(238, 390, "Cross-layer connecting", 14, INK, "start", "700")
    p_us = plane(242, 404, 178, 56, 34, "#dff0df", "#7aa77a",
                 [(.10, .25), (.34, .64), (.55, .18), (.75, .66), (.93, .30)],
                 [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)], "#3f9c4f", "#2c6e3a",
                 ["TSM", "UMC", "ASX", "AMD", "CHT"])
    p_tw = plane(242, 490, 178, 56, 34, "#fde3cf", "#d79a6a",
                 [(.10, .25), (.30, .66), (.52, .20), (.73, .66), (.91, .30)],
                 [(0, 1), (1, 3), (2, 3), (3, 4), (0, 2)], "#e07b39", "#a8542a",
                 ["2330", "2303", "3711", "2454", "2412"])
    for k in (0, 1, 2, 4):
        add(f'<line x1="{p_us[k][0]:.1f}" y1="{p_us[k][1]:.1f}" '
            f'x2="{p_tw[k][0]:.1f}" y2="{p_tw[k][1]:.1f}" stroke="{INK}" '
            f'stroke-width="1.7"/>')
    text(474, 414, "7 TW stocks have a US listing", 12, GREY, "start")
    text(474, 432, "of the same company", 12, GREY, "start")
    text(474, 464, "the other 43 have none —", 12, GREY, "start")
    text(474, 482, "they reach the US side only through the", 12, GREY, "start")
    text(474, 500, "market factor and residual at right", 12, GREY, "start")

    # ───── Phase 2 ─────
    # 2026-08-31 改版。教授指出舊版把三項畫成上下相連的 "Path 1/2/3"，
    # 會被讀成順序關係。實際上三項是同時從同一個 h1 算出、相加。四個修正：
    #   (a) 刪掉框與框之間的箭頭，改成「左側匯流排分出、右側匯流排收回、
    #       收斂到 ⊕」——三線收歛到一點是「合併」的通用語法
    #   (b) 拿掉 "Path N —" 字樣，只留名字（沒有詞就沒有詞會被誤讀）
    #   (c) 新增分流節點：h1 先拆成橫截面平均與殘差，兩者互斥地餵給
    #       市場因子與殘差結構。這是三項為什麼是三項的真正結構，
    #       也正面回答「②+③ 為什麼不合起來」的質疑（proposal §33、§46）
    #   (d) B 的熱圖不再掛在資料流上——B 是參數不是資料，且它只屬於
    #       殘差項。畫在最上方但不接任何箭頭，標題直接寫明歸屬
    region(984, 128, 470, 672, "Phase 2 — Factor-exposure coupling and fusion")
    heat(1104, 190, 216, 84, rgb_beta(d["B"], d["pair"]),
         title="Learned residual coupling",
         sub="one weight per US-TW pair, used by the residual term only",
         rowlab="30 US stocks", collab="50 TW stocks")
    # 分流節點：本版真正新增的東西是「把市場暴露與殘差分開」，故標為新增
    box(1000, 306, 438, 54,
        ["Split the US state",
         "a cross-sectional mean, and what remains of each stock"],
        AMB_F, AMB_S, 1.8, hf="#7c3f06", bf="#7c4a09")
    # 框寬 386 而非原本的 438：左右各讓出匯流排的空間。內文同步縮短到
    # 386px 放得下（12.5px 字約 59 字元），避免溢出框線。
    box(1022, 390, 386, 52,
        ["ADR partner",
         "the 7 paired stocks copy their US twin, one weight each"])
    box(1022, 452, 386, 52,
        ["US market factor",
         "the mean, with one learnable exposure per stock"],
        AMB_F, AMB_S, 1.8, hf="#7c3f06", bf="#7c4a09")
    box(1022, 514, 386, 52,
        ["Residual structure",
         "each US stock after its market factor is removed"])
    # 左側匯流排：三項同源、同時
    line([(1219, 360), (1219, 372), (1004, 372), (1004, 540)])
    for cy in (416, 478, 540):
        arrow([(1004, cy), (1020, cy)])
    # 右側匯流排收回 ⊕：三項相加，不是相接
    for cy in (416, 478, 540):
        line([(1408, cy), (1428, cy)])
    line([(1428, 416), (1428, 540)])
    arrow([(1428, 540), (1428, 572)])
    oplus(1428, 588, 14)
    arrow([(1428, 602), (1428, 624), (1219, 624), (1219, 650)])
    box(1000, 652, 438, 74,
        ["Gated fusion",
         "one valve per dimension decides how much",
         "US state to mix into each TW stock"],
        BLUE_F, "#5b7fa6", 1.8, hf="#173d61", bf="#3d5f7f")
    arrow([(982, 301), (992, 301), (992, 333), (998, 333)])
    arrow([(982, 301 + DY), (992, 301 + DY), (992, 689), (998, 689)])

    # ───── Phase 3 ─────
    region(1486, 128, 300, 672, "Phase 3 — Rank-oriented prediction")
    arrow([(1438, 689), (1462, 689), (1462, 288), (1520, 288)])
    stack(1522, 254, 226, 68, ["Prediction head", "32 to 64 to 1, one ReLU"])
    arrow([(1636, 322), (1636, 362)])
    box(1500, 364, 268, 96,
        ["Predicted return", "one number per TW stock, day t+1",
         "ranked across the 50 stocks"])
    arrow([(1636, 460), (1636, 500)])
    box(1500, 502, 268, 96,
        ["Training objective", "squared error, ranking loss,",
         "variance floor"],
        BLUE_F, "#5b7fa6", 1.8, hf="#173d61", bf="#3d5f7f")
    text(1636, 640, "AdamW, walk-forward split, never shuffled",
         12, GREY, "middle", "400", "italic")
    text(1636, 660, "model chosen on validation IC", 12, GREY, "middle",
         "400", "italic")

    # ───── 圖例 ─────
    add(f'<rect x="984" y="812" width="18" height="12" fill="{AMB_F}" '
        f'stroke="{AMB_S}" stroke-width="1.4"/>')
    text(1010, 822, "new in this version", 12, GREY, "start")
    text(1160, 822, f"heatmaps are real data — correlation graphs from {date}, "
                    f"trained coupling weights", 12, GREY, "start", "400",
         "italic")

    add("</svg>")
    return "\n".join(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="產生 MAGNET 架構圖")
    # runs/ 於 2026-08-30 改成樹狀結構（commit 6c23b78），預設路徑同步更新
    ap.add_argument("--run", default="runs/tw50/beta/20260824_1740_tw50_beta_s42")
    ap.add_argument("--date", default="2025-04-10")
    ap.add_argument("--out", default="docs/figures/magnet_flow_v6.svg")
    a = ap.parse_args()
    d = load_real(ROOT / a.run, a.date)
    out = ROOT / a.out
    out.write_text(build(d, a.date))
    print(f"wrote {out}  ({W}x{H}, ratio {W/H:.2f}:1)")


if __name__ == "__main__":
    main()
