"""
gen_tw_relation_figure.py — 台股層的「同產業邊」與「不同產業邊」示意圖（英文版）。

以台股層平面（L2: TWSE；平面形狀、配色與節點樣式沿用既有的台股層示意圖）為底，畫 11 檔台股：
    同產業邊    這 11 檔裡同產業的每一對都畫（實線）
    不同產業邊  只畫示範股（2330）到其他產業的各檔（虛線）。每一檔都和所有
                不同產業的股票相連，全畫出來會蓋住同產業的結構，所以只畫一檔
節點依產業分群擺放（同產業排成正多邊形）。產業標籤、每群在 50 檔中的檔數與兩種關係的邊數，
都從 universe 的 industry 欄與模型的 _build_tw_relations 算出，不寫死。

主結果（SA，proposal §80、§83）只用同產業邊；不同產業關係在元件消融（§79）之後拿掉，圖下註腳寫明。

    .venv/bin/python scripts/gen_tw_relation_figure.py

輸出：docs/figures/tw_industry_edges_en.svg
產生後會檢查：文字兩兩不重疊、文字不壓到節點與邊、邊不穿過非端點的節點、節點都在平面內。
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.config import load_universe  # noqa: E402
from src.models.multiplex_gnn import MAGNET  # noqa: E402

W, H = 1800, 900          # 與架構圖相同的 2:1

INK = "#1f2933"           # 以下三色與 gen_magnet_figure.py 相同
GREY = "#5b6672"
LINE = "#7d8a99"
PLANE = "#ffcc99"         # 以下三色取自既有的台股層示意圖
NODE = "#ea6b12"
NODE_RIM = "#fb9563"
SAME = "#b45309"          # 同產業邊：架構圖 v7 的琥珀深色（本版新增）
DIFF = LINE               # 不同產業邊：灰藍虛線

FOCAL = "2330"
EN = {                    # 公開資訊觀測站產業類別的英文簡稱
    "半導體業": "Semiconductors",
    "金融保險業": "Finance and insurance",
    "電子零組件業": "Electronic parts",
    "其他電子業": "Other electronics",
    "通信網路業": "Communications",
}

# 平面：既有示意圖的平行四邊形（左上角 + 上緣向量 U + 左緣向量 V），等比縮小
TL = (470.0, 34.0)
U = (1112.0, 98.0)
V = (-189.0, 573.0)
R_NODE = 52.0


def P(u: float, v: float) -> tuple[float, float]:
    """平面座標 (u, v) -> 畫布座標。u 沿上緣（0 左、1 右），v 沿左緣（0 上、1 下）。"""
    return (TL[0] + u * U[0] + v * V[0], TL[1] + u * U[1] + v * V[1])


def polygon_pts(center, ru, rv, angles_deg):
    cu, cv = center
    return [(cu + ru * math.cos(math.radians(a)), cv - rv * math.sin(math.radians(a))) for a in angles_deg]


# 節點位置（平面座標）。半導體排成正五邊形、金融排成正三角形（群內全連）；
# 示範股 2330 在五邊形朝右的頂點，虛線往右散開，不穿過自己的群。
_semi = polygon_pts((0.245, 0.47), 0.11, 0.255, (0, 72, 144, 216, 288))
_fin = polygon_pts((0.80, 0.395), 0.07, 0.20, (0, 120, 240))
LAYOUT = {
    "2330": _semi[0], "2454": _semi[1], "2303": _semi[2], "3711": _semi[3], "8150": _semi[4],
    "2882": _fin[0], "2881": _fin[1], "2891": _fin[2],
    "2308": (0.46, 0.765), "2317": (0.65, 0.765), "2412": (0.84, 0.765),
}

s: list[str] = []
add = s.append
TEXTS: list[dict] = []    # 供碰撞檢查：每個文字的估計外框與它該在平面內或外


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Helvetica 的字寬（em）估計，只供碰撞檢查
_NARROW = set("ijlI.,:;'|!")
_SEMI = set("frt()-[] ")
_WIDE = set("mwMW")


def text_width(t: str, size: float) -> float:
    w = 0.0
    for ch in t:
        if ch in _NARROW:
            w += 0.25
        elif ch in _SEMI:
            w += 0.33
        elif ch in _WIDE:
            w += 0.86
        elif ch.isupper():
            w += 0.68
        else:
            w += 0.56
    return w * size


def text(x, y, t, size=18, fill=INK, anchor="middle", weight="400", style="normal",
         where="inside", check=True, markup=None):
    body = markup if markup is not None else esc(t)
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}" font-style="{style}">{body}</text>')
    if check:
        w = text_width(t, size)
        x0 = {"start": x, "middle": x - w / 2, "end": x - w}[anchor]
        TEXTS.append({"t": t, "box": (x0, y - 0.74 * size, x0 + w, y + 0.24 * size), "where": where})


# ══════════════════════════ 幾何檢查 ══════════════════════════
def seg_point_dist(a, b, p) -> float:
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(ax + t * dx - px, ay + t * dy - py)


def boxes_overlap(b1, b2, pad=2.0) -> bool:
    return not (b1[2] + pad <= b2[0] or b2[2] + pad <= b1[0] or b1[3] + pad <= b2[1] or b2[3] + pad <= b1[1])


def box_circle_overlap(b, c, r) -> bool:
    nx = min(max(c[0], b[0]), b[2]); ny = min(max(c[1], b[1]), b[3])
    return math.hypot(nx - c[0], ny - c[1]) < r


def seg_box_intersect(a, b, box) -> bool:
    """線段與矩形是否相交（取樣檢查，畫布像素尺度下足夠）。"""
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 2))
    for i in range(n + 1):
        x = a[0] + (b[0] - a[0]) * i / n; y = a[1] + (b[1] - a[1]) * i / n
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            return True
    return False


def plane_poly():
    return [P(0, 0), P(1, 0), P(1, 1), P(0, 1)]


def inside_poly(p, poly, margin=0.0) -> bool:
    """凸多邊形（順時針）內、且離每條邊至少 margin。"""
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross / math.hypot(b[0] - a[0], b[1] - a[1]) < margin:
            return False
    return True


def box_corners(b):
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def run_checks(pos, edges) -> list[str]:
    bad = []
    poly = plane_poly()
    for a, b in combinations(pos, 2):
        d = math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1])
        if d < 2 * R_NODE + 24:
            bad.append(f"節點太近：{a}-{b} 距離 {d:.0f}")
    for c, p in pos.items():
        if not inside_poly(p, poly, R_NODE + 8):
            bad.append(f"節點 {c} 太靠近平面邊緣")
    for kind, a, b in edges:
        for c, p in pos.items():
            if c in (a, b):
                continue
            d = seg_point_dist(pos[a], pos[b], p)
            if d < R_NODE + 10:
                bad.append(f"{kind}邊 {a}-{b} 離節點 {c} 只有 {d:.0f}")
    for t1, t2 in combinations(TEXTS, 2):
        if boxes_overlap(t1["box"], t2["box"]):
            bad.append(f"文字重疊：{t1['t']!r} / {t2['t']!r}")
    for t in TEXTS:
        if t["where"] == "node":
            continue
        for c, p in pos.items():
            if box_circle_overlap(t["box"], p, R_NODE + 4):
                bad.append(f"文字 {t['t']!r} 壓到節點 {c}")
        for kind, a, b in edges:
            if seg_box_intersect(pos[a], pos[b], t["box"]):
                bad.append(f"文字 {t['t']!r} 壓到{kind}邊 {a}-{b}")
        corners = box_corners(t["box"])
        if t["where"] == "inside" and not all(inside_poly(q, poly, 6) for q in corners):
            bad.append(f"文字 {t['t']!r} 超出平面")
        if t["where"] == "outside" and any(inside_poly(q, poly, -6) for q in corners):
            bad.append(f"文字 {t['t']!r} 壓到平面")
        if not (0 <= t["box"][0] and t["box"][2] <= W and 0 <= t["box"][1] and t["box"][3] <= H):
            bad.append(f"文字 {t['t']!r} 超出畫布")
    return bad


# ══════════════════════════ 圖 ══════════════════════════
def build() -> tuple[str, list[str]]:
    u = load_universe("tw50")
    lab = {c: u.industry[c] for c in u.tw_nodes}
    missing = [c for c in LAYOUT if c not in lab]
    if missing:
        raise ValueError(f"{missing} 不在 tw50 universe")
    size = Counter(lab.values())
    M = MAGNET._build_tw_relations(["same_industry", "other_industry"], u)
    n_same, n_diff = int((M[0] > 0).sum()), int((M[1] > 0).sum())
    n_ind = len(size)
    n_alone = sum(1 for c in u.tw_nodes if size[lab[c]] == 1)
    shown = list(LAYOUT)
    groups: dict[str, list[str]] = {}
    for c in shown:
        groups.setdefault(lab[c], []).append(c)
    unknown = [g for g in groups if g not in EN]
    if unknown:
        raise ValueError(f"產業 {unknown} 沒有英文名稱")

    pos = {c: P(*LAYOUT[c]) for c in shown}
    same_edges = [("同產業", a, b) for a, b in combinations(shown, 2) if lab[a] == lab[b]]
    diff_edges = [("不同產業", FOCAL, b) for b in shown if lab[b] != lab[FOCAL]]

    add(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="\'Helvetica Neue\', Arial, sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    poly = plane_poly()
    add('<path d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in poly) + f' Z" fill="{PLANE}"/>')
    lx, ly = 98, 391
    text(lx, ly, "L2: TWSE", 47, "#000000", "start", where="outside",
         markup='<tspan font-style="italic">L2</tspan>: TWSE')

    def rim_line(a, b, colour, width, dash=None):
        (ax, ay), (bx, by) = pos[a], pos[b]
        d = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / d, (by - ay) / d
        x1, y1 = ax + ux * R_NODE, ay + uy * R_NODE
        x2, y2 = bx - ux * R_NODE, by - uy * R_NODE
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" '
            f'stroke-width="{width}" stroke-linecap="round"{extra}/>')

    for _, a, b in same_edges:
        rim_line(a, b, SAME, 3.4)
    for _, a, b in diff_edges:
        rim_line(a, b, DIFF, 2.4, "9,7")

    for c in shown:
        x, y = pos[c]
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{R_NODE}" fill="{NODE}" stroke="{NODE_RIM}" stroke-width="3"/>')
        text(x, y + 10.5, c, 30, "#ffffff", "middle", "500", where="node")

    # 產業標籤：多檔的群標在群的上方，單檔的標在節點下方；第二行是這 11 檔裡有幾檔 / 50 檔裡有幾檔
    def group_label(x, y, name, k, n):
        text(x, y, name, 22, INK, "middle", "700")
        text(x, y + 24, f"{k} of {n} shown", 17, GREY)

    for g, members in groups.items():
        xs = [pos[c][0] for c in members]; ys = [pos[c][1] for c in members]
        if len(members) > 1:
            group_label(sum(xs) / len(xs), min(ys) - R_NODE - 40, EN[g], len(members), size[g])
        else:
            group_label(xs[0], ys[0] + R_NODE + 29, EN[g], 1, size[g])

    # 圖例（平面右下方的空白）：第二行是全部 50 檔的有向邊數
    def legend(x, y, colour, width, dash, name, lines):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        add(f'<line x1="{x}" y1="{y - 7}" x2="{x + 46}" y2="{y - 7}" stroke="{colour}" '
            f'stroke-width="{width}" stroke-linecap="round"{extra}/>')
        TEXTS.append({"t": f"圖例線段（{name}）", "box": (x - 2, y - 10, x + 48, y - 4), "where": "outside"})
        text(x + 60, y, name, 20, INK, "start", "700", where="outside")
        for i, ln in enumerate(lines):
            text(x + 60, y + 24 * (i + 1), ln, 16.5, GREY, "start", where="outside")

    legend(1450, 604, SAME, 3.4, None, "Same-industry edge",
           [f"{n_same} directed edges (all {u.n_l2})", f"{len(same_edges)} pairs here, all drawn"])
    legend(1450, 692, DIFF, 2.4, "9,7", "Different-industry edge",
           [f"{n_diff:,} directed edges (all {u.n_l2})", f"drawn for {FOCAL} only"])

    # 註腳（與 docs/experimental_setup.md 同口徑）
    fy = 800
    text(60, fy, f"{len(shown)} of the {u.n_l2} TW stocks shown; each line stands for two directed edges. "
         f"Industries are MOPS 2018Q4 categories, fixed for the whole sample "
         f"({n_ind} industries; {n_alone} stocks have no same-industry peer).",
         16.5, GREY, "start", where="outside")
    text(60, fy + 27, "Every stock has a different-industry edge to each stock outside its industry.",
         16.5, GREY, "start", where="outside")
    text(60, fy + 54, "MAGNET (main result) passes messages along same-industry edges only; "
         "the different-industry relation was dropped after the component ablation.",
         16.5, GREY, "start", where="outside")

    add("</svg>")
    return "\n".join(s), run_checks(pos, same_edges + diff_edges)


def main() -> None:
    ap = argparse.ArgumentParser(description="台股層同產業邊與不同產業邊示意圖")
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "tw_industry_edges_en.svg"))
    args = ap.parse_args()
    svg, bad = build()
    if "marker-end" in svg:
        bad.append("圖上出現箭頭標記")
    out = Path(args.out)
    out.write_text(svg + "\n")
    print(f"-> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}（{len(TEXTS)} 個文字）")
    if bad:
        print(f"檢查發現 {len(bad)} 處：")
        for b in bad:
            print("  -", b)
        sys.exit(1)
    print("檢查：文字 0 處重疊、0 處壓到節點或邊；邊不穿過其他節點；節點都在平面內")


if __name__ == "__main__":
    main()
