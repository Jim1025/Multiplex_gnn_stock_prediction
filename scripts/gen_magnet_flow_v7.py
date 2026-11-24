"""Redraw magnet_flow_v7.svg: bigger bold text, less whitespace, no overlaps.
Heatmap PNGs are reused from the original file."""
import re, sys
from pathlib import Path

SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2])
imgs = list(dict.fromkeys(re.findall(r'href="(data:image/png;base64,[^"]+)"', SRC.read_text())))
assert len(imgs) == 3
US_IMG, TW_IMG, B_IMG = imgs

INK, MUTED, ARROW = "#1f2933", "#5b6672", "#7d8a99"
NEW = dict(fill="#fbe4a8", stroke="#b45309", sw=1.8)
BLUE = dict(fill="#d7e5f7", stroke="#5b7fa6", sw=1.8, tc="#173d61", sc="#3d5f7f")
PLAIN = dict(fill="#ffffff", stroke=INK, sw=1.6)
T_BIG, T_SUB, T_PHASE, T_SEC, T_NOTE = 17.5, 13.5, 19.5, 15.5, 12.5

o = []
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def text(x, y, s, size=T_SUB, fill=MUTED, anchor="middle", bold=False, italic=False, extra=""):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
             f'font-weight="{700 if bold else 400}" font-style="{"italic" if italic else "normal"}"{extra}>{esc(s)}</text>')
def rect(x, y, w, h, fill="none", stroke=INK, sw=1.6, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def path(d, arrow=True):
    m = ' marker-end="url(#tri)"' if arrow else ""
    o.append(f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="1.7"{m}/>')
def box(x, y, w, h, title, subs=(), st=PLAIN, tc=INK, sc=MUTED, stack=False):
    if stack:
        for k in (2, 1):
            rect(x + 6 * k, y - 6 * k, w, h, "#ffffff", ARROW, 1.2, 4)
    rect(x, y, w, h, st["fill"], st["stroke"], st["sw"])
    tc = st.get("tc", tc); sc = st.get("sc", sc)
    n = len(subs); lh = 17.5
    top = y + h / 2 - (T_BIG * 0.35 + n * lh) / 2 + T_BIG * 0.7 - 2
    text(x + w / 2, round(top, 1), title, T_BIG, tc, bold=True)
    for i, s in enumerate(subs):
        text(x + w / 2, round(top + (i + 1) * lh + 1, 1), s, T_SUB, sc)
def image(x, y, w, h, href):
    o.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" '
             f'image-rendering="pixelated" href="{href}"/>')
    rect(x, y, w, h, "none", "#9fb8d8", 1.5, 0)
def cyl(cx, cy, label, sub):
    rx, ry, hh = 62, 11, 66
    top = cy - hh / 2
    o.append(f'<path d="M{cx-rx},{top} v{hh} a{rx},{ry} 0 0 0 {2*rx},0 v-{hh}" fill="#fdfdfd" stroke="{INK}" stroke-width="1.6"/>')
    o.append(f'<ellipse cx="{cx}" cy="{top}" rx="{rx}" ry="{ry}" fill="#ffffff" stroke="{INK}" stroke-width="1.6"/>')
    text(cx, cy + 8, label, T_BIG, INK, bold=True)
    text(cx, cy + 28, sub, T_SUB)

W, H = 1980, 730
FT, FB = 58, 668                       # frame top / bottom
US_Y, TW_Y, BH = 160, 548, 72          # encoder rows (box top), box height
uc, tc_ = US_Y + BH / 2, TW_Y + BH / 2

# ---------------- left: data + construction bar ----------------
cyl(84, uc, "US OHLCV", "30 tickers")
cyl(84, tc_, "TW OHLCV", "50 tickers")
text(84, 355, "Yahoo Finance", T_NOTE, italic=True)
text(84, 373, "2019-01 to 2025-12", T_NOTE, italic=True)
rect(160, FT, 34, FB - FT, "#cfe3cf", "#4d7c4d", 1.6, 7)
o.append(f'<text transform="translate(183,{(FT+FB)/2}) rotate(-90)" font-size="{T_SEC+1}" fill="#22452a" '
         f'text-anchor="middle" font-weight="700">Feature and graph construction</text>')
for yy in (uc, tc_):
    path(f"M148,{yy} L156,{yy}"); path(f"M196,{yy} L206,{yy}")

# ---------------- Phase 1 ----------------
P1L, P1R = 210, 1018
rect(P1L, FT, P1R - P1L, FB - FT, "none", "#2b2b2b", 1.8, 6, "9,6")
text(P1L, FT - 14, "Phase 1 — Dual-market encoding", T_PHASE, INK, "start", True)
cols = [(222, 122), (370, 124), (522, 140), (700, 130), (858, 148)]   # (x, w)
def row(y, tag, gnn_title, gnn_sub, gnn_style, shared_note, dim1, dim2, out_note):
    (x0, w0), (x1, w1), (x2, w2), (x3, w3), (x4, w4) = cols
    c = y + BH / 2
    box(x0, y, w0, BH, f"{tag} features", ["day t", "1 per stock"])
    box(x1, y, w1, BH, "Input norm", ["per feature,", "across stocks"], NEW)
    box(x2, y, w2, BH, "Shared LSTM", ["hidden 64"], stack=True)
    box(x3, y, w3, BH, gnn_title, [gnn_sub], gnn_style, *(("#7c3f06", "#7c4a09") if gnn_style is NEW else ()), stack=True)
    box(x4, y, w4, BH, "Type projection", ["to 32 dims"])
    path(f"M{x0+w0+2},{c} L{x1-2},{c}"); path(f"M{x1+w1+2},{c} L{x2-2},{c}")
    path(f"M{x2+w2+14},{c} L{x3-2},{c}"); path(f"M{x3+w3+14},{c} L{x4-2},{c}")
    text((x0 + w0 + x1) / 2, y + BH + 18, dim1, T_NOTE, italic=True)
    text((x2 + w2 + 12 + x3) / 2, y + BH + 18, dim2, T_NOTE, italic=True)
    text(x2 + w2 / 2 + 6, y + BH + 34, shared_note, T_NOTE, italic=True)
row(US_Y, "US", "GATv2", "1 layer, 1 head", PLAIN, "shared LSTM weights", "30 x 1", "30 x 64", "to Phase 2")
row(TW_Y, "TW", "R-GCN", "self + peer mean", NEW, "same LSTM weights", "50 x 1", "50 x 64", "to fusion")

gx = cols[3][0] + 18
def graph_note(y_img, img, title, l1, l2):
    image(gx, y_img, 64, 64, img)
    text(gx + 78, y_img + 18, title, T_SEC, INK, "start", True)
    text(gx + 78, y_img + 38, l1, T_SUB, anchor="start")
    text(gx + 78, y_img + 57, l2, T_SUB, anchor="start")
graph_note(US_Y - 88, US_IMG, "US correlation graph", "802 directed edges", "rebuilt every day")
path(f"M{gx+32},{US_Y-22} L{gx+32},{US_Y-8}")
graph_note(TW_Y - 88, TW_IMG, "TW industry groups", "260 directed edges", "same industry, fixed")
path(f"M{gx+32},{TW_Y-22} L{gx+32},{TW_Y-8}")

# cross-layer mini graph
CY = 302
text(222, CY, "Cross-layer connecting", T_SEC + 0.5, INK, "start", True)
S, ox = 1.2, 222
def P(x, y, y0): return (round(ox + (x - 242) * S, 1), round(CY + 18 + (y - y0) * S, 1))
def plane(y0, fill, stroke, edge, node, nstroke, nodes, edges):
    pts = [P(276, y0, y0), P(454, y0, y0), P(420, y0 + 56, y0), P(242, y0 + 56, y0)]
    o.append('<path d="M' + " L".join(f"{a},{b}" for a, b in pts) + f' Z" fill="{fill}" stroke="{stroke}" stroke-width="1.3" opacity="0.85"/>')
    pos = {k: P(x, y, y0) for k, (x, y) in nodes.items()}
    for a, b in edges:
        o.append(f'<line x1="{pos[a][0]}" y1="{pos[a][1]}" x2="{pos[b][0]}" y2="{pos[b][1]}" stroke="{edge}" stroke-width="1.2" opacity="0.75"/>')
    return pos
us_nodes = {"TSM": (285.3, 418.0), "UMC": (314.8, 439.8), "ASX": (367.8, 414.1), "AMD": (387.1, 441.0), "CHT": (431.3, 420.8)}
tw_nodes = {"2330": (285.3, 504.0), "2303": (307.0, 527.0), "3711": (361.8, 501.2), "2454": (383.5, 527.0), "2412": (427.8, 506.8)}
# TW plane drawn 84 px (scaled) below the US plane
pu = plane(404, "#dff0df", "#7aa77a", "#2c6e3a", "#3f9c4f", "#2c6e3a", us_nodes,
           [("TSM", "UMC"), ("UMC", "ASX"), ("ASX", "AMD"), ("AMD", "CHT"), ("TSM", "ASX")])
o_us = len(o)
tw_shift = 82
def plane_tw():
    global CY
    CY += tw_shift
    p = plane(490, "#fde3cf", "#d79a6a", "#a8542a", "#e07b39", "#a8542a", tw_nodes,
              [("2330", "2303"), ("2303", "2454"), ("3711", "2454"), ("2454", "2412"), ("2330", "3711")])
    CY -= tw_shift
    return p
pt = plane_tw()
for a, b in [("TSM", "2330"), ("UMC", "2303"), ("ASX", "3711"), ("CHT", "2412")]:
    o.append(f'<line x1="{pu[a][0]}" y1="{pu[a][1]}" x2="{pt[b][0]}" y2="{pt[b][1]}" stroke="{INK}" stroke-width="1.7"/>')
def nodes(pos, fill, stroke):
    for k, (x, y) in pos.items():
        o.append(f'<circle cx="{x}" cy="{y}" r="11" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        text(x, round(y + 3.2, 1), k, 8.5, "#ffffff", bold=True)
nodes(pu, "#3f9c4f", "#2c6e3a"); nodes(pt, "#e07b39", "#a8542a")
tx = 490
for i, s in enumerate(["7 TW stocks have a US listing", "of the same company"]):
    text(tx, CY + 36 + i * 19, s, T_SUB, anchor="start")
for i, s in enumerate(["the other 43 have none —", "they reach the US side only through",
                       "the market factor and residual at right"]):
    text(tx, CY + 90 + i * 19, s, T_SUB, anchor="start")

# ---------------- Phase 2 ----------------
P2L, P2R = 1046, 1516
rect(P2L, FT, P2R - P2L, FB - FT, "none", "#2b2b2b", 1.8, 6, "9,6")
text(P2L, FT - 14, "Phase 2 — Factor-exposure coupling and fusion", T_PHASE, INK, "start", True)
mid = (P2L + P2R) / 2
text(mid, FT + 24, "Learned residual coupling", T_SEC + 0.5, INK, bold=True)
text(mid, FT + 42, "one weight per US-TW pair, used by the residual term only", T_NOTE)
image(mid - 120, FT + 48, 240, 84, B_IMG)
text(mid, FT + 147, "50 TW stocks", T_NOTE)
o.append(f'<text transform="translate({mid-128},{FT+90}) rotate(-90)" font-size="{T_NOTE}" fill="{MUTED}" text-anchor="middle">30 US stocks</text>')
bx, bw = P2L + 16, P2R - P2L - 32          # wide boxes
ix, iw = P2L + 60, P2R - P2L - 104         # inner boxes
SPL = 216
box(bx, SPL, bw, 58, "Split the US state", ["a cross-sectional mean, and what remains of each stock"], NEW, "#7c3f06", "#7c4a09")
text(ix, SPL + 84, "macro — the shared market state", T_NOTE + 0.5, INK, "start", True, True)
MF = SPL + 92
box(ix, MF, iw, 54, "US market factor", ["the mean, with one learnable exposure per stock"], NEW, "#7c3f06", "#7c4a09")
text(ix, MF + 74, "micro — which US stock it is", T_NOTE + 0.5, INK, "start", True, True)
AD = MF + 80; RS = AD + 60
box(ix, AD, iw, 54, "ADR partner", ["the 7 paired stocks copy their US twin, one weight each"])
box(ix, RS, iw, 54, "Residual structure", ["each US stock after its market factor is removed"])
lx, lx2, rx = P2L + 20, P2L + 40, ix + iw + 22
path(f"M{mid},{SPL+58} L{mid},{SPL+64} L{lx},{SPL+64} L{lx},{AD+27}", False)
path(f"M{lx},{MF+27} L{ix-2},{MF+27}")
path(f"M{lx},{AD+27} L{lx2},{AD+27} L{lx2},{RS+27}", False)
path(f"M{lx2},{AD+27} L{ix-2},{AD+27}"); path(f"M{lx2},{RS+27} L{ix-2},{RS+27}")
for yy in (MF + 27, AD + 27, RS + 27):
    path(f"M{ix+iw},{yy} L{rx},{yy}", False)
path(f"M{rx},{MF+27} L{rx},{RS+27}", False)
PL = RS + 70
path(f"M{rx},{RS+27} L{rx},{PL-16}")
o.append(f'<circle cx="{rx}" cy="{PL}" r="14" fill="#ffffff" stroke="{INK}" stroke-width="1.8"/>')
o.append(f'<path d="M{rx-7.3},{PL} H{rx+7.3} M{rx},{PL-7.3} V{PL+7.3}" fill="none" stroke="{INK}" stroke-width="1.8"/>')
GF = TW_Y - 4
path(f"M{rx},{PL+14} L{rx},{PL+18} L{mid},{PL+18} L{mid},{GF-2}")
box(bx, GF, bw, 80, "Gated fusion", ["one valve per dimension decides how much", "US state to mix into each TW stock"], BLUE)
# phase-1 outputs into phase 2
p1o = cols[4][0] + cols[4][1]
path(f"M{p1o+2},{uc} L{P2L+6},{uc} L{P2L+6},{SPL+29} L{bx-2},{SPL+29}")
path(f"M{p1o+2},{tc_} L{bx-2},{tc_}")

# ---------------- Phase 3 ----------------
P3L, P3R = 1536, 1962
rect(P3L, FT, P3R - P3L, FB - FT, "none", "#2b2b2b", 1.8, 6, "9,6")
text(P3L, FT - 14, "Phase 3 — Cross-sectional return prediction", T_PHASE, INK, "start", True)
c3 = (P3L + P3R) / 2
pw = P3R - P3L - 36
PH = 124
box(P3L + 18, PH, pw - 12, 76, "Prediction head", ["32 to 64 to 1, one ReLU"], stack=True)
path(f"M{c3-6},{PH+78} L{c3-6},{PH+120}")
PR = PH + 122
box(P3L + 18, PR, pw, 110, "Predicted return", ["one number per TW stock, day t+1", "ranked across the 50 stocks"])
path(f"M{c3},{PR+112} L{c3},{PR+152}")
TO = PR + 154
box(P3L + 18, TO, pw, 110, "Training objective", ["squared error, ranking loss,", "variance floor"], BLUE)
text(c3, TO + 150, "AdamW, walk-forward split, never shuffled", T_NOTE + 0.5, italic=True)
text(c3, TO + 172, "model chosen on validation IC", T_NOTE + 0.5, italic=True)
path(f"M{bx+bw+2},{GF+40} L{P3L-10},{GF+40} L{P3L-10},{PH+38} L{P3L+16},{PH+38}")

# ---------------- footnotes ----------------
fy = FB + 24
text(20, fy, "9 technical indicators computed, only log return fed to the model", T_NOTE, anchor="start", italic=True)
text(20, fy + 19, "US edge drawn iff |correlation| > 0.3, 60-day window ending at t-1", T_NOTE, anchor="start", italic=True)
rect(P2L, fy - 11, 18, 12, "#fbe4a8", "#b45309", 1.4, 0)
text(P2L + 26, fy, "new in this version", T_NOTE, anchor="start")
text(P2L, fy + 19, "heatmaps are real data — US graph from 2025-04-10, TW industry groups, trained coupling weights",
     T_NOTE, anchor="start", italic=True)

head = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'font-family="\'Helvetica Neue\', Arial, sans-serif">\n'
        f'<defs><marker id="tri" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto">'
        f'<path d="M0,0 L6.5,3 L0,6 Z" fill="{ARROW}"/></marker></defs>\n'
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>\n')
OUT.write_text(head + "\n".join(o) + "\n</svg>\n")
