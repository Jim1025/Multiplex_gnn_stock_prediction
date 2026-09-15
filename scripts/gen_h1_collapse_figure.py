"""
gen_h1_collapse_figure.py — 給教授的說明圖：h₁ᵢ 為什麼會塌縮，以及後果。

**這不是架構圖**，所以刻意不套 `gen_magnet_figure.py` 的房屋風格
（那份的規則是「只畫架構、不放讀數」）。這張圖的論證本身就是那些讀數，
拿掉讀數就沒有內容了。

畫的是一條因果鏈，由左至右讀：

    原始特徵 -> LSTM -> GAT 鄰居平均 -> 投影層 +bias -> h₁ᵢ -> B 的梯度

扇形的**張角是按實測餘弦換算的，不是示意**：
兩兩餘弦 rho 對應到繞平均向量的半角 arccos(sqrt((1+rho)/2))。

數字來源：`scripts/coupling_geometry.py` 區塊 B / C / I（舊 arm、已訓練）。

    .venv/bin/python scripts/gen_h1_collapse_figure.py
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "h1_collapse.svg"

# (標題, 副標, 維度, 實測兩兩餘弦)  區塊 B / C
#
# 四站的**維度不同**（9 / 64 / 64 / 32），但餘弦是每個空間**內部**的正規化
# 內積，量的是「這 30 個向量彼此多對齊」，所以跨站比較合法。
# 原始特徵那站資料層是 9 欄，模型（F3 arm）實際只吃 3 欄，但餘弦幾乎一樣
# （9 欄 0.2945 / 3 欄 0.2954 / 1 欄 0.3112），結論不受影響。
STAGES = [
    ("原始特徵", "逐特徵標準化後", 9, 0.2945),
    ("LSTM 之後", "時序編碼", 64, 0.6533),
    ("GAT 之後", "圖密度 65%，每檔對 18.8 檔取加權平均", 64, 0.9815),
    ("投影層 + bias 之後（= h₁ᵢ）", "bias 比橫截面差異大 4.3 倍", 32, 0.9956),
]
BAR_MAX = 94.0         # 柱高上限，碰撞檢查靠它
N_ARROWS = 30          # 30 檔美股，一檔一根（後兩站糊成一片——那正是重點）
W, H = 1760, 720
INK, MUTE, HL, BG = "#1a1a1a", "#8a8a8a", "#b45309", "#ffffff"


def half_angle(rho: float) -> float:
    """兩兩餘弦 rho -> 繞平均向量的半角（度）。"""
    return math.degrees(math.acos(min(1.0, math.sqrt((1.0 + rho) / 2.0))))


def fan(cx: float, cy: float, rho: float, r: float = 100.0) -> str:
    """從 (cx, cy) 往上畫一束箭頭，張角由 rho 決定。"""
    ha = half_angle(rho)
    out = []
    for k in range(N_ARROWS):
        f = -1.0 + 2.0 * k / (N_ARROWS - 1)          # -1 .. +1
        a = math.radians(-90.0 + f * ha)
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        out.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
                   f'stroke="{INK}" stroke-width="1.0" marker-end="url(#a)"/>')
    return "\n".join(out)


def bars(x0: float, y0: float, equal: bool) -> str:
    """30 根偏導數。equal=True 代表幾乎同高（= 只能推總和）。"""
    import random
    random.seed(0)
    out, w, gap = [], 7.0, 3.0
    for k in range(20):
        h = 58.0 + (random.uniform(-1.2, 1.2) if equal else random.uniform(-34, 34))
        h = min(h, BAR_MAX)
        out.append(f'<rect x="{x0 + k * (w + gap):.1f}" y="{y0 - h:.1f}" '
                   f'width="{w}" height="{h:.1f}" fill="{HL if equal else MUTE}"/>')
    return "\n".join(out)


def main() -> None:
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Helvetica Neue, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>',
         '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
         'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
         f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker></defs>']

    p.append(f'<text x="56" y="56" font-size="27" font-weight="600" fill="{INK}">'
             '30 檔美股的表示 h₁ᵢ 如何一路被壓成同一個向量，以及它的後果</text>')
    p.append(f'<text x="56" y="88" font-size="16" fill="{MUTE}">'
             '一根箭頭 = 一檔美股，共 30 根。扇形張角按實測兩兩餘弦換算，非示意。'
             '四站維度不同（9 / 64 / 64 / 32），但餘弦量的是「這 30 個向量彼此多對齊」，'
             '跨站可比。數字為已訓練的舊寫法 arm。</text>')

    base_y, x0, dx = 330, 170, 268
    for i, (title, sub, dim, rho) in enumerate(STAGES):
        cx = x0 + i * dx
        p.append(fan(cx, base_y, rho))
        p.append(f'<text x="{cx}" y="{base_y + 34}" font-size="18" font-weight="600" '
                 f'text-anchor="middle" fill="{INK}">{title}</text>')
        p.append(f'<text x="{cx}" y="{base_y + 60}" font-size="15" font-weight="600" '
                 f'text-anchor="middle" fill="{INK}">30 檔 x {dim} 維</text>')
        p.append(f'<text x="{cx}" y="{base_y + 84}" font-size="13" '
                 f'text-anchor="middle" fill="{MUTE}">{sub}</text>')
        col = HL if i >= 2 else INK
        p.append(f'<text x="{cx}" y="{base_y + 118}" font-size="21" font-weight="700" '
                 f'text-anchor="middle" fill="{col}">兩兩餘弦 {rho:.4f}</text>')
        p.append(f'<text x="{cx}" y="{base_y + 142}" font-size="13" '
                 f'text-anchor="middle" fill="{MUTE}">'
                 f'（張角 ±{half_angle(rho):.0f}°）</text>')
        if i:
            xa, xb = cx - dx + 118, cx - 118
            p.append(f'<line x1="{xa}" y1="{base_y - 46}" x2="{xb}" y2="{base_y - 46}" '
                     f'stroke="{MUTE}" stroke-width="1.4" marker-end="url(#a)"/>')

    # 右段：後果
    gx = x0 + 3 * dx + 214
    p.append(f'<line x1="{gx - 84}" y1="100" x2="{gx - 84}" y2="{H - 90}" '
             f'stroke="#e0e0e0" stroke-width="1.4"/>')
    p.append(f'<text x="{gx}" y="168" font-size="19" font-weight="600" fill="{INK}">'
             '後果：B 的 30 個偏導數</text>')
    p.append(f'<text x="{gx}" y="196" font-size="14" fill="{MUTE}">'
             '∂L/∂B[i,j] = ⟨g_j , h₁ᵢ⟩</text>')
    # 柱狀圖由基線往上長 BAR_MAX，所以標題要留在基線上方 BAR_MAX + 行高之外
    p.append(f'<text x="{gx}" y="250" font-size="14" fill="{MUTE}">'
             '假如 h₁ᵢ 彼此不同 -&gt; 偏導數不同：</text>')
    p.append(bars(gx, 356, equal=False))
    p.append(f'<text x="{gx}" y="380" font-size="13.5" fill="{MUTE}">'
             '可以造出個股結構</text>')
    p.append(f'<text x="{gx}" y="440" font-size="14" fill="{INK}">'
             '實際上 h₁ᵢ 幾乎相同 -&gt; 偏導數幾乎相同：</text>')
    p.append(bars(gx, 546, equal=True))
    p.append(f'<text x="{gx}" y="572" font-size="14" font-weight="600" fill="{HL}">'
             '整欄只能一起推 = 只改變總和</text>')
    p.append(f'<text x="{gx}" y="596" font-size="13.5" fill="{MUTE}">'
             '梯度 99.99% 在推市場曝險，0.01% 能造個股結構</text>')

    # 底部一行結論
    y = H - 54
    p.append(f'<line x1="56" y1="{y - 42}" x2="{W - 56}" y2="{y - 42}" '
             f'stroke="#e0e0e0" stroke-width="1.4"/>')
    p.append(f'<text x="56" y="{y}" font-size="17" fill="{INK}">'
             '結果：舊寫法的 |B|max 中位數停在 <tspan font-weight="700">0.0005</tspan>'
             '（幾乎沒離開 0 初始值），新寫法 <tspan font-weight="700">0.41</tspan>；'
             '同樣 1,493 條邊全部拿掉，舊寫法 ΔIC '
             '<tspan font-weight="700">+0.0000</tspan>，'
             '新寫法 <tspan font-weight="700">−0.0269</tspan>。</text>')
    p.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(p), encoding="utf-8")
    print(f"-> {OUT}")
    for _, _, _, r in STAGES:
        print(f"   rho {r:.4f} -> 半角 {half_angle(r):5.1f}°")


if __name__ == "__main__":
    main()
