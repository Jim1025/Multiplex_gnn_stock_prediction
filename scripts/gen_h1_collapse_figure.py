"""
gen_h1_collapse_figure.py — 給教授的說明圖：h₁ᵢ 為什麼會塌縮，以及後果。

**這不是架構圖**，所以不套 `gen_magnet_figure.py` 的房屋風格（那份的規則是
「只畫架構、不放讀數」）。這張圖的論證本身就是那些讀數。

一條因果鏈，由左至右，**全部在同一條前向傳遞上、用整個 test split（246 天）
量的**，舊 arm 訓練後：

    原始特徵 -> BatchNorm -> LSTM -> GAT -> 投影層+bias = h₁ᵢ
     1.0000     0.3287     0.6533  0.9815      0.9956

扇形的**張角是按實測餘弦換算的，不是示意**：
兩兩餘弦 rho 對應繞平均向量的半角 arccos(sqrt((1+rho)/2))。

三站各自的主因（都是量出來的，不是推測）：
  LSTM   bias。全部歸零 -> 0.6533 掉回 0.3252，等於它的輸入值
  GAT    鄰居平均。有效鄰居 19.8，共同成分權重和為 1 不動、偏差縮 1/sqrt(k)，
         預測比值放大 sqrt(19.8)=4.45x，實測 4.28x（差 3.8%）
  投影層  Linear 的 bias。||b|| 是橫截面偏差的 4.53 倍（§59）

**增量會低估後面的站**：三個來源是冗餘的，前面把餘弦推到接近 1 之後，
後面就沒有空間再推。投影層在完整鏈上只 +0.0142，但在「無鄰居」的隔離
量測下是 +0.0781（訓練後）／+0.2192（初始化）。

    .venv/bin/python scripts/gen_h1_collapse_figure.py
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "h1_collapse.svg"

# (標題, 維度, 餘弦, 主因標題, 主因說明)
STAGES = [
    ("原始特徵", 3, 1.0000, "", "模型吃的 3 欄未標準化；\nRSI_14 的尺度主導範數"),
    ("BatchNorm 後", 3, 0.3287, "唯一拉低的一站",
     "逐特徵跨節點對齊尺度；\n這才是 LSTM 的實際輸入"),
    ("LSTM 之後", 64, 0.6533, "主因：LSTM 的 bias",
     "bias 全部歸零 -> 0.3252，\n等於掉回它的輸入值"),
    ("GAT 之後", 64, 0.9815, "主因：鄰居平均",
     "有效鄰居 19.8。共同成分不動、\n偏差縮 1/sqrt(k)，實測 4.28x"),
    ("投影層 + bias（= h₁ᵢ）", 32, 0.9956, "主因：Linear 的 bias",
     "每個節點加同一個 b，\n而 ||b|| 是橫截面偏差的 4.53 倍"),
]
BAR_MAX = 94.0
N_ARROWS = 30
W, H = 1840, 1000
INK, MUTE, HL, GOOD, BG = "#1a1a1a", "#8a8a8a", "#b45309", "#15803d", "#ffffff"


def half_angle(rho: float) -> float:
    return math.degrees(math.acos(min(1.0, math.sqrt((1.0 + rho) / 2.0))))


def fan(cx: float, cy: float, rho: float, r: float = 100.0) -> str:
    ha = half_angle(rho)
    out = []
    for k in range(N_ARROWS):
        f = -1.0 + 2.0 * k / (N_ARROWS - 1)
        a = math.radians(-90.0 + f * ha)
        out.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" '
                   f'x2="{cx + r * math.cos(a):.1f}" y2="{cy + r * math.sin(a):.1f}" '
                   f'stroke="{INK}" stroke-width="1.0" marker-end="url(#a)"/>')
    return "\n".join(out)


def bars(x0: float, y0: float, equal: bool) -> str:
    import random
    random.seed(0)
    out, w, gap = [], 7.0, 3.0
    for k in range(30):
        h = min(58.0 + (random.uniform(-1.2, 1.2) if equal
                        else random.uniform(-34, 34)), BAR_MAX)
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

    p.append(f'<text x="56" y="58" font-size="28" font-weight="600" fill="{INK}">'
             '30 檔美股的表示 h₁ᵢ 如何一路被壓成同一個向量，以及它的後果</text>')
    p.append(f'<text x="56" y="90" font-size="15.5" fill="{MUTE}">'
             '一根箭頭 = 一檔美股，共 30 根。扇形張角按實測兩兩餘弦換算，非示意。'
             '整條鏈在同一次前向傳遞上、用整個 test split（246 天）量，舊寫法訓練後。</text>')
    p.append(f'<text x="56" y="114" font-size="15.5" fill="{MUTE}">'
             '各站維度不同（3 / 3 / 64 / 64 / 32），但餘弦量的是'
             '「這 30 個向量彼此多對齊」，是各空間內部的正規化內積，跨站可比。</text>')

    base_y, x0, dx = 330, 176, 352
    for i, (title, dim, rho, cause, note) in enumerate(STAGES):
        cx = x0 + i * dx
        p.append(fan(cx, base_y, rho))
        p.append(f'<text x="{cx}" y="{base_y + 36}" font-size="18.5" font-weight="600" '
                 f'text-anchor="middle" fill="{INK}">{title}</text>')
        p.append(f'<text x="{cx}" y="{base_y + 60}" font-size="14.5" '
                 f'text-anchor="middle" fill="{INK}">30 檔 x {dim} 維</text>')
        col = HL if i >= 2 else (GOOD if i == 1 else INK)
        p.append(f'<text x="{cx}" y="{base_y + 96}" font-size="22" font-weight="700" '
                 f'text-anchor="middle" fill="{col}">兩兩餘弦 {rho:.4f}</text>')
        p.append(f'<text x="{cx}" y="{base_y + 120}" font-size="13" '
                 f'text-anchor="middle" fill="{MUTE}">'
                 f'（張角 ±{half_angle(rho):.0f}°）</text>')
        if cause:
            c2 = GOOD if i == 1 else HL
            p.append(f'<text x="{cx}" y="{base_y + 158}" font-size="16" '
                     f'font-weight="700" text-anchor="middle" fill="{c2}">{cause}</text>')
        for j, ln in enumerate(note.split("\n")):
            p.append(f'<text x="{cx}" y="{base_y + 184 + j * 21}" font-size="13.5" '
                     f'text-anchor="middle" fill="{MUTE}">{ln}</text>')
        if i:
            d = rho - STAGES[i - 1][2]
            xa, xb = cx - dx + 118, cx - 118
            p.append(f'<line x1="{xa}" y1="{base_y - 52}" x2="{xb}" y2="{base_y - 52}" '
                     f'stroke="{MUTE}" stroke-width="1.4" marker-end="url(#a)"/>')
            p.append(f'<text x="{(xa + xb) / 2:.0f}" y="{base_y - 62}" font-size="15" '
                     f'font-weight="700" text-anchor="middle" '
                     f'fill="{GOOD if d < 0 else HL}">{d:+.4f}</text>')

    # 後果
    yb = 660
    p.append(f'<line x1="56" y1="{yb - 34}" x2="{W - 56}" y2="{yb - 34}" '
             f'stroke="#e0e0e0" stroke-width="1.4"/>')
    p.append(f'<text x="56" y="{yb}" font-size="20" font-weight="600" fill="{INK}">'
             '後果：B 這一欄的 30 個偏導數  ∂L/∂B[i,j] = ⟨g_j , h₁ᵢ⟩</text>')
    p.append(f'<text x="56" y="{yb + 34}" font-size="14.5" fill="{MUTE}">'
             '假如 h₁ᵢ 彼此不同 -&gt; 偏導數不同，可以造出個股結構：</text>')
    p.append(bars(56, yb + 152, equal=False))
    p.append(f'<text x="640" y="{yb + 34}" font-size="14.5" font-weight="600" fill="{INK}">'
             '實際上 h₁ᵢ 幾乎相同 -&gt; 偏導數幾乎相同：</text>')
    p.append(bars(640, yb + 152, equal=True))
    p.append(f'<text x="640" y="{yb + 186}" font-size="15.5" font-weight="700" fill="{HL}">'
             '整欄只能一起推 = 只改變總和 = 只改市場曝險</text>')
    p.append(f'<text x="640" y="{yb + 212}" font-size="14" fill="{MUTE}">'
             '梯度 99.99% 在推市場曝險，只有 0.01% 能造出個股結構</text>')
    p.append(f'<text x="1240" y="{yb + 34}" font-size="14.5" fill="{MUTE}">'
             '而市場曝險網路別處已經能表達，</text>')
    p.append(f'<text x="1240" y="{yb + 58}" font-size="14.5" fill="{MUTE}">'
             '對 7 檔配對股還會跟權重固定為 1 的</text>')
    p.append(f'<text x="1240" y="{yb + 82}" font-size="14.5" fill="{MUTE}">'
             '恆等邊搶同一個方向——</text>')
    p.append(f'<text x="1240" y="{yb + 112}" font-size="15.5" font-weight="700" fill="{HL}">'
             '它唯一推得動的方向是多餘的。</text>')

    y = H - 52
    p.append(f'<line x1="56" y1="{y - 40}" x2="{W - 56}" y2="{y - 40}" '
             f'stroke="#e0e0e0" stroke-width="1.4"/>')
    p.append(f'<text x="56" y="{y}" font-size="17" fill="{INK}">'
             '結果：舊寫法的 |B|max 中位數停在 <tspan font-weight="700">0.0005</tspan>'
             '（幾乎沒離開 0 初始值），新寫法 <tspan font-weight="700">0.41</tspan>；'
             '同樣 1,493 條邊全部拿掉，舊寫法 ΔIC '
             '<tspan font-weight="700">+0.0000</tspan>，新寫法 '
             '<tspan font-weight="700">−0.0269</tspan>。</text>')
    p.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(p), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
