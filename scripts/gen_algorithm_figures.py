"""
gen_algorithm_figures.py — 方法章節的三張演算法圖（英文，對應現行主結果 SA）

    docs/figures/alg/alg1_phase1.svg   Algorithm 1  Phase 1 — Dual-market Encoding
    docs/figures/alg/alg2_phase2.svg   Algorithm 2  Phase 2 — Factor-exposure Coupling & Gated Fusion
    docs/figures/alg/alg3_phase3.svg   Algorithm 3  Phase 3 — Rank-oriented Prediction & Training

2026-11-20 依現行設計重畫（取代 2026-08 手寫的「兩級邊」版本）。內容逐項對過程式：
    Phase 1   src/models/encoders.py（BN_F、SharedLSTM、GATEncoder、TypeProjection）、
              src/models/multiplex_gnn.py 的 _tw_rgcn（同產業關係卷積，取代 GAT_L2）
    Phase 2   src/models/multiplex_gnn.py 的 _augment_weak（beta 層三項）、src/models/fusion.py（閘門）
    Phase 3   src/models/prediction_head.py（Head、CombinedLoss）、src/train/train.py（AdamW、排程、早停）
    超參數    runs/tw50/relational/20261023_2142_tw50_rgcnSA_s42/config_snapshot.yaml

文字標記：**粗體**＝關鍵字（直立）、_{…} 下標、^{…} 上標、`…` 直立字；其餘為斜體。註解（▷）畫在右欄。

    .venv/bin/python scripts/gen_algorithm_figures.py [--out DIR]
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

FONT = "'Times New Roman','Nimbus Roman',Georgia,serif"
SIZE, SUB, SUB2 = 15, 10, 8          # 本文、下標、下標裡的下標
COMMENT_COLOR = "#1f4e8c"
ROW = 24                             # 本文行距
HEAD_ROW = 23                        # Input / Output 行距


def _sub_digits(s: str) -> str:
    """上標 ^{…} 之外的 ₀₁₂ 換成 _{0} _{1} _{2}。"""
    out, i, level = [], 0, 0
    while i < len(s):
        if s.startswith("^{", i) and level == 0:
            k, d = i + 2, 1
            while d:
                d += {"{": 1, "}": -1}.get(s[k], 0)
                k += 1
            out.append(s[i:k])
            i = k
            continue
        c = s[i]
        out.append("_{" + "012"["₀₁₂".index(c)] + "}" if c in "₀₁₂" else c)
        i += 1
    return "".join(out)


def rich(s: str, depth: int = 0) -> str:
    """把簡單標記轉成 SVG tspan。

    Unicode 下標數字（₀₁₂）在本文裡換成 _{…}：它們在 Times 斜體裡像小寫字母。
    上標裡（例如 R^{n₁×d′}）保留 Unicode：上標裡再疊一層 baseline-shift，下標會掉回本文的基線。
    """
    if depth == 0:
        s = _sub_digits(s)
    out, buf, i = [], [], 0

    def flush():
        if buf:
            out.append(escape("".join(buf), quote=False))
            buf.clear()

    while i < len(s):
        if s.startswith("**", i):
            j = s.index("**", i + 2)
            flush()
            out.append(f'<tspan font-style="normal" font-weight="bold">{escape(s[i + 2:j], quote=False)}</tspan>')
            i = j + 2
        elif s[i] == "`":
            j = s.index("`", i + 1)
            flush()
            out.append(f'<tspan font-style="normal">{rich(s[i + 1:j], depth)}</tspan>')
            i = j + 1
        elif s[i] in "_^" and i + 1 < len(s) and s[i + 1] == "{":
            k, level = i + 2, 1
            while level:
                level += {"{": 1, "}": -1}.get(s[k], 0)
                k += 1
            flush()
            shift = "sub" if s[i] == "_" else "super"
            size = SUB if depth == 0 else SUB2
            out.append(f'<tspan baseline-shift="{shift}" font-size="{size}">{rich(s[i + 2:k - 1], depth + 1)}</tspan>')
            i = k
        else:
            buf.append(s[i])
            i += 1
    flush()
    return "".join(out)


def algorithm(path: Path, number: int, title: str, inputs: list[str], outputs: list[str],
              body: list[tuple[int, str, str]], width: int = 940, comment_x: int = 610) -> None:
    """body：(縮排層級, 本文, 註解)。

    comment_x 取該圖本文最長一行的右緣再加約 32，註解欄緊接在本文之後；
    width 取輸入列與註解欄右緣的較大者再加約 30，右側不留空白。右緣是在瀏覽器裡量 SVG 文字的 bbox 得到的。
    """
    x0, x1 = 20, width - 20
    y = 18
    parts = [f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#000" stroke-width="2"/>',
             f'<text x="24" y="44" font-size="17" font-weight="bold">Algorithm {number}</text>',
             f'<text x="140" y="44" font-size="17">{escape(title, quote=False)}</text>',
             f'<line x1="{x0}" y1="56" x2="{x1}" y2="56" stroke="#000" stroke-width="1"/>']
    y = 84
    for label, rows in (("Input:", inputs), ("Output:", outputs)):
        for k, r in enumerate(rows):
            if k == 0:
                parts.append(f'<text x="24" y="{y}" font-size="{SIZE}" font-weight="bold">{label}</text>')
            parts.append(f'<text x="98" y="{y}" font-size="{SIZE}" font-style="italic">{rich(r)}</text>')
            y += HEAD_ROW
        y += 5
    parts.append(f'<line x1="{x0}" y1="{y - 12}" x2="{x1}" y2="{y - 12}" stroke="#000" stroke-width="1"/>')
    y += 14
    for n, (indent, text, note) in enumerate(body, start=1):
        parts.append(f'<text x="50" y="{y}" font-size="{SIZE}" text-anchor="end">{n}:</text>')
        parts.append(f'<text x="{60 + 22 * indent}" y="{y}" font-size="{SIZE}" font-style="italic">{rich(text)}</text>')
        if note:
            parts.append(f'<text x="{comment_x}" y="{y}" font-size="13" fill="{COMMENT_COLOR}">'
                         f'&#9655; {rich(note)}</text>')
        y += ROW
    y -= 8
    parts.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#000" stroke-width="2"/>')
    height = y + 14
    svg = (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">\n'
           f'  <rect width="{width}" height="{height}" fill="#ffffff"/>\n  '
           + "\n  ".join(parts) + "\n</svg>\n")
    path.write_text(svg, encoding="utf-8")


def phase1(out: Path) -> None:
    algorithm(
        out / "alg1_phase1.svg", 1, "Phase 1 — Dual-market Encoding",
        inputs=[
            "US features X₁ ∈ R^{n₁×T×F},  TW features X₂ ∈ R^{n₂×T×F}  `(T = 1, F = 1: log return)`",
            "US graph A₁(t) = (E₁, ω₁):  |ρ_{ik}| > τ over the T_{ρ} days up to t − 1",
            "TW same-industry neighbours N(j), fixed  `(2018Q4 industry labels)`",
            "Shared LSTM;  GATv2 encoder GAT₁;  relational weights W₀, W_{s}, b;  projections P₁, P₂",
        ],
        outputs=["Latent states h₁ ∈ R^{n₁×d′},  h₂ ∈ R^{n₂×d′},  d′ = 32"],
        body=[
            (0, "**for** ℓ = 1 **to** 2 **do** X_{ℓ} ← BN_{F}(X_{ℓ}) **end for**", "feature-wise norm, shared statistics"),
            (0, "**for** ℓ = 1 **to** 2 **do**", ""),
            (1, "**for each** node i in layer ℓ **do**", ""),
            (2, "s_{ℓ,i} ← LSTM(X_{ℓ,i})[T]", "one LSTM for both markets"),
            (1, "**end for**", ""),
            (0, "**end for**", ""),
            (0, "z₁ ← GAT₁(s₁, E₁, ω₁)", "US: correlation graph"),
            (0, "**for each** TW node j = 1 **to** n₂ **do**", ""),
            (1, "z_{2j} ← W₀ s_{2j} + (1 / |N(j)|) Σ_{k∈N(j)} W_{s} s_{2k} + b",
             "TW: self + same-industry mean (R-GCN)"),
            (0, "**end for**", ""),
            (0, "h₁ ← LayerNorm(GELU(P₁ · z₁));   h₂ ← LayerNorm(GELU(P₂ · z₂))", "type-specific projection"),
            (0, "**return** h₁, h₂", ""),
        ], width=770, comment_x=500)


def phase2(out: Path) -> None:
    algorithm(
        out / "alg2_phase2.svg", 2, "Phase 2 — Factor-exposure Coupling & Gated Fusion",
        inputs=[
            "Latent states h₁ ∈ R^{n₁×d′},  h₂ ∈ R^{n₂×d′}",
            "Pairing p(·)  `(7 ADR pairs)`;  mask M ∈ {0,1}^{n₁×n₂},  M_{ij} = 0 iff i = p(j)",
            "Exposures α, γ ∈ R^{n₂};  residual coupling B ∈ R^{n₁×n₂};  gate params W_{g}, b_{g}",
            "Init:  B = 0,  α_{j} ~ N(1, 0.3²),  γ_{j} ~ N(0, 0.3²)",
        ],
        outputs=["Fused state h_{f} ∈ R^{n₂×d′};  readouts g, B_{eff}"],
        body=[
            (0, "B_{eff} ← B ⊙ M", ""),
            (0, "h̄₁ ← (1 / n₁) Σ_{i} h_{1i}", "US market state (cross-sectional mean)"),
            (0, "**for each** TW node j = 1 **to** n₂ **do**", ""),
            (1, "**if** p(j) exists **then** id_{j} ← α_{j} · h_{1,p(j)}", "① ADR identity"),
            (1, "**else** id_{j} ← 0", ""),
            (1, "**end if**", ""),
            (1, "f_{j} ← γ_{j} · h̄₁", "② market factor"),
            (1, "r_{j} ← Σ_{i} B_{eff}[i, j] · (h_{1i} − h̄₁)", "③ residual coupling"),
            (1, "ĥ_{1j} ← id_{j} + f_{j} + r_{j}", "US-side state for TW node j"),
            (0, "**end for**", ""),
            (0, "**for each** TW node j = 1 **to** n₂ **do**", ""),
            (1, "g_{j} ← σ(W_{g} [ĥ_{1j} ; h_{2j}] + b_{g})", "per-node, per-dimension gate"),
            (1, "h_{f,j} ← g_{j} ⊙ ĥ_{1j} + (1 − g_{j}) ⊙ h_{2j}", "fused state"),
            (0, "**end for**", ""),
            (0, "**return** h_{f}, (g, B_{eff})", ""),
        ], width=570, comment_x=310)


def phase3(out: Path) -> None:
    algorithm(
        out / "alg3_phase3.svg", 3, "Phase 3 — Cross-sectional return scoring",
        inputs=[
            "Training days D_{tr}, validation days D_{va}  `(walk-forward, no shuffle)`",
            "Params θ;  loss weights w_{r} = 1, w_{v} = 0.1;  patience P = 15;  E_{max} = 100",
            "Head: R^{d′} → 64 → 1, ReLU, dropout 0.2",
        ],
        outputs=["θ* selected at best validation IC"],
        body=[
            (0, "best ← −∞;  wait ← 0", ""),
            (0, "**for** epoch = 1 **to** E_{max} **do**", ""),
            (1, "**for each** day-batch D_{b} ⊂ D_{tr}, in time order **do**", "32 days per batch"),
            (2, "ŷ(t) ← Head(Phase2(Phase1(t))),  for all t ∈ D_{b}", ""),
            (2, "L_{MSE} ← mean_{t,j} (ŷ_{j}(t) − y_{j}(t))²", ""),
            (2, "L_{rank} ← mean over pairs y_{u}(t) > y_{v}(t) of softplus(−(ŷ_{u}(t) − ŷ_{v}(t)))", "RankNet"),
            (2, "L_{var} ← mean_{t} (std_{j} ŷ_{j}(t) − std_{j} y_{j}(t))²", "keeps the cross-sectional spread"),
            (2, "L ← L_{MSE} + w_{r} · L_{rank} + w_{v} · L_{var}", ""),
            (2, "θ ← AdamW step on ∇_{θ} L", "lr 10⁻³, weight decay 10⁻³, clip 1"),
            (1, "**end for**", ""),
            (1, "IC_{va} ← mean_{t∈D_{va}} corr_{j}(ŷ_{j}(t), y_{j}(t))", ""),
            (1, "**if** IC_{va} > best **then** best ← IC_{va};  θ* ← θ;  wait ← 0", ""),
            (1, "**else** wait ← wait + 1", ""),
            (1, "**end if**", ""),
            (1, "**if** wait ≥ P **then break**", "early stop"),
            (1, "update learning rate", "cosine warm restarts, T₀ = 10, ×2"),
            (0, "**end for**", ""),
            (0, "**return** θ*", ""),
        ], width=740, comment_x=520)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/alg")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (phase1, phase2, phase3):
        f(out)
    print(f"寫入 {out}")


if __name__ == "__main__":
    main()
