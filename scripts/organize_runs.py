"""organize_runs.py — 把扁平的 runs/ 歸類成樹狀結構。

存在理由：runs/ 累積到 584 個目錄、1.7 GB，一眼看不出哪些屬於哪組實驗。

安全性：
  - 只搬目錄，不改任何 run 的內容
  - 所有分析腳本的 glob 已先改成 `runs/**/...`（對深度不敏感），
    且在搬動前驗證過輸出逐位相同
  - idempotent：已在樹裡的目錄跳過
  - 預設 dry-run，要加 --apply 才真的搬
  - INDEX.csv 與 comparison/ 留在 runs/ 根目錄不動

用法：
    .venv/bin/python scripts/organize_runs.py            # 只印計畫
    .venv/bin/python scripts/organize_runs.py --apply

注意：train.py 仍把新 run 寫在 runs/ 根目錄（扁平）。
所有分析腳本用的是 `runs/**/...` 遞迴 glob，兩種深度都吃得到，
所以不歸檔也不會壞；本腳本 idempotent，隨時可以再跑一次把新 run 收進樹裡。
"""
from __future__ import annotations
import argparse, re, shutil, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

# 順序有意義：由上而下第一個命中的規則決定分組
RULES: list[tuple[str, str]] = [
    (r"_(ridge|fvg)_",              "linear"),
    (r"_bipartite_",                "linear"),
    (r"_f2_",                       "fold2"),
    (r"_tw50_beta",                 "tw50/beta"),
    (r"_tw50_(inbn|plnorm)",        "tw50/inputnorm"),
    (r"_tw50_bl_",                  "tw50/baselines"),
    (r"_tw50_(T\d|L1_|efT|imed)",   "tw50/arch"),
    (r"_tw50(chk|traj)_",           "tw50/early"),
    (r"_tw50_smoke",                "tw50/early"),
]
KEEP_AT_ROOT = {"INDEX.csv", "comparison"}


def group_of(name: str) -> str:
    for pat, grp in RULES:
        if re.search(pat, name):
            return grp
    return "legacy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真的搬（預設只印計畫）")
    args = ap.parse_args()

    plan: list[tuple[Path, Path]] = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name in KEEP_AT_ROOT:
            continue
        if not re.match(r"^\d{8}_\d{4}_", d.name):      # 只動標準命名的 run
            continue
        dst = RUNS / group_of(d.name) / d.name
        if dst.exists():
            continue
        plan.append((d, dst))

    c = Counter(str(dst.parent.relative_to(RUNS)) for _, dst in plan)
    print(f"{'分組':22s} {'目錄數':>6s}")
    for g, n in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {g:20s} {n:>6d}")
    print(f"  {'合計':20s} {len(plan):>6d}")
    if not args.apply:
        print("\n（dry-run。加 --apply 才會搬）")
        return
    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"\n已搬移 {len(plan)} 個目錄")
    left = [p.name for p in RUNS.iterdir() if p.name not in KEEP_AT_ROOT
            and p.is_dir() and re.match(r"^\d{8}_\d{4}_", p.name)]
    print(f"runs/ 根目錄剩餘未分類的 run：{len(left)}")


if __name__ == "__main__":
    main()
