"""
e7_acceptance.py — E7：universe 擴充重構後的完整回歸驗收

為什麼不能只跑 freeze_k7.py --verify：
    那支腳本是拿磁碟上既有的 predictions CSV 去對雜湊，證明的是
    「檔案沒被改動」。E3-E6 改的是產生那些檔案的程式碼，所以必須
    **用重構後的程式碼重新產生一次**，再跟凍結基準比對。

做法：
    對 docs/frozen_k7_manifest.json 記錄的 17 個 run，逐一用它自己的
    config_snapshot.yaml 重跑，再依路徑類型比對：

      位元確定（純 LSTM，10 個）：predictions CSV 的 SHA-256 完全相同
      隨機（GNN scatter，7 個）  ：|Δ test_IC| <= 3 sigma_run = 0.081

    config_snapshot.yaml 是 E5 之前存的，沒有 data.universe 欄位，
    會走 DEFAULT_UNIVERSE=k7 的預設路徑——這正是要驗的向後相容性。

tag 命名：
    重跑的 tag 刻意去掉開頭的 "opt_"，因為 freeze_k7.pred_path() 用
    glob(f"*{tag}") 找目錄並取 hits[-1]（最新）。若重跑目錄名含有原
    tag 字串，之後的 --verify 會改去對重跑而非凍結基準，而且不會有
    任何警告。腳本會在每次重跑後斷言沒有發生這種汙染。

用法：
    .venv/bin/python scripts/e7_acceptance.py            # 全部 17 個
    .venv/bin/python scripts/e7_acceptance.py --only ef  # 只跑 tag 含 ef 的
    .venv/bin/python scripts/e7_acceptance.py --dry-run  # 只列出計畫
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_k7 import (  # noqa: E402
    DETERMINISTIC_RUNS,
    MANIFEST,
    SIGMA_RUN,
    STOCHASTIC_RUNS,
    mean_ic,
    pred_path,
    sha256,
)

RERUN_PREFIX = "e7v_"


def rerun_tag(tag: str) -> str:
    """重跑用的 tag。去掉 opt_ 前綴，避免汙染 freeze_k7 的 glob。"""
    return RERUN_PREFIX + tag.removeprefix("opt_")


def config_snapshot(tag: str) -> Path:
    hits = sorted(ROOT.glob(f"runs/**/*{tag}/config_snapshot.yaml"))
    if not hits:
        raise FileNotFoundError(f"找不到 {tag} 的 config_snapshot.yaml")
    return hits[-1]


def frozen_dir_count(tag: str) -> int:
    return len(sorted(ROOT.glob(f"runs/**/*{tag}")))


def run_one(tag: str) -> Path:
    """重跑一個 run，回傳新產出的 predictions CSV 路徑。"""
    new_tag = rerun_tag(tag)
    before = frozen_dir_count(tag)

    subprocess.run(
        [sys.executable, "-m", "src.train.train",
         "--config", str(config_snapshot(tag)), "--tag", new_tag],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )

    # 汙染檢查：重跑不得讓 pred_path(原 tag) 指到別的目錄
    after = frozen_dir_count(tag)
    if after != before:
        raise RuntimeError(
            f"重跑 tag '{new_tag}' 汙染了 freeze_k7 對 '{tag}' 的 glob "
            f"（{before} → {after} 個目錄）。之後的 --verify 會對到重跑結果。"
        )

    hits = sorted(ROOT.glob(f"runs/**/*{new_tag}/predictions/test_predictions.csv"))
    if not hits:
        raise FileNotFoundError(f"重跑 {new_tag} 未產出 predictions CSV")
    return hits[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="E7 完整回歸驗收")
    ap.add_argument("--only", default=None, help="只跑 tag 含此字串的 run")
    ap.add_argument("--dry-run", action="store_true", help="只列出計畫不重跑")
    args = ap.parse_args()

    m = json.loads(Path(MANIFEST).read_text())
    det = [t for t in DETERMINISTIC_RUNS if not args.only or args.only in t]
    sto = [t for t in STOCHASTIC_RUNS   if not args.only or args.only in t]

    print(f"[E7] manifest {m['created']}  git_ref={m['git_ref'][:8]}")
    print(f"[E7] universe n_l1={m['universe']['n_l1']} n_l2={m['universe']['n_l2']} "
          f"pairing_rate={m['universe']['pairing_rate']}")
    print(f"[E7] 重跑 {len(det)} 個位元確定 + {len(sto)} 個隨機 run")

    if args.dry_run:
        for t in det + sto:
            print(f"  {t:26s} → {rerun_tag(t):24s} cfg={config_snapshot(t).parent.name}")
        return

    fails = 0

    print("\n-- 位元確定路徑（SHA-256 必須完全相同）--")
    for tag in det:
        p = run_one(tag)
        got = sha256(p)
        exp = m["deterministic_runs"][tag]["sha256"]
        ok = got == exp
        fails += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'FAIL'}  {tag:26s} {exp[:16]} vs {got[:16]}")

    print(f"\n-- 隨機路徑（|Δ test_IC| <= 3 sigma_run = {3*SIGMA_RUN:.3f}）--")
    for tag in sto:
        p = run_one(tag)
        got = mean_ic(p)
        exp = m["stochastic_runs"][tag]["test_IC"]
        d = abs(got - exp)
        ok = d <= 3 * SIGMA_RUN
        fails += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'FAIL'}  {tag:26s} {exp:+.4f} vs {got:+.4f}  (|d|={d:.4f})")

    # 重跑完之後凍結基準本身必須仍然完好
    print("\n-- 凍結基準檔案未被動到 --")
    stale = 0
    for tag, rec in m["deterministic_runs"].items():
        if sha256(pred_path(tag)) != rec["sha256"]:
            print(f"  FAIL  {tag} 的凍結 CSV 已變動")
            stale += 1
    print(f"  {'OK  ' if stale == 0 else 'FAIL'}  10 個凍結 CSV 逐一比對")
    fails += stale

    print(f"\n[E7] {'全部通過' if fails == 0 else f'{fails} 項未通過'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
