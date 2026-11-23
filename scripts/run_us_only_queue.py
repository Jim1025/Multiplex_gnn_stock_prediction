"""
run_us_only_queue.py — 評估議題 2：MAGNET「只用美股」的訓練佇列（第一折）

登記：docs/proposal_1209_revision.md §7 附錄（訓練前寫下）。主結果 SA 的指令（proposal_819 §79.9）只多給 --us-only
（融合前把台股表示 h_L2 歸零，h_fused = gate ⊙ ĥ_1；架構、超參數、損失、早停、種子都不動）。

    tag tw50_rgcnSAusonly_s{S}，種子與 SA 相同的 10 顆

階段：
    check   只跑 s42，跑完把它的 config_snapshot.yaml 與 SA s42 的比對：只能多出 model.us_only: true，否則結束碼 1
    rest    其餘 9 顆（s42 已完成就略過），兩條佇列並行

每個 run 只跑一次；執行邏輯（已完成略過、失敗只重跑一次、不存紀錄檔）沿用 run_input_lag_queue.py。
長時間執行時用 caffeinate 包起來，避免系統閒置睡眠（螢幕可以關）：

    caffeinate -i -m -s sh -c '.venv/bin/python scripts/run_us_only_queue.py --stage check \
        && .venv/bin/python scripts/run_us_only_queue.py --stage rest --workers 2'
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_input_lag_queue import MAGNET_CFG, MAGNET_FLAGS, MAGNET_SEEDS, PY, finished_run, record, run_one  # noqa: E402

SA_S42 = ROOT / "runs" / "tw50" / "relational" / "20261023_2142_tw50_rgcnSA_s42" / "config_snapshot.yaml"


def job(seed: int) -> dict:
    tag = f"tw50_rgcnSAusonly_s{seed}"
    cmd = [PY, "-m", "src.train.train", "--config", MAGNET_CFG, "--seed", str(seed), "--tag", tag,
           *MAGNET_FLAGS, "--us-only"]
    return dict(model="magnet", seed=seed, tag=tag, cmd=cmd)


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def config_diff(run_dir: Path) -> dict:
    """run 的 config_snapshot 與 SA s42 的差：{鍵: (SA, 本 run)}。"""
    a = _flat(yaml.safe_load(SA_S42.read_text()))
    b = _flat(yaml.safe_load((run_dir / "config_snapshot.yaml").read_text()))
    return {k: (a.get(k, "<無>"), b.get(k, "<無>")) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def run_queue(jobs: list[dict], workers: int) -> None:
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker() -> None:
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            try:
                run_one(j)
            except Exception as e:                      # 佇列本身不因單一 run 的意外而停
                record(tag=j["tag"], event="error", error=repr(e))

    ts = [threading.Thread(target=worker) for _ in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "rest"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="只列出指令")
    args = ap.parse_args()

    jobs = [job(42)] if args.stage == "check" else [job(s) for s in MAGNET_SEEDS if s != 42]
    if args.dry_run:
        for j in jobs:
            print(("[已完成] " if finished_run(j["tag"]) else "") + " ".join(j["cmd"]))
        return
    record(event="queue_start", stage=args.stage, n_jobs=len(jobs), workers=args.workers)
    run_queue(jobs, 1 if args.stage == "check" else args.workers)

    missing = [j["tag"] for j in jobs if finished_run(j["tag"]) is None]
    if args.stage == "check" and not missing:
        diff = config_diff(finished_run("tw50_rgcnSAusonly_s42"))
        ok = diff == {"model.us_only": ("<無>", True)}
        record(event="config_check", ok=ok, diff=diff)
        if not ok:
            record(event="queue_end", stage=args.stage, reason="設定與 SA s42 的差不只 model.us_only，停止")
            sys.exit(1)
    record(event="queue_end", stage=args.stage, missing=missing)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
