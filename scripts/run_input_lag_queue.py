"""
run_input_lag_queue.py — 評估議題 3：MAGNET 與 MEIG 的美股輸入延遲版訓練佇列（第一折）

登記：docs/proposal_1209_revision.md §2.8 附錄（訓練前寫下）。模型、超參數、損失、早停都不動，
只多給 --us-lag l（美股層輸入落後 l 個美股交易日；圖、台股輸入與標籤不變）。

    MAGNET（主結果 SA 的指令，proposal_819 §79.9）  tag tw50_rgcnSAusl{l}_s{S}     10 顆種子
    MEIG [1]（主表 run 的 config_snapshot）         tag tw50_bl_meigusl{l}_s{S}    3 顆種子

階段：
    check   落後 1 天的重現（MAGNET s42、MEIG s42 各一個，tag …usl1chk_s42），確認目前環境與原 run 可比
    lag     l = 2、3、4 的全部 run（MAGNET 30 個、MEIG 9 個）
    lag1    備用：check 超出登記的容許範圍時，落後 1 天全部種子在目前環境重訓（tag …usl1_s{S}）

每個 run 只跑一次：已有 FINISHED 的同 tag run 就略過（佇列中斷後可直接重新執行本檔接續）；
非零結束碼或逾時只重跑一次。進度只印在終端機，不另存紀錄檔；run 失敗時印出它輸出的最後 40 行。
長時間執行時用 caffeinate 包起來，避免系統閒置睡眠（螢幕可以關）：

    caffeinate -i -m -s .venv/bin/python scripts/run_input_lag_queue.py --stage lag --workers 2
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")

MAGNET_CFG = "runs/tw50/beta/20260829_0629_tw50_betaF1nA2r1_s42/config_snapshot.yaml"
MAGNET_FLAGS = ["--tw-relational", "--tw-relational-placement", "replace_gat_l2",
                "--tw-relational-blocks", "4", "--tw-relational-relations", "same_industry"]
MAGNET_SEEDS = [42, 7, 123, 11, 99, 314, 555, 777, 888, 2026]
MEIG_CFG = "runs/tw50/baselines/20260814_1549_tw50_bl_meig_s42/config_snapshot.yaml"
MEIG_SEEDS = [42, 7, 123]
LAGS = [2, 3, 4]
TIMEOUT_S = 90 * 60        # 單一 run 的上限；平常 MAGNET 約 4–20 分鐘、MEIG 約 2–5 分鐘

_lock = threading.Lock()


def job(model: str, lag: int, seed: int, suffix: str | None = None) -> dict:
    name = suffix or f"usl{lag}"
    if model == "magnet":
        tag = f"tw50_rgcnSA{name}_s{seed}"
        cmd = [PY, "-m", "src.train.train", "--config", MAGNET_CFG, "--seed", str(seed), "--tag", tag,
               *MAGNET_FLAGS, "--us-lag", str(lag)]
    else:
        tag = f"tw50_bl_meig{name}_s{seed}"
        cmd = [PY, "-m", "src.train.train", "--config", MEIG_CFG, "--seed", str(seed), "--tag", tag,
               "--us-lag", str(lag)]
    return dict(model=model, lag=lag, seed=seed, tag=tag, cmd=cmd)


def jobs_for(stage: str) -> list[dict]:
    if stage == "check":
        return [job("magnet", 1, 42, "usl1chk"), job("meig", 1, 42, "usl1chk")]
    if stage == "lag":
        # 種子在外層：中途停下來時三個延遲的種子數相同
        out = [job("magnet", l, s) for s in MAGNET_SEEDS for l in LAGS]
        return out + [job("meig", l, s) for s in MEIG_SEEDS for l in LAGS]
    if stage == "lag1-magnet":
        return [job("magnet", 1, s) for s in MAGNET_SEEDS]
    if stage == "lag1-meig":
        return [job("meig", 1, s) for s in MEIG_SEEDS]
    raise ValueError(stage)


def finished_run(tag: str) -> Path | None:
    """已有同 tag 且 status = FINISHED 的 run 目錄就回傳（取最新一個）。"""
    hits = []
    for d in (ROOT / "runs").rglob(f"*_{tag}"):
        m = d / "meta.json"
        if d.is_dir() and m.exists():
            try:
                meta = json.loads(m.read_text())
            except json.JSONDecodeError:
                continue
            if meta.get("tag") == tag and meta.get("status") == "FINISHED":
                hits.append(d)
    return sorted(hits)[-1] if hits else None


def record(**row) -> None:
    row["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        print(f"[{row['time']}] {row.get('tag', '')} {row.get('event', '')} "
              + " ".join(f"{k}={v}" for k, v in row.items() if k not in ("time", "tag", "event", "cmd")),
              flush=True)


def run_one(j: dict) -> None:
    tag = j["tag"]
    done = finished_run(tag)
    if done is not None:
        record(tag=tag, event="skip", reason="已有 FINISHED 的 run", run=str(done.relative_to(ROOT)))
        return
    for attempt in (1, 2):
        record(tag=tag, event="start", attempt=attempt, cmd=" ".join(j["cmd"]))
        t0 = time.time()
        with tempfile.TemporaryFile("w+", errors="replace") as f:      # 訓練輸出只暫存，結束就丟
            try:
                rc = subprocess.run(j["cmd"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            f.seek(0)
            tail = "".join(f.readlines()[-40:])
        mins = round((time.time() - t0) / 60, 1)
        done = finished_run(tag)
        if rc == 0 and done is not None:
            tm = json.loads((done / "meta.json").read_text()).get("test_metrics") or {}
            record(tag=tag, event="done", attempt=attempt, minutes=mins, run=str(done.relative_to(ROOT)),
                   IC=round(tm.get("IC", float("nan")), 4), RankIC=round(tm.get("RankIC", float("nan")), 4))
            return
        record(tag=tag, event="fail", attempt=attempt, minutes=mins, returncode=rc)
        with _lock:
            print(f"---- {tag} 輸出的最後 40 行 ----\n{tail}----", flush=True)
    record(tag=tag, event="give_up")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "lag", "lag1-magnet", "lag1-meig"])
    ap.add_argument("--workers", type=int, default=2, help="並行的佇列數（預設 2）")
    ap.add_argument("--dry-run", action="store_true", help="只列出指令")
    args = ap.parse_args()

    jobs = jobs_for(args.stage)
    if args.dry_run:
        for j in jobs:
            print(("[已完成] " if finished_run(j["tag"]) else "") + " ".join(j["cmd"]))
        return
    record(event="queue_start", stage=args.stage, n_jobs=len(jobs), workers=args.workers)
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

    ts = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    missing = [j["tag"] for j in jobs if finished_run(j["tag"]) is None]
    record(event="queue_end", stage=args.stage, missing=missing)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
