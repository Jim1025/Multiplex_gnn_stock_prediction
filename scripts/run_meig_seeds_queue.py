"""
run_meig_seeds_queue.py — MEIG [1] 從 3 顆種子補到 10 顆的訓練佇列（proposal_1209 §8）

登記：docs/proposal_1209_revision.md §8 附錄（訓練前寫下）。MEIG 的方法不動：每個設定都用既有 s42 的
config_snapshot.yaml，只換 --seed 與 --tag。新增的 7 顆種子與 MAGNET 相同：{11, 99, 314, 555, 777, 888, 2026}。

    第一折主設定        tag tw50_bl_meig_s{S}          config = tw50_bl_meig_s42 的快照（主表）
    第二折主設定        tag f2_bl_meig_s{S}            config = f2_bl_meig_s42 的快照（主表第二折）
    第一折美股落後 l 天  tag tw50_bl_meigusl{l}_s{S}    config 同第一折主設定，加 --us-lag l（l = 2、3、4，§2）

每個 run 跑完，把它的 config_snapshot.yaml 與同一設定的 s42 快照比對：只能差 training.seed。
不符就記錄並停止派新的 run（已在跑的讓它跑完），結束碼 1。
執行邏輯（已完成略過、失敗只重跑一次、不存紀錄檔）沿用 run_input_lag_queue.py。

    caffeinate -i -m -s .venv/bin/python scripts/run_meig_seeds_queue.py --workers 2
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
from run_input_lag_queue import MEIG_CFG, PY, finished_run, job as lag_job, record, run_one  # noqa: E402

NEW_SEEDS = [11, 99, 314, 555, 777, 888, 2026]
F2_CFG = "runs/fold2/baselines/20261017_1458_f2_bl_meig_s42/config_snapshot.yaml"
LAGS = [2, 3, 4]


def _ref_lag(l: int) -> str:
    hits = sorted((ROOT / "runs").rglob(f"*_tw50_bl_meigusl{l}_s42/config_snapshot.yaml"))
    assert len(hits) == 1, hits
    return str(hits[0].relative_to(ROOT))


def jobs() -> list[dict]:
    out = []
    for s in NEW_SEEDS:                      # 種子在外層：中途停下時各設定的種子數相同
        out.append(dict(seed=s, tag=f"tw50_bl_meig_s{s}", ref=MEIG_CFG,
                        cmd=[PY, "-m", "src.train.train", "--config", MEIG_CFG, "--seed", str(s),
                             "--tag", f"tw50_bl_meig_s{s}"]))
        out.append(dict(seed=s, tag=f"f2_bl_meig_s{s}", ref=F2_CFG,
                        cmd=[PY, "-m", "src.train.train", "--config", F2_CFG, "--seed", str(s),
                             "--tag", f"f2_bl_meig_s{s}"]))
        for l in LAGS:
            j = lag_job("meig", l, s)
            j["ref"] = _ref_lag(l)
            out.append(j)
    return out


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def config_diff(run_dir: Path, ref: str) -> dict:
    a = _flat(yaml.safe_load((ROOT / ref).read_text()))
    b = _flat(yaml.safe_load((run_dir / "config_snapshot.yaml").read_text()))
    return {k: (a.get(k, "<無>"), b.get(k, "<無>")) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="只列出指令")
    args = ap.parse_args()

    js = jobs()
    if args.dry_run:
        for j in js:
            print(("[已完成] " if finished_run(j["tag"]) else "") + " ".join(j["cmd"]) + f"   ref={j['ref']}")
        return
    record(event="queue_start", n_jobs=len(js), workers=args.workers)
    q: queue.Queue = queue.Queue()
    for j in js:
        q.put(j)
    stop = threading.Event()
    bad: list[str] = []

    def worker() -> None:
        while not stop.is_set():
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            try:
                run_one(j)
                d = finished_run(j["tag"])
                if d is not None:
                    diff = config_diff(d, j["ref"])
                    ok = set(diff) == {"training.seed"} and diff["training.seed"][1] == j["seed"]
                    record(tag=j["tag"], event="config_check", ok=ok, diff=diff)
                    if not ok:
                        bad.append(j["tag"])
                        stop.set()
            except Exception as e:                      # 佇列本身不因單一 run 的意外而停
                record(tag=j["tag"], event="error", error=repr(e))

    ts = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    missing = [j["tag"] for j in js if finished_run(j["tag"]) is None]
    record(event="queue_end", missing=missing, config_mismatch=bad)
    sys.exit(1 if (missing or bad) else 0)


if __name__ == "__main__":
    main()
