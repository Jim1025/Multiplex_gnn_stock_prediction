"""
tw50_rollback.py — E1：由現行成分股回推指定日期的臺灣50成分股

為什麼要 assertion：
  回推法對錯誤零容忍——變動紀錄中任何一列錯誤，都會讓最終名單靜默地錯掉，
  而且不會有任何徵兆。本腳本在每一步撤銷時檢查三個不變量，
  讓錯誤在「發生的那一季」當場炸掉，而不是在最後產出一份看似正常的錯名單。

不變量：
  I1  撤銷「新增」：該檔此刻必須在集合內（否則變動紀錄前後矛盾）
  I2  撤銷「移除」：該檔此刻必須不在集合內（否則它被重複移除）
  I3  每步之後集合大小必須恆為 50（審核為等量換入換出）
  I4  每次審核的 added 與 removed 數量必須相等

時間慣例：
  定期審核於 3/6/9/12 月，生效日為該月「第三個星期五」收盤後。
  故目標日 D 適用的名單 = 撤銷所有「生效日 > D」的審核之後的結果。
  例：D = 2019-01-02 → 2018-12 審核（生效 2018-12-21）已生效，不撤銷；
      2019-03 審核（生效 2019-03-15）尚未發生，須撤銷。

用法：
    .venv/bin/python scripts/tw50_rollback.py
    .venv/bin/python scripts/tw50_rollback.py --target-date 2019-01-02 --write
"""

from __future__ import annotations

import argparse
import json
import sys
from calendar import Calendar, FRIDAY
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "configs" / "universe" / "tw50_change_log.json"
DEFAULT_OUT = ROOT / "configs" / "universe" / "tw50_constituents_reconstructed.json"

EXPECTED_SIZE = 50


class LedgerError(AssertionError):
    """變動紀錄自我矛盾——指出是哪一季、哪一檔。"""


def third_friday(year: int, month: int) -> date:
    """該月第三個星期五（審核生效日）。"""
    fridays = [
        d for d in Calendar().itermonthdates(year, month)
        if d.month == month and d.weekday() == FRIDAY
    ]
    return fridays[2]


def effective_date(quarter: str) -> date:
    y, m = quarter.split("-")
    return third_friday(int(y), int(m))


def rollback(
    current: list[str],
    reviews: list[dict],
    target: date,
    verbose: bool = True,
) -> tuple[set[str], list[str]]:
    """
    由 current 逆序撤銷所有生效日晚於 target 的審核。

    Returns:
        (重建後的成分股集合, 稽核軌跡)
    """
    members = set(current)
    trail: list[str] = []

    if len(members) != len(current):
        dup = [c for c in current if current.count(c) > 1]
        raise LedgerError(f"current_list 有重複代號：{sorted(set(dup))}")
    if len(members) != EXPECTED_SIZE:
        raise LedgerError(
            f"current_list 有 {len(members)} 檔，預期 {EXPECTED_SIZE}"
        )

    # 只撤銷生效日晚於目標日者，由新到舊
    todo = [r for r in reviews if effective_date(r["quarter"]) > target]
    todo.sort(key=lambda r: r["quarter"], reverse=True)

    for r in todo:
        q, added, removed = r["quarter"], r["added"], r["removed"]
        eff = effective_date(q)

        # I4：等量換入換出
        if len(added) != len(removed):
            raise LedgerError(
                f"[{q}] added {len(added)} 檔但 removed {len(removed)} 檔，數量不等："
                f"added={added} removed={removed}"
            )

        # I1：撤銷新增 —— 該檔此刻必須在集合內
        for code in added:
            if code not in members:
                raise LedgerError(
                    f"[{q}] 撤銷新增失敗：{code} 不在當前集合中。"
                    f"代表 {q} 之後、{q} 與現在之間某一季把它移除了卻未記錄，"
                    f"或本季的 added 名單有誤。"
                )
            members.remove(code)

        # I2：撤銷移除 —— 該檔此刻必須不在集合內
        for code in removed:
            if code in members:
                raise LedgerError(
                    f"[{q}] 撤銷移除失敗：{code} 已在當前集合中。"
                    f"代表它在 {q} 之後被重複移除，或本季的 removed 名單有誤。"
                )
            members.add(code)

        # I3：規模恆定
        if len(members) != EXPECTED_SIZE:
            raise LedgerError(
                f"[{q}] 撤銷後集合大小為 {len(members)}，預期 {EXPECTED_SIZE}"
            )

        line = (f"undo {q} (生效 {eff})  -{','.join(added)}  +{','.join(removed)}"
                f"  → n={len(members)}")
        trail.append(line)
        if verbose:
            print("  " + line)

    return members, trail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--target-date", default="2019-01-02",
                    help="要重建名單的日期（首個訓練日）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--write", action="store_true", help="寫出重建結果")
    args = ap.parse_args()

    log = json.loads(Path(args.log).read_text())
    target = date.fromisoformat(args.target_date)
    names: dict[str, str] = log.get("names", {})

    print(f"[tw50] 來源：{log['source']}（{log['source_retrieved']}）")
    print(f"[tw50] 驗證狀態：{log['verification_status']}")
    for c in log.get("known_conflicts", []):
        print(f"[tw50] 已知衝突 {c['review']}：{c['issue']}")
    print(f"[tw50] 現行名單基準日 {log['current_list_date']}，"
          f"目標日 {target}\n")

    try:
        members, trail = rollback(log["current_list"], log["reviews"], target)
    except LedgerError as e:
        print(f"\n[tw50] 不變量檢查失敗：\n  {e}\n")
        print("[tw50] 變動紀錄有矛盾，請核對上述季度後重跑。名單未產出。")
        sys.exit(1)

    ordered = sorted(members)
    print(f"\n[tw50] 重建完成：{target} 的成分股共 {len(ordered)} 檔，"
          f"撤銷 {len(trail)} 次審核")
    for i in range(0, len(ordered), 5):
        row = ordered[i:i + 5]
        print("  " + "  ".join(f"{c} {names.get(c, '?'):<12}" for c in row))

    unknown = [c for c in ordered if c not in names]
    if unknown:
        print(f"\n[tw50] 警告：{len(unknown)} 檔缺少名稱對照：{unknown}")

    if args.write:
        out = {
            "target_date":        str(target),
            "n_constituents":     len(ordered),
            "verification_status": log["verification_status"],
            "derived_from":       {
                "source":            log["source"],
                "current_list_date": log["current_list_date"],
                "reviews_undone":    len(trail),
            },
            "constituents": [
                {"code": c, "name": names.get(c, "")} for c in ordered
            ],
            "audit_trail": trail,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"\n[tw50] 寫出 {args.out}")
    else:
        print("\n[tw50] 未寫檔（加 --write 才輸出）")


if __name__ == "__main__":
    main()
