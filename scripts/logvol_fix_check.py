"""logvol_fix_check.py — `log_volume_z` 的零成交量缺陷對文獻 baseline 的影響（proposal §55.9(h)）

問題：data/features/tw 有 398 列成交量為 0——7 個全市場日期（346 列：3 天休市、3 天資料商缺漏、
2021-08-17 的 46 檔佔位列）與 52 列個股層級（多為停牌）。pipeline 的 log_volume = log1p(Volume)，
這些列等於 0，平常約 15–20；這個離群值在 60 日滾動標準差裡停留 60 天，把 log_volume_z 壓扁。
六個文獻 baseline 吃 9 個特徵（含 log_volume_z）；MAGNET 主結果 arm 只吃 log_return，不受影響。

**不動資料層**：data/ 與 src/dataset/ 都不寫入，修正版特徵與重訓的 run 都放在呼叫者指定的目錄。

  build   OUT        修正版特徵寫到 OUT：成交量為 0 的列改用前一天的成交量，只重算 log_volume_z
                     （公式與 pipeline._step5_normalize 相同），其餘欄位逐位元不變；並印出壓縮的量測
  workdir DIR FEAT   建隔離的工作目錄：configs 與 data/* 以 symlink 指回專案，只有 data/features 指向
                     FEAT。在裡面跑 train.py，run 與 INDEX.csv 都落在 DIR/runs，不碰專案的 runs/
  run     DIR        在 DIR 內依序重訓六個 baseline x 3 顆種子，各用存檔的 config_snapshot.yaml
  compare DIR        處理組（DIR/runs）對照組（runs/tw50/baselines 的 CPU 重評值 = 主表數字）逐種子配對

對照組的正當性：用**原始**特徵在隔離目錄重跑 Adv-ALSTM s42，epoch 數與最佳 epoch 相同、test IC 與
存檔的 CPU 重評值逐位元一致，所以存檔的重評值可以直接當對照組，不必重跑。

用法（路徑自選）：
    .venv/bin/python scripts/logvol_fix_check.py build   <feat_fix>
    .venv/bin/python scripts/logvol_fix_check.py workdir <work_fix> <feat_fix>
    .venv/bin/python scripts/logvol_fix_check.py run     <work_fix>          # 約 1 小時 40 分
    .venv/bin/python scripts/logvol_fix_check.py compare <work_fix>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WIN = 60                                   # 與 pipeline.ZSCORE_WIN 相同
SEEDS = (42, 7, 123)
BASELINES = {"man_sf": "MAN-SF [11]", "meig": "MEIG [1]", "hgt": "HGT [14]",
             "adv_alstm": "Adv-ALSTM [10]", "delta_lag": "DeltaLag [13]", "hats": "HATS [12]"}
MET = ("IC", "RankIC", "ICIR")


def z_of(lv: pd.Series) -> pd.Series:
    """pipeline._step5_normalize 的同一個公式。"""
    rm = lv.rolling(WIN, min_periods=60).mean()
    rs = lv.rolling(WIN, min_periods=60).std()
    return (lv - rm) / rs.replace(0, np.nan)


def tw_files() -> list[str]:
    return sorted(glob.glob(str(ROOT / "data/features/tw/[0-9]*.csv")))


def build(out: Path) -> None:
    from scipy.stats import spearmanr
    if out.exists():
        raise SystemExit(f"{out} 已存在；請給一個新的目錄")
    (out / "tw").mkdir(parents=True)
    shutil.copytree(ROOT / "data/features/adr", out / "adr")     # 美股端沒有零成交量，原樣複製
    shutil.copy(ROOT / "data/features/tw/feature_report.csv", out / "tw/feature_report.csv")

    repro, parts, Z0, Z1, R = 0.0, [], {}, {}, {}
    for f in tw_files():
        raw = pd.read_csv(f, dtype=str, keep_default_na=False)   # 字串讀回，其餘欄位原封寫出
        x = pd.read_csv(f)
        lv0 = np.log1p(x.Volume.astype(float))
        z0 = z_of(lv0)
        s = x.log_volume_z
        if not (s.notna() == z0.notna()).all():
            raise SystemExit(f"{f}：重現 pipeline 的 NaN 位置不符")
        m = s.notna()
        repro = max(repro, float((s[m] - z0[m]).abs().max()))
        v = x.Volume.astype(float).replace(0, np.nan).ffill()
        z1 = z_of(np.log1p(v))
        infl = lv0.rolling(WIN, min_periods=60).std() / np.log1p(v).rolling(WIN, min_periods=60).std()
        parts.append(pd.DataFrame({"z0": z0, "z1": z1, "infl": infl, "zero": x.Volume == 0,
                                   "date": x.Date.astype(str).str[:10]}))
        c = Path(f).stem
        Z0[c], Z1[c] = z0.set_axis(x.Date.astype(str).str[:10]), z1.set_axis(x.Date.astype(str).str[:10])
        R[c] = x.log_return.set_axis(x.Date.astype(str).str[:10])
        raw["log_volume_z"] = ["" if not np.isfinite(t) else repr(float(t)) for t in z1]
        raw.to_csv(out / "tw" / Path(f).name, index=False)

    # 其餘欄位必須逐位元不變
    bad = 0
    for f in tw_files():
        a = pd.read_csv(f, index_col=0, parse_dates=True)
        b = pd.read_csv(out / "tw" / Path(f).name, index_col=0, parse_dates=True)
        other = [k for k in a.columns if k != "log_volume_z"]
        bad += int(not a[other].equals(b[other]))
    print(f"重現 pipeline 的 log_volume_z：max|diff| = {repro:.2e}；其餘欄位不同的檔案 {bad}/50")
    if bad:
        raise SystemExit("修正版特徵動到了其他欄位")

    P = pd.concat(parts)
    zc = P[P.zero]
    print(f"成交量為 0 的列 {len(zc)}（{zc.date.nunique()} 個日期）；"
          f"全市場日期（>= 40 檔）{int((zc.groupby('date').size() >= 40).sum())} 個")
    P = P[P.z0.notna() & P.z1.notna()]
    aff = P.infl > 1.5
    dd = P.groupby("date").infl.median()
    print(f"滾動 sd 被放大 > 1.5 倍的股票-日：{int(aff.sum())}/{len(P)}（{100 * aff.mean():.1f}%）；"
          f"受影響者的放大倍數中位數 {P.infl[aff].median():.2f}、最大 {P.infl.max():.2f}")
    print(f"全市場層級受影響的日期（該日 50 檔的中位放大倍數 > 1.5）：{int((dd > 1.5).sum())}/{dd.size}")
    print(f"成交量為 0 那一列的 z：原本中位數 {P.z0[P.zero].median():+.2f}，修正後 {P.z1[P.zero].median():+.2f}")

    Z0, Z1, R = (pd.DataFrame(d).sort_index() for d in (Z0, Z1, R))
    Rn = R.shift(-1)
    zero = R.std(1) < 1e-12
    ok = ~zero & ~zero.shift(-1, fill_value=False)
    changed = (Z0 - Z1).abs().max(1) > 1e-9

    def cs(A, B, rows):
        o = []
        for t in A.index[rows]:
            a, b = A.loc[t], B.loc[t]
            m = a.notna() & b.notna()
            if m.sum() > 10 and a[m].std() > 0 and b[m].std() > 0:
                o.append(spearmanr(a[m], b[m])[0])
        return np.array(o)

    rk = cs(Z0, Z1, (changed & ok).values)
    print(f"特徵被改動的 {int(changed.sum())} 天：修正前後各股排序的 Spearman 中位數 {np.median(rk):+.3f}"
          f"（10% 分位 {np.quantile(rk, .1):+.3f}）——受損的是尺度，不是排序")
    for nm, (lo, hi) in {"第一折訓練窗": ("2019-04-11", "2023-12-19"),
                         "第一折測試窗": ("2024-12-26", "2025-12-30")}.items():
        rows = (ok & (Z0.index >= lo) & (Z0.index <= hi)).values
        a, b = cs(Z0, Rn, rows), cs(Z1, Rn, rows)
        print(f"  {nm} log_volume_z 對隔天報酬的橫截面 RankIC：原始 {a.mean():+.4f}（t {a.mean() / a.std() * np.sqrt(len(a)):+.2f}）"
              f"  修正 {b.mean():+.4f}（t {b.mean() / b.std() * np.sqrt(len(b)):+.2f}），{len(a)} 天")


def workdir(d: Path, feat: Path) -> None:
    if d.exists():
        raise SystemExit(f"{d} 已存在；請給一個新的目錄")
    if not (feat / "tw").is_dir():
        raise SystemExit(f"{feat} 不是特徵目錄（缺 tw/）")
    (d / "data").mkdir(parents=True)
    (d / "configs").symlink_to(ROOT / "configs")
    for e in sorted((ROOT / "data").iterdir()):
        if e.name != "features":
            (d / "data" / e.name).symlink_to(e)
    (d / "data" / "features").symlink_to(feat.resolve())
    print(f"-> {d}（data/features -> {feat.resolve()}）")


def stored(name: str, seed: int) -> Path:
    hit = glob.glob(str(ROOT / f"runs/tw50/baselines/*_tw50_bl_{name}_s{seed}"))
    if len(hit) != 1:
        raise SystemExit(f"找不到唯一的存檔 run：{name} s{seed}")
    return Path(hit[0])


def run(d: Path) -> None:
    if not (d / "data" / "features").exists():
        raise SystemExit(f"{d} 不是 workdir 建出來的工作目錄")
    (d / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for name in ("adv_alstm", "delta_lag", "hats", "hgt", "man_sf", "meig"):
        for seed in SEEDS:
            cfg = stored(name, seed) / "config_snapshot.yaml"
            tag = f"fix_{name}_s{seed}"
            print(f"start {tag}", flush=True)
            with open(d / "logs" / f"{tag}.log", "w") as log:
                rc = subprocess.run([sys.executable, str(ROOT / "src/train/train.py"),
                                     "--config", str(cfg), "--tag", tag],
                                    cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"done  {tag} exit={rc}", flush=True)


def compare(d: Path) -> None:
    from scipy import stats
    allD = {m: [] for m in MET}
    print(f"{'baseline':16s} {'主表 IC':>9s} {'修正後 IC':>10s} {'ΔIC':>8s} {'同號':>5s} {'配對 p':>7s}")
    for key, lab in BASELINES.items():
        c_, t_ = [], []
        for seed in SEEDS:
            t = glob.glob(str(d / "runs" / f"*fix_{key}_s{seed}" / "meta.json"))
            if not t:
                raise SystemExit(f"處理組缺 {key} s{seed}")
            cm = json.load(open(stored(key, seed) / "meta_reeval.json"))["reevaluated"]
            tm = json.load(open(t[0]))["test_metrics"]
            c_.append([cm[m] for m in MET])
            t_.append([tm[m] for m in MET])
        c_, t_ = np.array(c_), np.array(t_)
        for k, m in enumerate(MET):
            allD[m] += list(t_[:, k] - c_[:, k])
        dIC = t_[:, 0] - c_[:, 0]
        same = max(int((dIC > 0).sum()), int((dIC < 0).sum()))
        print(f"{lab:16s} {c_[:, 0].mean():+9.4f} {t_[:, 0].mean():+10.4f} {dIC.mean():+8.4f}"
              f" {same:3d}/3 {stats.ttest_1samp(dIC, 0).pvalue:7.2f}")
    print("\n全部配對（6 個 baseline x 3 顆種子）")
    for m in MET:
        v = np.array(allD[m])
        print(f"  Δ{m:<7s} 平均 {v.mean():+.4f}  中位數 {np.median(v):+.4f}  範圍 [{v.min():+.4f}, {v.max():+.4f}]"
              f"  為正 {int((v > 0).sum())}/{len(v)}  配對 t p {stats.ttest_1samp(v, 0).pvalue:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="log_volume_z 零成交量缺陷對文獻 baseline 的影響（§55.9(h)）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").add_argument("out", type=Path)
    w = sub.add_parser("workdir")
    w.add_argument("dir", type=Path)
    w.add_argument("features", type=Path)
    sub.add_parser("run").add_argument("dir", type=Path)
    sub.add_parser("compare").add_argument("dir", type=Path)
    a = ap.parse_args()
    {"build": lambda: build(a.out), "workdir": lambda: workdir(a.dir, a.features),
     "run": lambda: run(a.dir), "compare": lambda: compare(a.dir)}[a.cmd]()


if __name__ == "__main__":
    main()
