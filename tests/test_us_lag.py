"""
test_us_lag.py — 美股輸入落後天數（MultiplexDataset 的 us_lag）的回歸與前視保護

評估議題 3（資訊時效，proposal_1209 §2）要把美股層的輸入換成落後 l 個美股交易日的版本，
其餘（台股輸入、圖、標籤）不動。這裡驗證：

  1. us_lag = 1（預設）的樣本與不給參數時逐位元相同
  2. us_lag = l 時，x_seq_L1 的每一欄確實是該檔美股在 target_date 之前倒數第 l 列往回 T 列，
     用另一條路徑（searchsorted 直接算列號）回頭比對原始 CSV
  3. 只有 x_seq_L1 變：x_seq_L2、y、兩層的邊都與 us_lag = 1 相同
  4. 沒有前視：落後 l 天用到的最後一列日期嚴格早於落後 l − 1 天的，且一律早於 target_date
  5. us_lag < 1 直接拋錯；不給參數時讀 config 的 data.us_lag

用本機的 tw50 快照（data/graphs/snapshots_tw50，不進版控）；沒有就略過。

    .venv/bin/python -m pytest tests/test_us_lag.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.multiplex_dataset import TECH_COLS, MultiplexDataset  # noqa: E402

CONFIG = ROOT / "configs" / "tw50.yaml"
SNAPSHOTS = ROOT / "data" / "graphs" / "snapshots_tw50"
FEATURES = ROOT / "data" / "features"
T = 3                     # 取 3 列，視窗的平移才測得到（主結果 T = 1，MEIG T = 20）
LAGS = (2, 3, 4)
# 每個 split 抽幾筆：頭、尾與中段
PICK = (0, 1, 57, 123, -2, -1)

pytestmark = pytest.mark.skipif(not SNAPSHOTS.exists(), reason="沒有本機的 tw50 快照")


def _ds(split: str, **kw) -> MultiplexDataset:
    return MultiplexDataset(snapshot_dir=str(SNAPSHOTS), features_dir=str(FEATURES), T=T,
                            split=split, config_path=str(CONFIG), **kw)


@pytest.fixture(scope="module")
def datasets() -> dict:
    out = {}
    for split in ("train", "test"):
        out[split] = {"default": _ds(split), **{l: _ds(split, us_lag=l) for l in (1, *LAGS)}}
    return out


@pytest.fixture(scope="module")
def us_csv() -> dict[str, pd.DataFrame]:
    """獨立於 Dataset 的另一條讀檔路徑。"""
    tickers = _ds("test").adr_tickers
    return {tk: pd.read_csv(FEATURES / "adr" / f"{tk}.csv", index_col=0, parse_dates=True)[TECH_COLS]
            for tk in tickers}


def _reference(us_csv: dict, tickers: list[str], target: pd.Timestamp, lag: int):
    """回傳 (x [T, n, F], 每檔用到的最後一列日期)。列號直接用 searchsorted 算。"""
    cols, last_dates = [], []
    for tk in tickers:
        df = us_csv[tk]
        n_before = int(df.index.searchsorted(target, side="left"))   # < target 的列數
        end = n_before - (lag - 1)                                   # 視窗結束的列號（不含）
        assert end - T >= 0, "測試抽到歷史不足的樣本"
        w = df.iloc[end - T:end]
        cols.append(np.nan_to_num(w.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        last_dates.append(w.index[-1])
    return np.stack(cols, axis=0).transpose(1, 0, 2), last_dates


def test_default_is_lag1(datasets: dict) -> None:
    """不給 us_lag 與 us_lag = 1 的樣本逐位元相同。"""
    for split in datasets:
        d0, d1 = datasets[split]["default"], datasets[split][1]
        assert d0.us_lag == 1 and d1.us_lag == 1
        for i in PICK:
            a, b = d0[i], d1[i]
            assert a["target_date"] == b["target_date"]
            for k in ("x_seq_L1", "x_seq_L2", "y"):
                assert torch.equal(a[k], b[k]), (split, i, k)


@pytest.mark.parametrize("lag", (1, *LAGS))
def test_us_window_matches_csv(datasets: dict, us_csv: dict, lag: int) -> None:
    """x_seq_L1 的第 j 欄 = 第 j 檔美股在 target_date 之前倒數第 lag 列往回 T 列。"""
    for split in datasets:
        ds = datasets[split][lag]
        for i in PICK:
            s = ds[i]
            ref, _ = _reference(us_csv, ds.adr_tickers, pd.Timestamp(s["target_date"]), lag)
            assert s["x_seq_L1"].shape == ref.shape
            assert np.array_equal(s["x_seq_L1"].numpy(), ref), (split, i, lag)


@pytest.mark.parametrize("lag", LAGS)
def test_only_us_input_changes(datasets: dict, lag: int) -> None:
    """台股輸入、標籤、兩層的邊都不受 us_lag 影響；美股輸入確實變了。"""
    for split in datasets:
        base, lagged = datasets[split][1], datasets[split][lag]
        n_diff = 0
        for i in PICK:
            a, b = base[i], lagged[i]
            assert a["target_date"] == b["target_date"]
            for k in ("x_seq_L2", "y", "edge_index_L1", "edge_attr_L1", "edge_index_L2", "edge_attr_L2",
                      "is_long_gap_L1", "is_long_gap_L2"):
                assert torch.equal(a[k], b[k]), (split, i, k)
            n_diff += int(not torch.equal(a["x_seq_L1"], b["x_seq_L1"]))
            # 平移 lag − 1 列：落後 lag 的後 T − (lag − 1) 列 = 落後 1 的前 T − (lag − 1) 列
            keep = T - (lag - 1)
            if keep > 0:
                assert torch.equal(b["x_seq_L1"][-keep:], a["x_seq_L1"][:keep]), (split, i, lag)
        assert n_diff == len(PICK), "us_lag 沒有改到美股輸入"


def test_no_lookahead(datasets: dict, us_csv: dict) -> None:
    """用到的最後一列日期：落後 l 嚴格早於落後 l − 1，且全部嚴格早於 target_date。"""
    ds = datasets["test"][1]
    for i in PICK:
        target = pd.Timestamp(ds[i]["target_date"])
        prev = None
        for lag in (1, *LAGS):
            _, last = _reference(us_csv, ds.adr_tickers, target, lag)
            assert all(d < target for d in last), (i, lag)
            if prev is not None:
                assert all(d < p for d, p in zip(last, prev)), (i, lag)
            prev = last


def test_invalid_lag_raises() -> None:
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            _ds("test", us_lag=bad)


def test_lag_read_from_config(tmp_path: Path) -> None:
    """不給參數時讀 config 的 data.us_lag（用 run 的 config_snapshot.yaml 重建資料集時靠這個）。"""
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["data"]["us_lag"] = 3
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    kw = dict(snapshot_dir=str(SNAPSHOTS), features_dir=str(FEATURES), T=T, split="test")
    assert MultiplexDataset(config_path=str(p), **kw).us_lag == 3
    assert MultiplexDataset(config_path=str(p), us_lag=2, **kw).us_lag == 2      # 明確傳入的優先
