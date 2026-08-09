"""
test_dataset_universe.py — E5：MultiplexDataset 的 universe 感知回歸保護

E5 之前，Dataset 用模組層級的 k7 常數組裝 x_seq，而 y 與邊來自快照檔。
兩者一旦源自不同 universe，只要節點數碰巧相同就不會有任何例外，
訓練照跑但每一欄對到不同公司。這裡驗證三件事：

  1. k7 行為不變（節點數、順序、樣本 shape）
  2. tw50 下 Dataset 真的吃到 30 / 50 的不對稱節點集，且第 j 欄
     確實是 tw_nodes[j] 這檔股票的歷史
  3. 快照與 universe 不符時建構階段就拋錯，而非靜默算出爛結果

tw50 的快照不進版控（data/graphs/snapshots/ 是 k7 凍結基準的一部分，
不得覆寫），因此測試用 tmp_path 現場建幾張。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.config import load_universe  # noqa: E402
from src.dataset.graph_builder import MultiplexGraphBuilder  # noqa: E402
from src.dataset.multiplex_dataset import (  # noqa: E402
    ADR_TICKERS,
    F,
    TECH_COLS,
    TW_CODES,
    MultiplexDataset,
    multiplex_collate,
)

CONFIG = str(ROOT / "configs" / "base.yaml")
FEATURES = str(ROOT / "data" / "features")
K7_SNAPSHOTS = ROOT / "data" / "graphs" / "snapshots"

# 現場建快照的日期範圍：需落在暖機期（corr_window=60）之後
BUILD_START = pd.Timestamp("2024-03-01")
BUILD_END = pd.Timestamp("2024-03-08")


@pytest.fixture(scope="module")
def tw50_snapshot_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """用 tw50 universe 現場建少量快照，回傳目錄。"""
    u = load_universe("tw50")
    out = tmp_path_factory.mktemp("snapshots_tw50")
    gb = MultiplexGraphBuilder(
        pair_map=u.pairing,
        adr_dir=str(Path(FEATURES) / "adr"),
        tw_dir=str(Path(FEATURES) / "tw"),
        universe=u,
    )
    gb.build_sequence(BUILD_START, BUILD_END, out_dir=str(out))
    n = len(list(out.glob("graph_*.pt")))
    assert n > 0, f"{BUILD_START.date()}~{BUILD_END.date()} 未產出任何 tw50 快照"
    return out


# ── k7：行為不得改變 ───────────────────────────────────────────

def test_k7_default_unchanged() -> None:
    """未指定 universe 時應取 base.yaml 的 data.universe（k7），順序與常數一致。"""
    ds = MultiplexDataset(
        snapshot_dir=str(K7_SNAPSHOTS), features_dir=FEATURES,
        split="all", config_path=CONFIG,
    )
    assert ds.universe.name == "k7"
    assert ds.adr_tickers == list(ADR_TICKERS)
    assert ds.tw_codes == list(TW_CODES)
    assert ds.n_l1 == ds.n_l2 == 7
    assert ds.pair_index == list(range(7)), "k7 的恆等邊應仍為完整對角線"

    sample = ds[0]
    assert sample["x_seq_L1"].shape == (ds.T, 7, F)
    assert sample["x_seq_L2"].shape == (ds.T, 7, F)
    assert sample["y"].shape == (7,)


def test_get_ticker_order_is_per_instance(tw50_snapshot_dir: Path) -> None:
    """E5 前這是 staticmethod，任何實例都回傳 k7 順序。"""
    k7 = MultiplexDataset(
        snapshot_dir=str(K7_SNAPSHOTS), features_dir=FEATURES,
        split="all", config_path=CONFIG,
    )
    tw50 = MultiplexDataset(
        snapshot_dir=str(tw50_snapshot_dir), features_dir=FEATURES,
        split="all", config_path=CONFIG, universe="tw50",
    )
    assert k7.get_ticker_order() != tw50.get_ticker_order()
    assert k7.get_ticker_order()[0] == list(ADR_TICKERS)
    assert len(tw50.get_ticker_order()[0]) == 30
    assert len(tw50.get_ticker_order()[1]) == 50


# ── tw50：不對稱節點集 ─────────────────────────────────────────

def test_tw50_sample_shapes(tw50_snapshot_dir: Path) -> None:
    """兩層節點數不相等，是擴充的核心；y 只在 TW 層。"""
    ds = MultiplexDataset(
        snapshot_dir=str(tw50_snapshot_dir), features_dir=FEATURES,
        split="all", config_path=CONFIG, universe="tw50",
    )
    assert (ds.n_l1, ds.n_l2) == (30, 50)
    assert ds.n_nodes == (30, 50)

    sample = ds[0]
    assert sample["x_seq_L1"].shape == (ds.T, 30, F)
    assert sample["x_seq_L2"].shape == (ds.T, 50, F)
    assert sample["y"].shape == (50,)
    assert sample["is_long_gap_L1"].shape == (30,)
    assert sample["is_long_gap_L2"].shape == (50,)
    assert sample["edge_index_L1"].shape[0] == 2
    assert sample["edge_index_L2"].shape[0] == 2


def test_tw50_pair_index_is_sparse_and_not_diagonal(tw50_snapshot_dir: Path) -> None:
    """50 檔台股只有 7 檔有配對——恆等邊不再是對角線。"""
    ds = MultiplexDataset(
        snapshot_dir=str(tw50_snapshot_dir), features_dir=FEATURES,
        split="all", config_path=CONFIG, universe="tw50",
    )
    paired = [(j, i) for j, i in enumerate(ds.pair_index) if i >= 0]
    assert len(paired) == 7
    assert ds.pair_index != list(range(50))
    for j, i in paired:
        assert ds.universe.pairing[ds.adr_tickers[i]] == ds.tw_codes[j]


def test_tw50_sequence_columns_match_tickers(tw50_snapshot_dir: Path) -> None:
    """
    最關鍵的一項：x_seq 的第 j 欄必須真的是 tickers[j] 的歷史。

    節點數改變時若清單與索引脫鉤，shape 仍然正確、訓練仍然會跑，
    只是每一欄都對到錯的公司。這裡直接回頭比對原始 CSV。
    """
    ds = MultiplexDataset(
        snapshot_dir=str(tw50_snapshot_dir), features_dir=FEATURES,
        split="all", config_path=CONFIG, universe="tw50",
    )
    sample = ds[0]
    target = pd.Timestamp(sample["target_date"])

    def expect(market: str, ticker: str) -> np.ndarray:
        df = pd.read_csv(
            Path(FEATURES) / market / f"{ticker}.csv", index_col=0, parse_dates=True
        )
        past = df.loc[df.index < target][TECH_COLS].astype(np.float32)
        w = past.iloc[-ds.T:].to_numpy(dtype=np.float32)
        return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    # 兩層各抽頭、中、尾三欄
    for j in (0, ds.n_l1 // 2, ds.n_l1 - 1):
        got = sample["x_seq_L1"][:, j, :].numpy()
        np.testing.assert_allclose(got, expect("adr", ds.adr_tickers[j]), rtol=0, atol=0)
    for j in (0, ds.n_l2 // 2, ds.n_l2 - 1):
        got = sample["x_seq_L2"][:, j, :].numpy()
        np.testing.assert_allclose(got, expect("tw", ds.tw_codes[j]), rtol=0, atol=0)


def test_tw50_collate_keeps_asymmetry(tw50_snapshot_dir: Path) -> None:
    ds = MultiplexDataset(
        snapshot_dir=str(tw50_snapshot_dir), features_dir=FEATURES,
        split="all", config_path=CONFIG, universe="tw50",
    )
    b = multiplex_collate([ds[i] for i in range(min(2, len(ds)))])
    B = b["y"].shape[0]
    assert b["x_seq_L1"].shape == (B, ds.T, 30, F)
    assert b["x_seq_L2"].shape == (B, ds.T, 50, F)
    assert b["y"].shape == (B, 50)


# ── 一致性守衛 ─────────────────────────────────────────────────

def test_mismatched_universe_fails_fast(tw50_snapshot_dir: Path) -> None:
    """拿 k7 的清單去讀 tw50 的快照，必須在建構時就拋錯。"""
    with pytest.raises(ValueError, match="快照與 universe 不符"):
        MultiplexDataset(
            snapshot_dir=str(tw50_snapshot_dir), features_dir=FEATURES,
            split="all", config_path=CONFIG, universe="k7",
        )


def test_k7_snapshots_rejected_by_tw50_universe() -> None:
    """反向也要擋：tw50 的清單不得讀 k7 快照。"""
    with pytest.raises(ValueError, match="快照與 universe 不符"):
        MultiplexDataset(
            snapshot_dir=str(K7_SNAPSHOTS), features_dir=FEATURES,
            split="all", config_path=CONFIG, universe="tw50",
        )


def test_unknown_universe_name_rejected() -> None:
    with pytest.raises(ValueError, match="未知 universe"):
        MultiplexDataset(
            snapshot_dir=str(K7_SNAPSHOTS), features_dir=FEATURES,
            split="all", config_path=CONFIG, universe="tw100",
        )


def test_tw50_snapshots_not_written_into_frozen_dir(tw50_snapshot_dir: Path) -> None:
    """凍結基準目錄必須仍是 7 節點——測試本身不得污染它。"""
    snap = torch.load(sorted(K7_SNAPSHOTS.glob("graph_*.pt"))[0], weights_only=False)
    assert snap["adr"].x.shape[0] == 7
    assert snap["tw"].y.shape[0] == 7
