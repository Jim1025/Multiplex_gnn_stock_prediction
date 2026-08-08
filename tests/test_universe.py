"""
test_universe.py — E3：Universe API 的回歸保護

最重要的一組是「k7 相容性」：新結構必須讓 k=7 的節點順序、配對關係與
pair_index 與既有常數完全一致。E7 回歸驗收要求把 universe 設回 7 對後
能重現凍結基準，若順序或配對變動，那個驗收就失去意義。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.config import (  # noqa: E402
    PAIR_MAP,
    Universe,
    get_pair_dict,
    load_universe,
)
from src.dataset.multiplex_dataset import ADR_TICKERS, N_NODES, TW_CODES  # noqa: E402


# ── k7 相容性（最關鍵） ────────────────────────────────────────

def test_k7_node_order_matches_existing_constants() -> None:
    """節點順序即張量索引；與既有常數不符會讓凍結基準失效。"""
    u = load_universe("k7")
    assert list(u.us_nodes) == list(ADR_TICKERS)
    assert list(u.tw_nodes) == list(TW_CODES)
    assert u.n_l1 == u.n_l2 == N_NODES


def test_k7_pairing_matches_pair_map() -> None:
    u = load_universe("k7")
    assert u.pairing == get_pair_dict()
    assert u.n_pairs == len(PAIR_MAP)
    assert u.pairing_rate == 1.0


def test_k7_pair_index_is_the_diagonal() -> None:
    """舊實作硬編碼 p(j)=j；新結構在 k7 下必須退化成同一件事。"""
    u = load_universe("k7")
    assert u.pair_index == tuple(range(N_NODES))
    assert all(i >= 0 for i in u.pair_index)


def test_default_universe_is_k7() -> None:
    """遷移期間預設必須維持 k7，否則既有路徑行為會變。"""
    assert load_universe().name == "k7"


# ── tw50 ────────────────────────────────────────────────────

def test_tw50_shape() -> None:
    u = load_universe("tw50")
    assert (u.n_l1, u.n_l2, u.n_pairs) == (30, 50, 7)
    assert abs(u.pairing_rate - 0.14) < 1e-9


def test_tw50_has_unpaired_nodes() -> None:
    """擴充的核心：多數 TW 節點沒有恆等邊，只能走候選邊。"""
    u = load_universe("tw50")
    pi = u.pair_index
    assert sum(1 for i in pi if i >= 0) == 7
    assert sum(1 for i in pi if i < 0) == 43


def test_tw50_pair_index_points_at_the_right_adr() -> None:
    u = load_universe("tw50")
    for j, i in enumerate(u.pair_index):
        if i < 0:
            continue
        adr = u.us_nodes[i]
        assert PAIR_MAP[adr]["tw"] == u.tw_nodes[j]


def test_tw50_every_node_has_industry() -> None:
    u = load_universe("tw50")
    for code in u.tw_nodes:
        assert u.industry.get(code)


# ── 驗證器 ──────────────────────────────────────────────────

def test_validate_rejects_pairing_outside_node_lists() -> None:
    with pytest.raises(ValueError, match="pairing 的 US 端"):
        Universe("bad", ("A",), ("x",), {"B": "x"}, {"x": "i"}).validate()
    with pytest.raises(ValueError, match="pairing 的 TW 端"):
        Universe("bad", ("A",), ("x",), {"A": "y"}, {"x": "i"}).validate()


def test_validate_rejects_duplicate_nodes() -> None:
    with pytest.raises(ValueError, match="us_nodes 有重複"):
        Universe("bad", ("A", "A"), ("x",), {}, {"x": "i"}).validate()


def test_validate_rejects_many_to_one_pairing() -> None:
    """恆等邊必須 1-to-1，否則對角化的語意會被破壞。"""
    with pytest.raises(ValueError, match="1-to-1"):
        Universe("bad", ("A", "B"), ("x",), {"A": "x", "B": "x"},
                 {"x": "i"}).validate()


def test_validate_rejects_missing_industry() -> None:
    with pytest.raises(ValueError, match="缺產業別"):
        Universe("bad", ("A",), ("x",), {"A": "x"}, {}).validate()


def test_load_universe_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="未知 universe"):
        load_universe("nope")
