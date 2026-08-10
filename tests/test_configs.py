"""
test_configs.py — base.yaml 與 tw50.yaml 的漂移守護

兩份 config 是刻意的複本：base.yaml 是 k=7 凍結基準的協議記錄，不能被
改成 tw50，否則 freeze_k7.py --verify 與 e7_acceptance.py 就失去意義。
代價是超參數存在兩份，一旦有人只調其中一份，k7 與 tw50 的結果就不再
可比——而且不會有任何錯誤訊息。

這裡把「允許不同的鍵」列成白名單，其餘一律要求逐字相同。新增合理的
差異時要同步改白名單與 tw50.yaml 檔頭，強迫這件事被寫下來。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.config import VALID_UNIVERSES, load_universe  # noqa: E402

BASE = ROOT / "configs" / "base.yaml"
TW50 = ROOT / "configs" / "tw50.yaml"

# 允許兩份 config 不同的鍵。每一項都要在 tw50.yaml 檔頭有對應說明。
ALLOWED_DIFFS = {
    "data.universe",
    "data.snapshot_dir",
    "data.end_date",
    "evaluation.portfolio.n_side",
    "mlflow.experiment_name",
}


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


@pytest.fixture(scope="module")
def cfgs() -> tuple[dict, dict]:
    return (_flatten(yaml.safe_load(BASE.read_text())),
            _flatten(yaml.safe_load(TW50.read_text())))


def test_same_key_set(cfgs) -> None:
    """兩份 config 的鍵集合必須相同——少一個鍵會讓某個超參悄悄吃到程式預設值。"""
    base, tw50 = cfgs
    assert set(base) == set(tw50), (
        f"只在 base: {sorted(set(base) - set(tw50))}；"
        f"只在 tw50: {sorted(set(tw50) - set(base))}"
    )


def test_only_whitelisted_keys_differ(cfgs) -> None:
    base, tw50 = cfgs
    differing = {k for k in base if base[k] != tw50[k]}
    unexpected = differing - ALLOWED_DIFFS
    assert not unexpected, (
        f"未列入白名單的差異：{sorted(unexpected)}。"
        f"若是刻意的，請同步更新 ALLOWED_DIFFS 與 tw50.yaml 檔頭。"
    )


def test_whitelisted_keys_actually_differ(cfgs) -> None:
    """白名單不得留下已經不成立的項目，否則它會遮蔽真正的漂移。"""
    base, tw50 = cfgs
    stale = {k for k in ALLOWED_DIFFS if base.get(k) == tw50.get(k)}
    assert not stale, f"白名單中這些鍵其實相同，應移除：{sorted(stale)}"


def test_split_indices_identical(cfgs) -> None:
    """
    split 是位置索引。兩份 config 的快照日期清單逐一相同，因此同一組索引
    必須給出同樣的 train/val/test 邊界——這是 k7 與 tw50 可比的前提。
    """
    base, tw50 = cfgs
    for k in ("data.split.train_end", "data.split.val_end"):
        assert base[k] == tw50[k], k


def test_universe_names_valid(cfgs) -> None:
    base, tw50 = cfgs
    assert base["data.universe"] == "k7"
    assert tw50["data.universe"] == "tw50"
    for name in (base["data.universe"], tw50["data.universe"]):
        assert name in VALID_UNIVERSES


def test_n_side_fits_universe(cfgs) -> None:
    """多空各取 n_side 檔，2*n_side 不得超過該 universe 的節點數。"""
    for cfg in cfgs:
        u = load_universe(cfg["data.universe"])
        n_side = int(cfg["evaluation.portfolio.n_side"])
        assert 2 * n_side <= u.n_l2, f"{u.name}: 2*{n_side} > {u.n_l2}"


def test_snapshot_dirs_are_distinct(cfgs) -> None:
    """
    tw50 絕不可指向 data/graphs/snapshots——那是 k=7 凍結基準的一部分，
    graph_builder 的 CLI 也有同樣的防呆。
    """
    base, tw50 = cfgs
    assert base["data.snapshot_dir"] != tw50["data.snapshot_dir"]
    assert tw50["data.snapshot_dir"] != "data/graphs/snapshots"


def test_align_loss_disabled_on_asymmetric_universe(cfgs) -> None:
    """n1 != n2 時對角線不再是正樣本集合，align loss 會在 CombinedLoss 拋錯。"""
    _, tw50 = cfgs
    u = load_universe(tw50["data.universe"])
    if u.n_l1 != u.n_l2:
        assert tw50["align_loss.enabled"] is False
        assert float(tw50["loss_weights.align"]) == 0.0
