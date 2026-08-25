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
IMED = ROOT / "configs" / "tw50_imed_wl0.yaml"

# 允許兩份 config 不同的鍵。每一項都要在 tw50.yaml 檔頭有對應說明。
ALLOWED_DIFFS = {
    "data.universe",
    "data.snapshot_dir",
    "data.end_date",
    "evaluation.portfolio.n_side",
    "mlflow.experiment_name",
}

# tw50_imed_wl0.yaml 是 2x2 消融（融合位置 x 跨層邊可學性）的第四格，
# 對照組是 tw50chk_wl0_*（融合在圖之後、λ=0）。兩格只能差在融合位置，
# 其餘任何一項漂移都會讓「主效果 vs 交互作用」的分離失去意義。
IMED_ALLOWED_DIFFS = {
    "model.architecture",
    "mlflow.experiment_name",
}
# 2026-08-24：lambda_sparse 與 beta_* 一併搬進 base.yaml / tw50.yaml
# （超參一律從 configs 讀，不留在 .py 的 .get() 預設裡），
# 所以 imed 唯一還能多出來的鍵只剩 mode——它由 --architecture 決定，
# 不是超參。
IMED_ALLOWED_EXTRA = {
    "model.weak_links.mode",
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


# ---------------------------------------------------------------------------
# tw50_imed_wl0.yaml — 2x2 消融第四格
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def imed() -> tuple[dict, dict]:
    return (_flatten(yaml.safe_load(TW50.read_text())),
            _flatten(yaml.safe_load(IMED.read_text())))


def test_imed_key_set(imed) -> None:
    """只准多出 weak_links 兩個鍵；少任何一個鍵會讓超參悄悄吃到程式預設值。"""
    tw50, im = imed
    assert set(tw50) - set(im) == set(), f"tw50 有但 imed 缺：{sorted(set(tw50) - set(im))}"
    extra = set(im) - set(tw50)
    assert extra == IMED_ALLOWED_EXTRA, f"未預期的新增鍵：{sorted(extra - IMED_ALLOWED_EXTRA)}"


def test_imed_only_whitelisted_keys_differ(imed) -> None:
    tw50, im = imed
    differing = {k for k in tw50 if tw50[k] != im[k]}
    unexpected = differing - IMED_ALLOWED_DIFFS
    assert not unexpected, (
        f"未列入白名單的差異：{sorted(unexpected)}。這一格與 tw50chk_wl0_* "
        f"只能差在融合位置，其餘漂移會讓 2x2 消融失去意義。"
    )


def test_imed_is_the_intended_cell(imed) -> None:
    """
    這一格的定義：融合提前（magnet_intermediate）+ 候選邊可學（mode=free）
    + λ=0。λ 必須與對照組 tw50chk_wl0_* 相同，否則比較的就不只是融合位置。
    """
    _, im = imed
    assert im["model.architecture"] == "magnet_intermediate"
    assert im["model.weak_links.mode"] == "free"
    assert float(im["model.weak_links.lambda_sparse"]) == 0.0
