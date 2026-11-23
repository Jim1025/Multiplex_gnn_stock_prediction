"""
test_us_only.py — MAGNET「只用美股」（model.us_only）的回歸與隔離檢查

評估議題 2（資訊來源，proposal_1209 §7）要補 MAGNET 只用美股的版本：融合前把台股表示 h_L2 歸零，
h_fused = gate ⊙ ĥ_1（disable_a12 的鏡像），架構、參數量、計算圖不變。這裡驗證：

  1. 預設值不變：用改過的程式重評主結果 SA s42 的 best.pt（CPU），與存下的 test 預測逐筆相同（≤ 1e-6）
  2. us_only 的輸出與台股輸入 x_seq_L2 完全無關（把 x_seq_L2 換成雜訊，ŷ 逐位元相同），但會隨美股輸入改變
  3. 預設值下 ŷ 會隨 x_seq_L2 改變（確認 2 不是因為台股輸入本來就沒作用）
  4. us_only 不改參數：同一個種子初始化，state_dict 的鍵、形狀與數值都相同
  5. us_only 與 disable_a12、tw_lag_channel 同時開會拋錯

用本機的 SA s42 run 與 tw50 快照（runs/、data/ 不進版控）；沒有就略過。

    .venv/bin/python -m pytest tests/test_us_only.py -v
"""

from __future__ import annotations

import copy
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

from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train.utils import load_checkpoint  # noqa: E402

SA = ROOT / "runs" / "tw50" / "relational" / "20261023_2142_tw50_rgcnSA_s42"
CFG_PATH = SA / "config_snapshot.yaml"

pytestmark = pytest.mark.skipif(not (SA / "checkpoints" / "best.pt").exists(), reason="沒有本機的 SA s42 run")


def _cfg(**model_kw) -> dict:
    cfg = yaml.safe_load(open(CFG_PATH))
    cfg["model"].update(model_kw)
    return cfg


def _model(seed: int = 0, **model_kw):
    torch.manual_seed(seed)
    m = build_model(_cfg(**model_kw))
    m.eval()
    return m


@pytest.fixture(scope="module")
def test_ds() -> MultiplexDataset:
    cfg = _cfg()
    return MultiplexDataset(snapshot_dir=str(ROOT / cfg["data"]["snapshot_dir"]),
                            features_dir=str(ROOT / cfg["data"]["features_dir"]),
                            T=cfg["model"]["lstm"]["T_history"], split="test", config_path=str(CFG_PATH))


@pytest.fixture(scope="module")
def batch(test_ds) -> dict:
    return multiplex_collate([test_ds[i] for i in (0, 1, 100, 245)])


def _noise_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(x.shape, generator=g, dtype=x.dtype) * x.std() + x.mean()


def _predict_all(model, ds) -> np.ndarray:
    """與 src.train.evaluator 相同的批次與排列：[天, 檔]，欄位順序為 ds.tw_codes。"""
    bs = int(_cfg()["training"]["batch_size"])
    rows = []
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            y_hat, _ = model(multiplex_collate([ds[k] for k in range(i, min(i + bs, len(ds)))]))
            rows.append(y_hat.numpy())
    return np.vstack(rows)


def test_default_reproduces_sa_checkpoint(test_ds):
    model = build_model(_cfg())
    load_checkpoint(SA / "checkpoints" / "best.pt", model, optimizer=None, map_location="cpu")
    model.eval()
    assert model.us_only is False
    pred = _predict_all(model, test_ds)                                      # [246, 50]
    df = pd.read_csv(SA / "predictions" / "test_predictions.csv", dtype={"ticker": str})
    H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
    H = H[[str(t) for t in test_ds.tw_codes]]
    assert pred.shape == H.shape
    np.testing.assert_allclose(pred, H.to_numpy(), atol=1e-6, rtol=0)


def test_us_only_ignores_tw_input(batch):
    m = _model(us_only=True)
    with torch.no_grad():
        y0, ex0 = m(batch)
        b_tw = copy.copy(batch)
        b_tw["x_seq_L2"] = _noise_like(batch["x_seq_L2"], 1)
        y1, _ = m(b_tw)
        b_us = copy.copy(batch)
        b_us["x_seq_L1"] = _noise_like(batch["x_seq_L1"], 2)
        y2, _ = m(b_us)
    assert torch.equal(y0, y1), "us_only 的輸出不應受台股輸入影響"
    assert not torch.allclose(y0, y2), "us_only 的輸出應隨美股輸入改變"
    assert torch.count_nonzero(ex0["h_L2"]) == 0


def test_default_uses_tw_input(batch):
    m = _model()
    with torch.no_grad():
        y0, _ = m(batch)
        b_tw = copy.copy(batch)
        b_tw["x_seq_L2"] = _noise_like(batch["x_seq_L2"], 1)
        y1, _ = m(b_tw)
    assert not torch.allclose(y0, y1)


def test_us_only_same_parameters():
    a, b = _model(seed=7), _model(seed=7, us_only=True)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k]), k


def test_us_only_conflicts():
    with pytest.raises(ValueError, match="disable_a12"):
        build_model(_cfg(us_only=True, disable_a12=True))
    with pytest.raises(ValueError, match="tw_lag_channel"):
        build_model(_cfg(us_only=True, tw_lag_channel={"enabled": True, "blend_w": 0.5}))
