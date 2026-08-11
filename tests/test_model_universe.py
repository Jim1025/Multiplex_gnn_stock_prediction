"""
test_model_universe.py — E6：模型層的 universe 感知回歸保護

E6 之前，需要知道節點結構的模型各自從 PAIR_MAP / k7 常數推導，
並且把恆等邊當成對角線（torch.eye）、把兩層節點數當成同一個 n。
這裡驗證三件事：

  1. k7 下所有架構的輸出不變（舊行為是新結構的特例）
  2. tw50 下 13 個架構全部跑得動，且輸出欄數 = n_l2
  3. MAGNET 的 h1_{p(j)} 對齊正確：無配對節點恆等項為零，
     有配對節點恰好取到 h_L1[p(j)]

tw50 的快照不進版控，測試用 tmp_path 現場建幾張。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.config import load_universe  # noqa: E402
from src.dataset.graph_builder import MultiplexGraphBuilder  # noqa: E402
from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate  # noqa: E402
from src.models import VALID_ARCHITECTURES, build_model  # noqa: E402
from src.models._universe import universe_from_cfg  # noqa: E402
from src.models.multiplex_gnn import MAGNET  # noqa: E402
from src.models.prediction_head import CombinedLoss  # noqa: E402

CONFIG = ROOT / "configs" / "base.yaml"
FEATURES = str(ROOT / "data" / "features")
K7_SNAPSHOTS = str(ROOT / "data" / "graphs" / "snapshots")


def _cfg(universe: str, arch: str | None = None) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["data"] = {**cfg["data"], "universe": universe}
    if arch is not None:
        cfg["model"] = {**cfg["model"], "architecture": arch}
    return cfg


@pytest.fixture(scope="module")
def k7_batch() -> dict:
    ds = MultiplexDataset(snapshot_dir=K7_SNAPSHOTS, features_dir=FEATURES,
                          split="all", config_path=str(CONFIG))
    return multiplex_collate([ds[i] for i in range(2)])


@pytest.fixture(scope="module")
def tw50_batch(tmp_path_factory: pytest.TempPathFactory) -> dict:
    u = load_universe("tw50")
    out = tmp_path_factory.mktemp("snapshots_tw50_models")
    MultiplexGraphBuilder(
        pair_map=u.pairing, adr_dir=f"{FEATURES}/adr", tw_dir=f"{FEATURES}/tw",
        universe=u,
    ).build_sequence(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-08"),
                     out_dir=str(out))
    ds = MultiplexDataset(snapshot_dir=str(out), features_dir=FEATURES,
                          split="all", config_path=str(CONFIG), universe="tw50")
    return multiplex_collate([ds[i] for i in range(2)])


# ── 全架構形狀 ─────────────────────────────────────────────────

@pytest.mark.parametrize("arch", VALID_ARCHITECTURES)
def test_all_architectures_k7(arch: str, k7_batch: dict) -> None:
    y_hat, _ = build_model(_cfg("k7", arch)).eval()(k7_batch)
    assert y_hat.shape == (2, 7)


# magnet_weak_industry 在 tw50 下刻意無法建構（產業分類法未對齊），
# 由 test_industry_mode_blocked_on_tw50 專門驗證
TW50_ARCHITECTURES = tuple(a for a in VALID_ARCHITECTURES if a != "magnet_weak_industry")


@pytest.mark.parametrize("arch", TW50_ARCHITECTURES)
def test_all_architectures_tw50(arch: str, tw50_batch: dict) -> None:
    """架構在不對稱 universe 下都必須輸出 n_l2 欄。"""
    y_hat, _ = build_model(_cfg("tw50", arch)).eval()(tw50_batch)
    assert y_hat.shape == (2, 50)


# ── MAGNET：p(j) 對齊 ──────────────────────────────────────────

def test_magnet_extras_shapes_tw50(tw50_batch: dict) -> None:
    """h_L1 停留在 L1 的索引空間，其餘都在 L2。"""
    m = MAGNET(_cfg("tw50")).eval()
    d = m.head.mlp[0].in_features
    _, ex = m(tw50_batch)
    assert ex["h_L1"].shape   == (2, 30, d)
    assert ex["h_L2"].shape   == (2, 50, d)
    assert ex["h_fused"].shape == (2, 50, d)
    assert ex["alpha"].shape  == (2, 50, 1)
    assert ex["gate"].shape   == (2, 50, d)


def test_augment_weak_gathers_pair_index() -> None:
    """
    恆等項必須是 h1_{p(j)}，無配對者為零向量。

    這是 E6 的核心：k7 下 p(j)=j，gather 退化成恆等；擴充後 43/50
    的 TW 節點沒有第一項。若這裡對錯了，模型仍然會跑，只是每個
    TW 節點吃到不相干公司的美股訊號。
    """
    u = load_universe("tw50")
    m = MAGNET(_cfg("tw50")).eval()
    h_L1 = torch.randn(2, u.n_l1, 32)
    out = m._augment_weak(h_L1)                       # weak_mode=None → 只有恆等項
    assert out.shape == (2, u.n_l2, 32)

    n_paired = 0
    for j, i in enumerate(u.pair_index):
        if i >= 0:
            assert torch.equal(out[:, j], h_L1[:, i]), f"TW {u.tw_nodes[j]} 取到錯的 ADR"
            n_paired += 1
        else:
            assert torch.all(out[:, j] == 0), f"TW {u.tw_nodes[j]} 無配對卻有恆等項"
    assert n_paired == 7
    assert n_paired < u.n_l2, "tw50 應有無配對節點，否則這個測試沒測到東西"


def test_augment_weak_is_identity_at_k7() -> None:
    """k7 下 gather 必須逐位元退化為恆等——凍結基準依賴這點。"""
    m = MAGNET(_cfg("k7")).eval()
    h_L1 = torch.randn(3, 7, 32)
    assert torch.equal(m._augment_weak(h_L1), h_L1)


# ── weak_mask：候選邊排除恆等邊 ────────────────────────────────

def test_weak_mask_shape_and_excludes_identity() -> None:
    u = load_universe("tw50")
    cfg = _cfg("tw50")
    cfg["model"] = {**cfg["model"], "weak_links": {"mode": "free"}}
    m = MAGNET(cfg)
    assert m.weak_mask.shape == (u.n_l1, u.n_l2)
    assert m.weak_beta.shape == (u.n_l1, u.n_l2)
    for j, i in enumerate(u.pair_index):
        if i >= 0:
            assert m.weak_mask[i, j] == 0, "恆等邊不得同時是候選邊"
    assert m.weak_mask.sum() == u.n_l1 * u.n_l2 - u.n_pairs


def test_weak_mask_k7_free_is_off_diagonal() -> None:
    cfg = _cfg("k7")
    cfg["model"] = {**cfg["model"], "weak_links": {"mode": "free"}}
    m = MAGNET(cfg)
    assert torch.equal(m.weak_mask, (~torch.eye(7, dtype=torch.bool)).float())


def test_weak_mask_industry_is_subset_of_free_k7() -> None:
    masks = {}
    for mode in ("free", "industry"):
        cfg = _cfg("k7")
        cfg["model"] = {**cfg["model"], "weak_links": {"mode": mode}}
        masks[mode] = MAGNET(cfg).weak_mask
    assert torch.all(masks["industry"] <= masks["free"])
    assert masks["industry"].sum() == 12, "k7 同產業候選邊為半導體 4×3"


def test_industry_mode_blocked_on_tw50() -> None:
    """
    tw50 兩側產業別分屬不同分類法（US 用 eda/fabless… 與 k7 手寫中文，
    TW 用 MOPS 的「半導體業」等 18 類），字串永遠不相等 → 遮罩全零。

    遮罩全零 + beta init 0 表示 magnet_weak_industry 會與一般 MAGNET
    完全等價，訓練跑得出來、結果一模一樣，會被誤讀成「同產業弱連結
    無效」。必須擋在建構階段。
    """
    cfg = _cfg("tw50")
    cfg["model"] = {**cfg["model"], "weak_links": {"mode": "industry"}}
    with pytest.raises(ValueError, match="產生 0 條候選邊"):
        MAGNET(cfg)


# ── universe 不一致必須 fail fast ──────────────────────────────

def test_magnet_rejects_mismatched_batch(tw50_batch: dict) -> None:
    """cfg 說 k7、batch 是 tw50——必須明確報錯而非索引越界。"""
    with pytest.raises(ValueError, match="與 cfg 的 universe"):
        MAGNET(_cfg("k7")).eval()(tw50_batch)


def test_align_loss_rejects_asymmetric_shapes() -> None:
    """不對稱 universe 下對角線不再是正樣本集合，須擋住而非算錯。"""
    crit = CombinedLoss(loss_cfg={"align": 1.0}, align_cfg={"enabled": True})
    with pytest.raises(ValueError, match="align loss 需要"):
        crit(y_hat=torch.randn(2, 50), y=torch.randn(2, 50),
             h_L1=torch.randn(2, 30, 32), h_L2=torch.randn(2, 50, 32))


# ── 其他模型的節點結構來源 ─────────────────────────────────────

def test_early_fusion_pair_src_follows_universe() -> None:
    from src.models.baseline_early_fusion import BaselineEarlyFusion
    u = load_universe("tw50")
    m = BaselineEarlyFusion(_cfg("tw50"))
    assert m.pair_src.numel() == u.n_l2
    assert int(m.has_pair.sum()) == u.n_pairs
    for j, i in enumerate(u.pair_index):
        if i >= 0:
            assert int(m.pair_src[j]) == i


def test_hats_sectors_follow_tw_nodes() -> None:
    from src.models.baseline_hats import BaselineHATS
    u = load_universe("tw50")
    m = BaselineHATS(_cfg("tw50"))
    assert m.stock2sector.numel() == u.n_l2
    assert m.n_sectors == len({u.industry[c] for c in u.tw_nodes})
    assert m.n_sectors > 4, "tw50 的產業別應遠多於 k7 的 4 類"


# ── 跨市場資訊的觸及範圍（結構不變量）─────────────────────────

def _adr_reach(arch: str, batch: dict) -> tuple[int, float, float]:
    """
    擾動整個 x_seq_L1，回傳 (有反應的節點數, 配對節點平均|Δ|, 無配對節點平均|Δ|)。

    這是純結構檢定：不依賴訓練、不依賴任何 p 值，量的是「ADR 輸入到各個
    TW 輸出之間有沒有前向路徑」。
    """
    u = load_universe("tw50")
    paired = [j for j, i in enumerate(u.pair_index) if i >= 0]
    unpaired = [j for j, i in enumerate(u.pair_index) if i < 0]
    torch.manual_seed(0)
    m = build_model(_cfg("tw50", arch)).eval()
    with torch.no_grad():
        y0, _ = m(batch)
        y1, _ = m({**batch, "x_seq_L1": torch.randn_like(batch["x_seq_L1"])})
    d = (y1 - y0).abs().mean(dim=0)
    return int((d > 1e-9).sum()), float(d[paired].mean()), float(d[unpaired].mean())


def test_late_fusion_cannot_reach_unpaired_nodes(tw50_batch: dict) -> None:
    """
    一般 MAGNET 的 ADR 資訊只到得了有恆等邊的節點。

    這不是缺陷測試而是現況的規格：L2 的圖跑在融合之前，融合是 per-node，
    之後沒有傳播，所以無配對節點在數值上完全不受 ADR 輸入影響。
    若哪天這條測試失敗，代表有人改動了融合的位置或順序。
    """
    n, paired, unpaired = _adr_reach("magnet", tw50_batch)
    assert n == 7, f"應只有 7 個配對節點有反應，實際 {n}"
    assert paired > 1e-6
    assert unpaired == 0.0, "無配對節點必須是位元零，不是「很小」"


def test_intermediate_fusion_reaches_all_nodes(tw50_batch: dict) -> None:
    """
    intermediate fusion 的存在理由：融合提前到 L2 圖之前，圖才能把跨市場
    訊號帶到無配對節點。這是該變體的主要交付物，與 IC 好壞無關。
    """
    n, paired, unpaired = _adr_reach("magnet_intermediate", tw50_batch)
    assert n == 50, f"全部 50 個節點都該有反應，實際 {n}"
    assert unpaired > 0.0
    assert unpaired / paired > 0.05, (
        f"無配對節點收到的訊號量僅為配對節點的 {unpaired/paired:.4f} 倍，"
        f"傳播雖然接通但幾乎沒有量"
    )


def test_intermediate_fusion_shapes(tw50_batch: dict, k7_batch: dict) -> None:
    from src.models.magnet_intermediate import MAGNETIntermediate
    for batch, n2 in ((k7_batch, 7), (tw50_batch, 50)):
        m = MAGNETIntermediate(_cfg("tw50" if n2 == 50 else "k7")).eval()
        y, ex = m(batch)
        assert y.shape == (2, n2)
        assert ex["h_out"].shape[1] == n2
        assert ex["gate"].shape[:2] == (2, n2)


def test_universe_from_cfg_defaults_to_k7() -> None:
    """舊的 config 快照沒有 data.universe，必須維持可重跑。"""
    assert universe_from_cfg({}).name == "k7"
    assert universe_from_cfg({"data": {}}).name == "k7"
