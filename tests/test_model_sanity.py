"""
test_model_sanity.py — M3 模型 sanity check
Corresponds to IMPLEMENTATION_SPEC §10 Step 5

驗證 MAGNET + MultiplexDataset 端到端可跑：
  - forward shape 正確（[B, n]）
  - 輸出無 NaN
  - backward 可微分（所有 trainable parameter grad 非 None）
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

# 確保 project root 在 sys.path（pytest 從 tests/ 目錄跑時）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.multiplex_dataset import (
    MultiplexDataset,
    multiplex_collate,
    N_NODES,
    F,
)
from src.dataset.config import PAIR_MAP
from src.dataset.multiplex_dataset import ADR_TICKERS, TW_CODES
from src.models import build_model
from src.models.baseline_early_fusion import BaselineEarlyFusion
from src.models.baseline_lstm import BaselineLSTM
from src.models.baseline_tw_gnn import BaselineTWGNN
from src.models.baseline_advalstm import BaselineAdvALSTM
from src.models.baseline_hats import BaselineHATS
from src.models.baseline_mansf import BaselineMANSF
from src.models.baseline_hgt import BaselineHGT
from src.models.baseline_deltalag import BaselineDeltaLag
from src.models.baseline_meig import BaselineMEIG
from src.models.multiplex_gnn import MAGNET


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="module")
def config(project_root: Path) -> dict:
    with open(project_root / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def train_dataset(project_root: Path) -> MultiplexDataset:
    # 用較小 T 避免 DataLoader 啟動慢；T=20 是 base.yaml 預設
    return MultiplexDataset(
        snapshot_dir=str(project_root / "data" / "graphs" / "snapshots"),
        features_dir=str(project_root / "data" / "features"),
        T=20,
        split="train",
        config_path=str(project_root / "configs" / "base.yaml"),
    )


@pytest.fixture(scope="module")
def model(config: dict) -> MAGNET:
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    return MAGNET(config)


@pytest.fixture(scope="module")
def small_batch(train_dataset: MultiplexDataset) -> dict:
    """組一個小 batch（B=4）作為通用測試輸入。"""
    samples = [train_dataset[i] for i in range(4)]
    return multiplex_collate(samples)


# ---------------------------------------------------------------------------
# 測試
# ---------------------------------------------------------------------------

def test_dataset_smoke(train_dataset: MultiplexDataset) -> None:
    """Dataset 基本健檢：長度合理、單筆 shape 正確、ticker 順序正確。"""
    assert len(train_dataset) > 0, "train split 不應為空"
    sample = train_dataset[0]

    # Shape 檢查
    assert sample["x_seq_L1"].shape == (20, N_NODES, F)
    assert sample["x_seq_L2"].shape == (20, N_NODES, F)
    assert sample["y"].shape == (N_NODES,)
    assert sample["edge_index_L1"].dim() == 2 and sample["edge_index_L1"].shape[0] == 2
    assert sample["edge_attr_L1"].shape[1] == 1
    assert sample["edge_index_L2"].dim() == 2 and sample["edge_index_L2"].shape[0] == 2
    assert sample["edge_attr_L2"].shape[1] == 1
    assert isinstance(sample["target_date"], str)

    # Ticker 順序固定（E5：get_ticker_order 改為實例方法，回傳該實例的 universe）
    adr_order, tw_order = train_dataset.get_ticker_order()
    assert len(adr_order) == N_NODES
    assert len(tw_order) == N_NODES
    assert adr_order == ["TSM", "UMC", "ASX", "CHT", "IMOS", "AUOTY", "HNHPF"]


def test_magnet_forward_shape(model: MAGNET, small_batch: dict) -> None:
    """forward 輸出 shape 必須 == [B, n]，且 extras 五個張量 shape 正確。"""
    model.eval()
    B = small_batch["x_seq_L1"].size(0)
    with torch.no_grad():
        y_hat, extras = model(small_batch)

    d_prime = model.head.mlp[0].in_features
    assert y_hat.shape == (B, N_NODES), f"y_hat shape 錯誤：{y_hat.shape}"
    assert extras["h_L1"].shape   == (B, N_NODES, d_prime)
    assert extras["h_L2"].shape   == (B, N_NODES, d_prime)
    assert extras["h_fused"].shape == (B, N_NODES, d_prime)
    assert extras["alpha"].shape   == (B, N_NODES, 1)
    assert extras["gate"].shape    == (B, N_NODES, d_prime)


def test_no_nan_in_output(model: MAGNET, small_batch: dict) -> None:
    """forward 輸出絕不可含 NaN（驗證資料前處理與模型初始化健康）。"""
    model.eval()
    with torch.no_grad():
        y_hat, extras = model(small_batch)
    assert torch.isfinite(y_hat).all(), "y_hat 含 NaN 或 inf"
    for key, t in extras.items():
        assert torch.isfinite(t).all(), f"extras[{key}] 含 NaN 或 inf"


def test_backward_pass(model: MAGNET, small_batch: dict) -> None:
    """loss.backward() 後所有「主路徑」trainable parameter grad 非 None 且為 finite。

    例外（已知未使用，故意設計）：
      - fusion.attn_mlp.*：per-node attention `alpha` 為 SPEC §4.1 規定的
        分析用張量，h_fused 公式（§4.2）只用 gate。alpha 預留給 §4.3
        Volatility Adaptive Weighting，MVP 階段 deferred，故此處不參與
        loss 計算、無 grad。
    """
    model.train()

    # forward
    y_hat, extras = model(small_batch)
    y = small_batch["y"]

    # 計算 combined loss
    loss, components = model.compute_loss(y_hat, y, extras)
    assert torch.isfinite(loss), f"loss 含 NaN：components={components}"

    # backward
    model.zero_grad()
    loss.backward()

    # SPEC §4.3 deferred：fusion.attn_mlp 在 MVP 中不參與主損失
    EXPECTED_UNUSED_PREFIXES = ("fusion.attn_mlp",)

    params_no_grad: list[str] = []
    params_nan_grad: list[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in EXPECTED_UNUSED_PREFIXES):
            continue   # 已知 deferred 模組，跳過
        if p.grad is None:
            params_no_grad.append(name)
        elif not torch.isfinite(p.grad).all():
            params_nan_grad.append(name)

    assert not params_no_grad, (
        f"以下主路徑 trainable parameter 沒有 grad：{params_no_grad[:5]} ..."
    )
    assert not params_nan_grad, (
        f"以下 parameter grad 含 NaN/inf：{params_nan_grad[:5]} ..."
    )


def test_dataloader_integration(train_dataset: MultiplexDataset, model: MAGNET) -> None:
    """DataLoader + collate_fn + MAGNET 端到端 1 個 batch 跑通。"""
    loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,            # walk-forward 訓練不要 shuffle（M4 也保持）
        collate_fn=multiplex_collate,
        num_workers=0,
    )
    batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        y_hat, _ = model(batch)
    assert y_hat.shape == (8, N_NODES)


# ---------------------------------------------------------------------------
# M6 Stage 0 — Ablation baseline smoke tests
# ---------------------------------------------------------------------------

def _smoke_forward_backward(model, batch: dict, extra_skip_prefixes: tuple = ()) -> None:
    """共用 helper：forward shape 對 + loss 可微分 + 主路徑 grad 非 None。

    Args:
        extra_skip_prefixes: 額外跳過的參數前綴。對 magnet_no_a12 而言，
                             gat_L1 / proj_L1 的 grad 預期為 None（ADR 路徑刻意切斷）。
    """
    B = batch["x_seq_L2"].size(0)
    model.train()
    y_hat, extras = model(batch)
    assert y_hat.shape == (B, N_NODES), f"y_hat shape 錯誤：{y_hat.shape}"
    assert torch.isfinite(y_hat).all(), "y_hat 含 NaN/inf"
    for key in ("h_L1", "h_L2", "h_fused"):
        assert key in extras, f"extras 缺少 {key}"

    loss, _ = model.compute_loss(y_hat, batch["y"], extras)
    assert torch.isfinite(loss), "loss 含 NaN"
    model.zero_grad()
    loss.backward()

    # MAGNET 已知 fusion.attn_mlp 在 MVP 未參與主損失（SPEC §4.3 deferred）
    skip = ("fusion.attn_mlp",) + tuple(extra_skip_prefixes)
    no_grad: list[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in skip):
            continue
        if p.grad is None:
            no_grad.append(name)
    assert not no_grad, f"以下參數無 grad：{no_grad[:5]} ..."


def test_baseline_lstm_smoke(config: dict, small_batch: dict) -> None:
    """BaselineLSTM（無圖、無 ADR）forward + backward 可跑。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineLSTM(config)
    _smoke_forward_backward(model, small_batch)


def test_baseline_tw_gnn_smoke(config: dict, small_batch: dict) -> None:
    """BaselineTWGNN（單層 TW GNN，無 ADR）forward + backward 可跑。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineTWGNN(config)
    _smoke_forward_backward(model, small_batch)


def test_magnet_no_a12_smoke(config: dict, small_batch: dict) -> None:
    """MAGNET + disable_a12（h_L1 強制零化）forward + backward 可跑。
    ADR 編碼路徑（gat_L1 / proj_L1）刻意切斷，預期無 grad。"""
    torch.manual_seed(config["training"]["seed"])
    cfg = {**config, "model": {**config["model"], "architecture": "magnet_no_a12"}}
    model = build_model(cfg)
    assert isinstance(model, MAGNET)
    assert model.disable_a12 is True
    # disable_a12=True 時 L1 端的 GAT / Projection 整段被切斷（h_L1 進 fusion 前歸零）
    _smoke_forward_backward(model, small_batch, extra_skip_prefixes=("gat_L1", "proj_L1"))


def test_build_model_dispatch(config: dict) -> None:
    """build_model 依 architecture 派對應 class，未知值要報錯。"""
    cfg_lstm = {**config, "model": {**config["model"], "architecture": "baseline_lstm"}}
    cfg_twgn = {**config, "model": {**config["model"], "architecture": "baseline_tw_gnn"}}
    cfg_mag  = {**config, "model": {**config["model"], "architecture": "magnet"}}
    cfg_adv  = {**config, "model": {**config["model"], "architecture": "adv_alstm"}}
    cfg_hats = {**config, "model": {**config["model"], "architecture": "hats"}}
    cfg_mansf = {**config, "model": {**config["model"], "architecture": "man_sf"}}
    cfg_hgt  = {**config, "model": {**config["model"], "architecture": "hgt"}}
    cfg_dl   = {**config, "model": {**config["model"], "architecture": "delta_lag"}}
    cfg_meig = {**config, "model": {**config["model"], "architecture": "meig"}}
    cfg_wf   = {**config, "model": {**config["model"], "architecture": "magnet_weak_free"}}
    cfg_wi   = {**config, "model": {**config["model"], "architecture": "magnet_weak_industry"}}
    cfg_bad  = {**config, "model": {**config["model"], "architecture": "nope"}}

    assert isinstance(build_model(cfg_lstm), BaselineLSTM)
    assert isinstance(build_model(cfg_twgn), BaselineTWGNN)
    assert isinstance(build_model(cfg_mag),  MAGNET)
    assert isinstance(build_model(cfg_adv),  BaselineAdvALSTM)
    assert isinstance(build_model(cfg_hats), BaselineHATS)
    assert isinstance(build_model(cfg_mansf), BaselineMANSF)
    assert isinstance(build_model(cfg_hgt), BaselineHGT)
    assert isinstance(build_model(cfg_dl),  BaselineDeltaLag)
    assert isinstance(build_model(cfg_meig), BaselineMEIG)
    m_wf = build_model(cfg_wf)
    m_wi = build_model(cfg_wi)
    assert isinstance(m_wf, MAGNET) and m_wf.weak_mode == "free"
    assert isinstance(m_wi, MAGNET) and m_wi.weak_mode == "industry"
    with pytest.raises(ValueError):
        build_model(cfg_bad)


# ---------------------------------------------------------------------------
# M7 External Baseline smoke tests
# ---------------------------------------------------------------------------

def test_adv_alstm_smoke(config: dict, small_batch: dict) -> None:
    """Adv-ALSTM (Feng 2019) forward + backward 可跑；train/eval 模式輸出差異合理。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineAdvALSTM(config)
    _smoke_forward_backward(model, small_batch)


def test_deltalag_smoke(config: dict, small_batch: dict) -> None:
    """DeltaLag (Zhou 2025) forward + backward 可跑；sparsified cross-attention 學 lead-lag。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineDeltaLag(config)
    _smoke_forward_backward(model, small_batch)


def test_deltalag_topk_selection(config: dict, small_batch: dict) -> None:
    """驗證 top-k selection 輸出合理的 leader index 和 lag value。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineDeltaLag(config)
    model.eval()
    with torch.no_grad():
        _, extras = model(small_batch)
    cand_idx = extras["topk_cand_idx"]      # [B, n, k]
    lag_vals = extras["topk_lag"]           # [B, n, k]

    n = small_batch["x_seq_L2"].size(2)
    # cand_idx 必須在 [0, 2n) 範圍
    assert (cand_idx >= 0).all() and (cand_idx < 2 * n).all(), (
        f"candidate index 超出範圍：{cand_idx.min()}..{cand_idx.max()}"
    )
    # lag values 必須在 [1, l_max] 範圍
    assert (lag_vals >= 1).all() and (lag_vals <= model.l_max).all(), (
        f"lag values 超出 [1, l_max={model.l_max}]：{lag_vals.min()}..{lag_vals.max()}"
    )
    # target i 不能選 candidate n+i (自己) 為 leader（self-loop 已 mask）
    for i in range(n):
        self_cand = n + i
        assert (cand_idx[:, i, :] != self_cand).all(), (
            f"target {i} 選到自己 (cand {self_cand}) 作 leader"
        )


def test_hgt_smoke(config: dict, small_batch: dict) -> None:
    """HGT (Hu 2020) forward + backward 可跑；heterogeneous graph with 4 edge types。
    預測僅用 'tw' node，最後一層 HGT 的 'adr' 輸出投影是 dead branch（預期無 grad）。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineHGT(config)
    # 最後一層 HGT 的 adr 輸出無 consumer（predictor 只吃 tw node）
    n_layers = len(model.hgt_layers)
    last = n_layers - 1
    _smoke_forward_backward(
        model, small_batch,
        extra_skip_prefixes=(
            f"hgt_layers.{last}.out_lin.lins.adr",
            f"hgt_layers.{last}.skip.adr",
        ),
    )


def test_mansf_smoke(config: dict, small_batch: dict) -> None:
    """MAN-SF (Sawhney 2020, no-text variant) forward + backward 可跑。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineMANSF(config)
    _smoke_forward_backward(model, small_batch)


def test_mansf_modality_attention_sums_to_1(config: dict, small_batch: dict) -> None:
    """驗證 modality attention 是合法 softmax（每個 stock 的 2 個模態權重和為 1）。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineMANSF(config)
    model.eval()
    with torch.no_grad():
        _, extras = model(small_batch)
    alpha = extras["modality_alpha"]                          # [B, n, 2]
    sums = alpha.sum(dim=-1)                                  # [B, n]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"modality attention 未 softmax normalize：{sums}"
    )


def test_hats_smoke(config: dict, small_batch: dict) -> None:
    """HATS (Kim 2019) forward + backward 可跑；sector 分層依 PAIR_MAP industry。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineHATS(config)
    _smoke_forward_backward(model, small_batch)


def test_hats_sector_mapping(config: dict, small_batch: dict) -> None:
    """驗證 sector 映射符合 PAIR_MAP 預期（TSM/UMC/ASX/IMOS → 半導體 idx=1）。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineHATS(config)
    stock2sector = model.stock2sector.tolist()
    # 依 config.py 的 ticker 順序 [TSM,UMC,ASX,CHT,IMOS,AUOTY,HNHPF]
    # 及 sorted sector [光電=0, 半導體=1, 電信=2, 電子=3]
    assert stock2sector == [1, 1, 1, 2, 1, 0, 3], (
        f"sector 映射非預期：{stock2sector}"
    )
    assert model.n_sectors == 4


def test_meig_smoke(config: dict, small_batch: dict) -> None:
    """MEIG-core (Bukhari 2025) forward + backward 可跑；3-branch GCN + CGAT。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineMEIG(config)
    _smoke_forward_backward(model, small_batch)


def test_meig_cgat_alpha_is_softmax(config: dict, small_batch: dict) -> None:
    """驗證 CGAT 權重是合法 softmax（3 個 branch 權重和為 1、皆為正）。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineMEIG(config)
    model.eval()
    with torch.no_grad():
        _, extras = model(small_batch)
    alpha = extras["cgat_alpha"]                              # [3]
    assert alpha.shape == (3,), f"cgat_alpha shape 錯誤：{alpha.shape}"
    assert torch.allclose(alpha.sum(), torch.tensor(1.0), atol=1e-5), (
        f"CGAT 權重未 softmax normalize：{alpha}"
    )
    assert (alpha > 0).all(), f"CGAT 權重應皆為正：{alpha}"


def test_meig_graph_block_separation(config: dict, small_batch: dict) -> None:
    """驗證三張圖的區塊分離：intra 圖不含跨市場邊、inter 圖只含跨市場邊。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineMEIG(config)
    model.eval()
    with torch.no_grad():
        model(small_batch)                                    # 觸發 mask 建立

    n = small_batch["x_seq_L1"].size(2)
    # intra_L1 mask 僅允許 ADR 區塊（前 n 個節點）
    assert not model.intra1_mask[:, n:].any(), "intra_L1 mask 含 TW 節點邊"
    assert not model.intra1_mask[n:, :].any(), "intra_L1 mask 含 TW 節點邊"
    # intra_L2 mask 僅允許 TW 區塊（後 n 個節點）
    assert not model.intra2_mask[:, :n].any(), "intra_L2 mask 含 ADR 節點邊"
    assert not model.intra2_mask[:n, :].any(), "intra_L2 mask 含 ADR 節點邊"
    # inter mask 僅允許跨區塊（無同市場邊、無對角線）
    assert not model.inter_mask[:n, :n].any(), "inter mask 含 ADR 同市場邊"
    assert not model.inter_mask[n:, n:].any(), "inter mask 含 TW 同市場邊"
    assert not model.inter_mask.diagonal().any(), "inter mask 含 self-loop"


# ---------------------------------------------------------------------------
# M8 Hierarchical A12 — weak-link tier smoke tests
# ---------------------------------------------------------------------------

def _build_weak(config: dict, arch: str) -> MAGNET:
    cfg = {**config, "model": {**config["model"], "architecture": arch}}
    return build_model(cfg)


def test_magnet_weak_free_smoke(config: dict, small_batch: dict) -> None:
    """MAGNET + 自由弱連結（42 條候選）forward + backward 可跑，含 L1 懲罰路徑。"""
    torch.manual_seed(config["training"]["seed"])
    model = _build_weak(config, "magnet_weak_free")
    _smoke_forward_backward(model, small_batch)


def test_magnet_weak_industry_smoke(config: dict, small_batch: dict) -> None:
    """MAGNET + 產業遮罩弱連結（12 條候選）forward + backward 可跑。"""
    torch.manual_seed(config["training"]["seed"])
    model = _build_weak(config, "magnet_weak_industry")
    _smoke_forward_backward(model, small_batch)


def test_weak_mask_structure(config: dict) -> None:
    """驗證弱連結遮罩：free = 42 條 off-diagonal；industry = 半導體區塊 12 條。"""
    torch.manual_seed(config["training"]["seed"])
    m_free = _build_weak(config, "magnet_weak_free")
    m_ind  = _build_weak(config, "magnet_weak_industry")

    n = m_free.weak_mask.size(0)
    # free：全 off-diagonal
    assert m_free.weak_mask.sum().item() == n * n - n            # 42
    assert m_free.weak_mask.diagonal().sum().item() == 0, "對角線屬強連結層，必須排除"

    # industry：依 PAIR_MAP 順序 [TSM,UMC,ASX,CHT,IMOS,AUOTY,HNHPF]，
    # 半導體 = {0,1,2,4}（4×4−4=12），其餘產業為單一成員 → 無弱連結候選
    semi = {0, 1, 2, 4}
    expected = 0.0
    for i in range(n):
        for j in range(n):
            allowed = (i != j) and (i in semi) and (j in semi)
            expected += float(allowed)
            assert m_ind.weak_mask[i, j].item() == float(allowed), (
                f"industry mask[{i},{j}] 非預期"
            )
    assert m_ind.weak_mask.sum().item() == expected == 12


def test_weak_beta_init_is_identity_to_magnet(config: dict, small_batch: dict) -> None:
    """beta init 0 → 弱連結變體在未訓練時輸出必須與原 MAGNET 完全相同
    （殘差式結構學習的起點保證）。"""
    seed = config["training"]["seed"]

    torch.manual_seed(seed)
    base = MAGNET(config)
    torch.manual_seed(seed)
    weak = _build_weak(config, "magnet_weak_free")   # weak_beta=zeros 不消耗 RNG

    base.eval()
    weak.eval()
    with torch.no_grad():
        y_base, _ = base(small_batch)
        y_weak, extras = weak(small_batch)

    assert torch.allclose(y_base, y_weak, atol=1e-7), (
        "beta=0 時弱連結變體應與原 MAGNET 輸出一致"
    )
    assert extras["weak_beta"].abs().sum().item() == 0.0


def test_weak_l1_penalty_in_loss(config: dict, small_batch: dict) -> None:
    """驗證 L1 稀疏懲罰進入 compute_loss：beta 非零時 loss 增加 lambda*|beta|。"""
    torch.manual_seed(config["training"]["seed"])
    model = _build_weak(config, "magnet_weak_free")
    model.train()
    y_hat, extras = model(small_batch)

    # beta = 0：weak_l1 分量應為 0
    loss0, comps0 = model.compute_loss(y_hat, small_batch["y"], extras)
    assert comps0["weak_l1"] == 0.0

    # 手動設 beta 後懲罰應等於 lambda * |beta*mask|.sum()
    with torch.no_grad():
        model.weak_beta.fill_(0.1)
    loss1, comps1 = model.compute_loss(y_hat, small_batch["y"], extras)
    expected_l1 = (model.weak_beta * model.weak_mask).abs().sum().item()
    assert abs(comps1["weak_l1"] - expected_l1) < 1e-6
    assert abs((loss1 - loss0).item() - model.weak_lambda * expected_l1) < 1e-6


def test_adv_alstm_adversarial_toggle(config: dict, small_batch: dict) -> None:
    """驗證 FGSM-approx adversarial 分支在 training 模式下確實會被觸發。

    停用 dropout 隔離 adversarial 效應：在 eval 模式下 y_hat_adv 分支不執行，
    輸出等於純 clean forward；train 模式下輸出是 (clean + λ·adv) 合成，
    兩者必須不同才能證明 adversarial 分支不是 dead code。
    """
    torch.manual_seed(config["training"]["seed"])
    model = BaselineAdvALSTM(config)

    # 停用 dropout 讓 train / eval 差異只由 adversarial 分支造成
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    model.eval()
    with torch.no_grad():
        y_eval, _ = model(small_batch)

    model.train()
    with torch.no_grad():
        y_train, _ = model(small_batch)

    assert not torch.allclose(y_eval, y_train, atol=1e-6), (
        "train / eval 模式輸出相同，adversarial perturbation 未被啟用"
    )


# ---------------------------------------------------------------------------
# 第 0 階段：early-fusion 對照組
# ---------------------------------------------------------------------------

def test_baseline_early_fusion_smoke(config: dict, small_batch: dict) -> None:
    """BaselineEarlyFusion（輸入層拼接 ADR）forward + backward 可跑。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineEarlyFusion(config)
    _smoke_forward_backward(model, small_batch)


def test_early_fusion_dispatch(config: dict) -> None:
    """build_model 能派給 baseline_early_fusion。"""
    cfg = {**config, "model": {**config["model"],
                               "architecture": "baseline_early_fusion"}}
    assert isinstance(build_model(cfg), BaselineEarlyFusion)


def test_early_fusion_pairing_map(config: dict) -> None:
    """p(j) 需由 PAIR_MAP 推得，而非假設 L1/L2 索引對齊。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineEarlyFusion(config)

    assert model.pair_src.numel() == len(TW_CODES)
    for j, tw_code in enumerate(TW_CODES):
        src = int(model.pair_src[j])
        if bool(model.has_pair[j]):
            adr_ticker = ADR_TICKERS[src]
            assert PAIR_MAP[adr_ticker]["tw"] == tw_code, (
                f"TW {tw_code} 被錯配到 ADR {adr_ticker}"
            )
        else:
            assert tw_code not in {v["tw"] for v in PAIR_MAP.values()}


def test_early_fusion_lstm_input_dim_is_doubled(config: dict) -> None:
    """拼接後 LSTM 輸入維度須為 2F，且 F 仍由 config 提供。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineEarlyFusion(config)
    F_base = config["model"]["lstm"]["input_dim"]
    assert model.lstm.lstm.input_size == F_base * 2


def test_early_fusion_actually_uses_adr(config: dict, small_batch: dict) -> None:
    """擾動 x_seq_L1 必須改變輸出——證明 ADR 半邊不是 dead code。

    這是本對照組存在的前提：它必須真的在用 ADR 資訊，
    否則它退化成 baseline_lstm，對照就沒有意義。
    """
    torch.manual_seed(config["training"]["seed"])
    model = BaselineEarlyFusion(config)
    model.eval()

    with torch.no_grad():
        y_ref, _ = model(small_batch)
        perturbed = {**small_batch,
                     "x_seq_L1": small_batch["x_seq_L1"] + 1.0}
        y_perturbed, _ = model(perturbed)

    assert not torch.allclose(y_ref, y_perturbed, atol=1e-6), (
        "擾動 ADR 輸入後預測不變，early-fusion 未實際使用 ADR 特徵"
    )


# ---------------------------------------------------------------------------
# 階段 A-7：耦合前的橫截面去均值（cs_demean）
# ---------------------------------------------------------------------------

def _magnet_with_demean(config: dict, l1: bool, l2: bool) -> MAGNET:
    """用同一顆種子建模型，只有 cs_demean 旗標不同。"""
    cfg = copy.deepcopy(config)
    cfg["model"]["cs_demean"] = {"l1": l1, "l2": l2}
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    return MAGNET(cfg)


def test_cs_demean_defaults_off(config: dict) -> None:
    """base.yaml 不含 cs_demean 區塊時兩個旗標都必須是 False。

    這條在守既有結果：只要預設變成 True，全部歷史 run 與 k7 凍結基準
    的數值都會改變，而那是靜默發生的。
    """
    torch.manual_seed(config["training"]["seed"])
    m = MAGNET(config)
    assert m.cs_demean_l1 is False
    assert m.cs_demean_l2 is False


def test_cs_demean_off_is_bit_identical(config: dict, small_batch: dict) -> None:
    """顯式關閉與未設定必須位元相同——關閉時不可有任何額外運算。"""
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    m_default = MAGNET(copy.deepcopy(config)).eval()
    m_off = _magnet_with_demean(config, False, False).eval()
    with torch.no_grad():
        y_default, _ = m_default(small_batch)
        y_off, _ = m_off(small_batch)
    assert torch.equal(y_default, y_off)


def test_cs_demean_zeroes_node_mean(config: dict, small_batch: dict) -> None:
    """開啟後，被處理那一層的逐維節點平均必須為 0（浮點誤差內）。"""
    m = _magnet_with_demean(config, True, True).eval()
    with torch.no_grad():
        _, extras = m(small_batch)
    for key in ("h_L1", "h_L2"):
        node_mean = extras[key].mean(dim=-2)          # [B, d']
        assert node_mean.abs().max() < 1e-5, (
            f"{key} 去均值後節點平均應為 0，實際最大 {node_mean.abs().max():.2e}"
        )


def test_cs_demean_is_per_layer(config: dict, small_batch: dict) -> None:
    """只開 l1 時 L2 必須完全不受影響，反之亦然。

    兩個旗標若不小心共用同一個分支，這條會抓到。
    """
    m_off = _magnet_with_demean(config, False, False).eval()
    m_l1 = _magnet_with_demean(config, True, False).eval()
    m_l2 = _magnet_with_demean(config, False, True).eval()
    with torch.no_grad():
        _, ex_off = m_off(small_batch)
        _, ex_l1 = m_l1(small_batch)
        _, ex_l2 = m_l2(small_batch)
    assert torch.equal(ex_l1["h_L2"], ex_off["h_L2"]), "只開 l1 卻動到了 h_L2"
    assert torch.equal(ex_l2["h_L1"], ex_off["h_L1"]), "只開 l2 卻動到了 h_L1"
    assert not torch.equal(ex_l1["h_L1"], ex_off["h_L1"]), "開了 l1 但 h_L1 沒變"
    assert not torch.equal(ex_l2["h_L2"], ex_off["h_L2"]), "開了 l2 但 h_L2 沒變"


def test_cs_demean_adds_no_parameters(config: dict) -> None:
    """去均值是無參數運算，開關不應改變參數量。"""
    n_off = sum(p.numel() for p in _magnet_with_demean(config, False, False).parameters())
    n_on = sum(p.numel() for p in _magnet_with_demean(config, True, True).parameters())
    assert n_off == n_on


# ---------------------------------------------------------------------------
# 階段 A-1+3：稠密可學耦合矩陣 A（per-node 載荷）
# ---------------------------------------------------------------------------

def _magnet_dense(config: dict, **coupling) -> MAGNET:
    cfg = copy.deepcopy(config)
    cfg["model"]["architecture"] = "magnet_dense_a"
    if coupling:
        cfg["model"]["coupling"] = coupling
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    return build_model(cfg)


def test_dense_a_defaults_off(config: dict) -> None:
    """未指定 coupling 時 MAGNET 不得建出 coupling_A。

    守的是既有結果：稠密 A 會改變所有歷史 run 與 k7 凍結基準的數值。
    """
    torch.manual_seed(config["training"]["seed"])
    m = MAGNET(config)
    assert m.coupling_mode is None
    assert not hasattr(m, "coupling_A")


def test_dense_a_param_count_and_init(config: dict) -> None:
    """A 為 [n1, n2]；配對位置 = init_paired，其餘 = init_other（預設 1/n1）。"""
    m = _magnet_dense(config)
    A = m.coupling_A.detach()
    assert A.shape == (m.n_l1, m.n_l2)
    assert A.numel() == m.n_l1 * m.n_l2
    for j in range(m.n_l2):
        if m.has_pair[j]:
            assert A[m.pair_src[j], j].item() == pytest.approx(1.0)
    # 找一個非配對位置驗 init_other
    off = [(i, j) for j in range(m.n_l2) for i in range(m.n_l1)
           if not (m.has_pair[j] and i == m.pair_src[j].item())]
    assert A[off[0]].item() == pytest.approx(1.0 / m.n_l1)


def test_dense_a_init_other_zero_matches_identity(config: dict, small_batch: dict) -> None:
    """init_other=0 時，A 只在配對位置為 1，耦合輸出必須等於現行的恆等邊路徑。

    這是 dense 模式的正確性錨點：它必須把現行行為含為特例，
    否則之後任何比較都分不清是「稠密邊有效」還是「實作換了語意」。
    """
    m_dense = _magnet_dense(config, init_other=0.0).eval()
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    m_ident = MAGNET(copy.deepcopy(config)).eval()          # coupling_mode=None
    d_prime = config["model"]["projection"]["d_prime"]
    h = torch.randn(2, m_dense.n_l1, d_prime)
    with torch.no_grad():
        out_dense = m_dense._augment_weak(h)
        out_ident = m_ident._augment_weak(h)
    assert torch.allclose(out_dense, out_ident, atol=1e-6), (
        f"最大差 {(out_dense - out_ident).abs().max():.2e}"
    )


def test_dense_a_conflicts_with_weak_links(config: dict) -> None:
    """dense 已涵蓋所有跨層邊，不可與 weak_links 併用，必須明確報錯。"""
    cfg = copy.deepcopy(config)
    cfg["model"]["architecture"] = "magnet_dense_a"
    cfg["model"]["weak_links"] = {"mode": "free"}
    with pytest.raises(ValueError, match="互斥"):
        build_model(cfg)


def test_dense_a_rejects_bad_mode(config: dict) -> None:
    cfg = copy.deepcopy(config)
    cfg["model"]["coupling"] = {"mode": "sparse"}
    with pytest.raises(ValueError, match="coupling.mode"):
        MAGNET(cfg)


# ---------------------------------------------------------------------------
# 階段 A-2a：原始特徵到耦合點的跳接（raw_skip）
# ---------------------------------------------------------------------------

def _magnet_skip(config: dict, **skip) -> MAGNET:
    cfg = copy.deepcopy(config)
    if skip:
        cfg["model"]["raw_skip"] = skip
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    return MAGNET(cfg)


def test_raw_skip_defaults_off(config: dict) -> None:
    """base.yaml 無 raw_skip 區塊時兩個旗標必須為 False，且不得建出跳接模組。"""
    torch.manual_seed(config["training"]["seed"])
    m = MAGNET(config)
    assert m.raw_skip_l1 is False and m.raw_skip_l2 is False
    assert not hasattr(m, "skip_L1") and not hasattr(m, "skip_L2")


def test_raw_skip_off_is_bit_identical(config: dict, small_batch: dict) -> None:
    """顯式關閉與未設定必須位元相同。"""
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    m_default = MAGNET(copy.deepcopy(config)).eval()
    m_off = _magnet_skip(config, l1=False, l2=False).eval()
    with torch.no_grad():
        y_default, _ = m_default(small_batch)
        y_off, _ = m_off(small_batch)
    assert torch.equal(y_default, y_off)


def test_raw_skip_zero_init_matches_baseline(config: dict, small_batch: dict) -> None:
    """zero_init=True 時跳接權重為 0，輸出必須等於未開啟跳接的模型。

    這是 2a 的正確性錨點：開啟後的模型必須把原行為含為起點，
    否則之後分不清效果來自跳接還是來自參數擾動。
    """
    m_zero = _magnet_skip(config, l1=True, l2=True, zero_init=True).eval()
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    m_base = MAGNET(copy.deepcopy(config)).eval()
    with torch.no_grad():
        y_zero, _ = m_zero(small_batch)
        y_base, _ = m_base(small_batch)
    assert torch.allclose(y_zero, y_base, atol=1e-6), (
        f"最大差 {(y_zero - y_base).abs().max():.2e}"
    )


def test_raw_skip_is_per_layer(config: dict, small_batch: dict) -> None:
    """只開 l1 時 h_L2 不得改變，反之亦然。"""
    m_off = _magnet_skip(config, l1=False, l2=False).eval()
    m_l1 = _magnet_skip(config, l1=True, l2=False).eval()
    m_l2 = _magnet_skip(config, l1=False, l2=True).eval()
    with torch.no_grad():
        _, ex_off = m_off(small_batch)
        _, ex_l1 = m_l1(small_batch)
        _, ex_l2 = m_l2(small_batch)
    assert torch.equal(ex_l1["h_L2"], ex_off["h_L2"]), "只開 l1 卻動到了 h_L2"
    assert torch.equal(ex_l2["h_L1"], ex_off["h_L1"]), "只開 l2 卻動到了 h_L1"
    assert not torch.equal(ex_l1["h_L1"], ex_off["h_L1"]), "開了 l1 但 h_L1 沒變"


def test_raw_skip_respects_feature_subset(config: dict) -> None:
    """跳接吃的欄位必須與 SharedLSTM 相同，否則等於偷偷多給資訊。"""
    cfg = copy.deepcopy(config)
    cfg["model"]["lstm"]["feature_subset"] = ["log_return", "RSI_14"]
    cfg["model"]["raw_skip"] = {"l1": True}
    torch.manual_seed(cfg["training"]["seed"])
    m = MAGNET(cfg)
    assert m.lstm.in_size == 2
    assert m.skip_L1[1].in_features == 2, "跳接的輸入維度未跟隨 feature_subset"
    x = torch.randn(2, 5, m.n_l1, cfg["model"]["lstm"]["input_dim"])
    assert m._raw_last(x).shape == (2, m.n_l1, 2)


# ---------------------------------------------------------------------------
# 階段 A-2a'：拼接式跳接（raw_skip mode=concat）
# ---------------------------------------------------------------------------

def test_raw_skip_concat_keeps_d_prime(config: dict, small_batch: dict) -> None:
    """concat 模式下 proj 讓出 concat_dim 維，h_L1 / h_L2 仍是 d'。

    這條在守 fusion 的介面：只要總維度不變，fusion 與 head 就不用改，
    也不強制 L1 與 L2 同時開啟。
    """
    d_prime = config["model"]["projection"]["d_prime"]
    m = _magnet_skip(config, l1=True, mode="concat").eval()
    assert m.proj_L1.linear.out_features == d_prime - m.raw_concat_dim
    assert m.proj_L2.linear.out_features == d_prime, "只開 l1 不該動到 proj_L2"
    with torch.no_grad():
        _, extras = m(small_batch)
    assert extras["h_L1"].shape[-1] == d_prime
    assert extras["h_L2"].shape[-1] == d_prime


def test_raw_skip_concat_tail_is_raw(config: dict, small_batch: dict) -> None:
    """拼接後的最後 concat_dim 維必須就是跳接的輸出，不得被其他運算污染。"""
    m = _magnet_skip(config, l1=True, mode="concat").eval()
    with torch.no_grad():
        _, extras = m(small_batch)
        direct = m.skip_L1(m._raw_last(small_batch["x_seq_L1"]))
    assert torch.equal(extras["h_L1"][..., -m.raw_concat_dim:], direct)


def test_raw_skip_concat_default_dim_is_in_size(config: dict) -> None:
    """未指定 concat_dim 時取 SharedLSTM 的 in_size：原始特徵幾維就給幾維。"""
    cfg = copy.deepcopy(config)
    cfg["model"]["lstm"]["feature_subset"] = ["log_return", "RSI_14"]
    cfg["model"]["raw_skip"] = {"l1": True, "mode": "concat"}
    torch.manual_seed(cfg["training"]["seed"])
    m = MAGNET(cfg)
    assert m.raw_concat_dim == 2 == m.lstm.in_size


def test_raw_skip_concat_rejects_bad_dim(config: dict) -> None:
    """concat_dim 必須落在 (0, d_prime)，否則 proj 的輸出維度會非法。"""
    d_prime = config["model"]["projection"]["d_prime"]
    for bad in (0, d_prime, d_prime + 1):
        cfg = copy.deepcopy(config)
        cfg["model"]["raw_skip"] = {"l1": True, "mode": "concat", "concat_dim": bad}
        with pytest.raises(ValueError, match="concat_dim"):
            MAGNET(cfg)


def test_raw_skip_concat_rejects_zero_init(config: dict) -> None:
    """concat 改變了 proj 的輸出維度，歸零跳接也還原不了原模型，須明確擋掉。"""
    cfg = copy.deepcopy(config)
    cfg["model"]["raw_skip"] = {"l1": True, "mode": "concat", "zero_init": True}
    with pytest.raises(ValueError, match="zero_init"):
        MAGNET(cfg)


def test_raw_skip_rejects_bad_mode(config: dict) -> None:
    cfg = copy.deepcopy(config)
    cfg["model"]["raw_skip"] = {"l1": True, "mode": "cat"}
    with pytest.raises(ValueError, match="raw_skip.mode"):
        MAGNET(cfg)


def test_raw_skip_norm_default_is_batchnorm(config: dict) -> None:
    """預設正規化必須是逐特徵跨節點（_FeatureNorm），不是 LayerNorm。

    LayerNorm 正規化特徵軸，F=3 時把 3 個數字壓到剩 1 個自由度，
    橫截面差異被刪掉——實測跳接路徑 raw +0.0874 -> +0.0332。
    階段 A-2a / 2a' 兩次實驗因此都不算數，這條測試防止再犯。
    """
    from src.models.multiplex_gnn import _FeatureNorm
    m = _magnet_skip(config, l1=True)
    assert m.raw_skip_norm == "batchnorm"
    assert isinstance(m.skip_L1[0], _FeatureNorm)


def test_feature_norm_preserves_cross_node_spread(config: dict) -> None:
    """_FeatureNorm 必須保留節點之間的差異，LayerNorm 則會抹掉。

    用一組尺度差很大、但節點間有明確排序的輸入：正規化後
    逐特徵的節點排序必須不變。
    """
    from src.models.multiplex_gnn import _FeatureNorm
    x = torch.tensor([[[1.0, 50.0, 0.5],
                       [2.0, 55.0, 0.7],
                       [0.5, 45.0, 0.3]]])            # [1, 3 nodes, 3 feat]
    fn = _FeatureNorm(3)
    fn.train()
    out = fn(x)
    for f in range(3):
        assert torch.equal(out[0, :, f].argsort(), x[0, :, f].argsort()), (
            f"特徵 {f} 的節點排序被改變了"
        )
    # 對照：LayerNorm 會讓三個節點的輸出幾乎相同
    ln_out = torch.nn.LayerNorm(3, elementwise_affine=False)(x)
    spread_fn = out[0].std(dim=0).mean()
    spread_ln = ln_out[0].std(dim=0).mean()
    assert spread_fn > spread_ln * 5, (
        f"_FeatureNorm 的節點間離散度 {spread_fn:.4f} 應遠大於 LayerNorm 的 {spread_ln:.4f}"
    )


def test_raw_skip_norm_rejects_bad_value(config: dict) -> None:
    cfg = copy.deepcopy(config)
    cfg["model"]["raw_skip"] = {"l1": True, "norm": "groupnorm"}
    with pytest.raises(ValueError, match="raw_skip.norm"):
        MAGNET(cfg)


# ---------------------------------------------------------------------------
# beta 層（階段 P1）
# ---------------------------------------------------------------------------

def _beta_cfg(**wl):
    """在 tw50 的 magnet_weak_free 設定上覆寫 weak_links。"""
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs" / "tw50.yaml"))
    cfg["model"]["architecture"] = "magnet_weak_free"
    cfg["model"]["lstm"]["T_history"] = 1
    cfg["model"]["gat"]["num_layers"] = 1
    cfg["model"]["weak_links"] = {**(cfg["model"].get("weak_links") or {}), **wl}
    return cfg


def test_beta_layer_degenerates_to_current_model():
    """beta_init_std=0、B 零初始化時，耦合必須逐位元等於現行 MAGNET。

    這是既有 run 不受影響的保證：新旋鈕預設關閉，而且就算打開、
    把離散度設為 0，數值也不能動。
    """
    import torch
    from src.models import build_model
    torch.manual_seed(42)
    m0 = build_model(_beta_cfg(beta_layer=False))
    torch.manual_seed(42)
    m1 = build_model(_beta_cfg(beta_layer=True, beta_init_std=0.0,
                               beta_alpha_mean=1.0, beta_gamma_mean=0.0,
                               beta_b_init_std=0.0))
    m0.eval(); m1.eval()
    d = m0.proj_L1.linear.out_features if hasattr(m0.proj_L1, 'linear') else 32
    x = torch.randn(3, m0.n_l1, d)
    with torch.no_grad():
        diff = (m0._augment_weak(x) - m1._augment_weak(x)).abs().max().item()
    assert diff == 0.0, f"beta 層在退化設定下應位元相同，實得 max|diff|={diff:.3e}"
    assert bool((m1.beta_alpha == 1).all()), "alpha 初始應全為 1"
    assert bool((m1.beta_gamma == 0).all()), "gamma 初始應全為 0"


def test_beta_layer_is_a_reparametrisation():
    """gamma_j = colsum_j(B) 時，beta 層與原式等價。

    這說明 beta 層做的事是把「因子暴露」從 B 的欄和裡解耦出來——
    原式強制 gamma 等於 colsum，beta 層讓兩者各自自由。
    """
    import torch
    from src.models import build_model
    torch.manual_seed(0)
    m_new = build_model(_beta_cfg(beta_layer=True, beta_init_std=0.0))
    torch.manual_seed(0)
    m_old = build_model(_beta_cfg(beta_layer=False))
    with torch.no_grad():
        m_new.weak_beta.normal_(0.0, 0.05)
        m_old.weak_beta.copy_(m_new.weak_beta)
        m_new.beta_gamma.copy_((m_new.weak_beta * m_new.weak_mask).sum(0))
    m_new.eval(); m_old.eval()
    d = m_new.proj_L1.linear.out_features if hasattr(m_new.proj_L1, 'linear') else 32
    x = torch.randn(3, m_new.n_l1, d)
    with torch.no_grad():
        diff = (m_new._augment_weak(x) - m_old._augment_weak(x)).abs().max().item()
    assert diff < 1e-5, f"重參數化應等價，實得 max|diff|={diff:.3e}"


def test_beta_layer_dispersed_init():
    """初始離散度必須真的出現在參數上（P0 設計原則的實作保證）。"""
    import torch
    from src.models import build_model
    torch.manual_seed(42)
    m = build_model(_beta_cfg(beta_layer=True, beta_init_std=0.3))
    assert m.beta_alpha.std().item() > 0.1, "alpha 初始離散度過小"
    assert m.beta_gamma.std().item() > 0.1, "gamma 初始離散度過小"


def test_per_target_head_degenerates_when_off():
    """per_target=False 時參數與前向必須與原式位元相同（退化保證）。"""
    import torch
    from src.models import build_model
    c1 = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c2 = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c2.setdefault("model", {}).setdefault("prediction_head", {})["per_target"] = False
    torch.manual_seed(0); m1 = build_model(c1)
    torch.manual_seed(0); m2 = build_model(c2)
    assert all(torch.equal(a, b) for a, b in
               zip(m1.state_dict().values(), m2.state_dict().values()))
    assert not m1.head.per_target


def test_per_target_head_is_dispersed_and_shaped():
    """per_target=True：每檔一條讀出向量，且初始就跨股票分散（P0 原則）。"""
    import torch
    from src.models import build_model
    c = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c.setdefault("model", {}).setdefault("prediction_head", {})["per_target"] = True
    torch.manual_seed(0); m = build_model(c)
    assert m.head.per_target
    assert m.head.readout_w.shape[0] == m.n_l2
    assert m.head.readout_w.std(0).mean().item() > 0.01, "讀出向量初始離散度過小"
    d = m.proj_L1.linear.out_features if hasattr(m.proj_L1, "linear") else 32
    y = m.head(torch.randn(2, m.n_l2, d))
    assert tuple(y.shape) == (2, m.n_l2)


def test_per_target_head_reduces_to_shared_when_rows_tied():
    """所有 w_j 相同時，per_target 讀出等價於共享讀出（嚴格泛化的證明）。"""
    import torch
    from src.models import build_model
    c = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c.setdefault("model", {}).setdefault("prediction_head", {})["per_target"] = True
    torch.manual_seed(0); m = build_model(c)
    m.eval()
    with torch.no_grad():
        m.head.readout_w.copy_(m.head.readout_w[0].expand_as(m.head.readout_w))
        m.head.readout_b.fill_(0.25)
    d = m.proj_L1.linear.out_features if hasattr(m.proj_L1, "linear") else 32
    x = torch.randn(2, m.n_l2, d)
    with torch.no_grad():
        y = m.head(x)
        z = m.head.trunk(x)
        ref = (z @ m.head.readout_w[0]) + 0.25
    assert (y - ref).abs().max().item() < 1e-6


def test_multi_factor_degenerates_at_k1():
    """beta_n_factors=1 必須與缺鍵時位元相同（退化保證）。"""
    import torch
    from src.models import build_model
    c1 = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c2 = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c2["model"]["weak_links"]["beta_n_factors"] = 1
    torch.manual_seed(0); m1 = build_model(c1)
    torch.manual_seed(0); m2 = build_model(c2)
    assert all(torch.equal(a, b) for a, b in
               zip(m1.state_dict().values(), m2.state_dict().values()))
    assert not hasattr(m1, "beta_factor_w")


def test_multi_factor_first_factor_is_equal_weight():
    """第 1 個因子初始化必須恰為等權平均（= 現行 h̄₁），其餘分散。"""
    import torch
    from src.models import build_model
    c = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c["model"]["weak_links"]["beta_n_factors"] = 3
    torch.manual_seed(0); m = build_model(c)
    assert tuple(m.beta_factor_w.shape) == (3, m.n_l1)
    assert tuple(m.beta_gamma.shape) == (m.n_l2, 3)
    assert torch.allclose(m.beta_factor_w[0],
                          torch.full((m.n_l1,), 1.0 / m.n_l1))
    assert m.beta_factor_w[1:].std().item() > 0, "其餘因子必須分散初始化"


def test_multi_factor_matches_k1_when_only_first_factor_used():
    """只留第 1 個因子（其餘 gamma 歸零）時，輸出等於 k=1 的 gamma_j·h̄₁。"""
    import torch
    from src.models import build_model
    c = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c["model"]["weak_links"]["beta_n_factors"] = 3
    torch.manual_seed(0); m = build_model(c); m.eval()
    with torch.no_grad():
        m.beta_gamma[:, 1:] = 0.0
        g0 = m.beta_gamma[:, 0].clone()
        d = m.proj_L1.linear.out_features if hasattr(m.proj_L1, "linear") else 32
        x = torch.randn(2, m.n_l1, d)
        out = m._augment_weak(x)
        hbar = x.mean(dim=1, keepdim=True)
        ident = x.index_select(1, m.pair_src) * m.has_pair.view(1, -1, 1).float()
        ref = (ident * m.beta_alpha.view(1, -1, 1)
               + g0.view(1, -1, 1) * hbar
               + torch.einsum("ij,bid->bjd", m.weak_beta * m.weak_mask, x - hbar))
    assert (out - ref).abs().max().item() < 1e-5


def test_beta_rank_degenerates_at_zero():
    """beta_rank=0 必須與缺鍵時位元相同，且不建立 U/V。"""
    import torch
    from src.models import build_model
    c1 = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c2 = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c2["model"]["weak_links"]["beta_rank"] = 0
    torch.manual_seed(0); m1 = build_model(c1)
    torch.manual_seed(0); m2 = build_model(c2)
    assert all(torch.equal(a, b) for a, b in
               zip(m1.state_dict().values(), m2.state_dict().values()))
    assert hasattr(m1, "weak_beta") and not hasattr(m1, "weak_U")


def test_beta_rank_shapes_and_param_count():
    """r>0：B = U Vᵀ，秩為 r，參數量由 n1*n2 降為 (n1+n2)*r。"""
    import torch
    from src.models import build_model
    for r in (1, 3):
        c = _beta_cfg(beta_layer=True, beta_init_std=0.3)
        c["model"]["weak_links"]["beta_rank"] = r
        torch.manual_seed(0); m = build_model(c)
        assert tuple(m.weak_U.shape) == (m.n_l1, r)
        assert tuple(m.weak_V.shape) == (m.n_l2, r)
        assert not hasattr(m, "weak_beta")
        with torch.no_grad():
            m.weak_V.normal_(0.0, 0.1)
        assert torch.linalg.matrix_rank(m._weak_beta_full()).item() == r


def test_beta_rank_starts_at_zero_but_v_gets_gradient():
    """LoRA 式初始化：B 起點為零（與全秩版同），但 V 必須拿得到梯度。"""
    import torch
    from src.models import build_model
    c = _beta_cfg(beta_layer=True, beta_init_std=0.3)
    c["model"]["weak_links"]["beta_rank"] = 3
    torch.manual_seed(0); m = build_model(c)
    assert bool((m._weak_beta_full() == 0).all()), "B 起點必須為零"
    d = m.proj_L1.linear.out_features if hasattr(m.proj_L1, "linear") else 32
    m._augment_weak(torch.randn(2, m.n_l1, d)).sum().backward()
    assert m.weak_V.grad.abs().mean().item() > 0, "V 必須拿得到梯度"
