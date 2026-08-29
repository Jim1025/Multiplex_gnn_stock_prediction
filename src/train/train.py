"""
train.py — M4 主訓練迴圈 + CLI 入口
Corresponds to IMPLEMENTATION_SPEC §7 (training loop) / §8.4

使用：
    python -m src.train.train --epochs 3 --tag sanity
    python -m src.train.train --tag full

流程：
    1. 讀 config → set_seed(42) → get_device("auto")
    2. 載入 train / val / test 三個 DataLoader（multiplex_collate）
    3. 建立 MAGNET、Optimizer (Adam)
    4. MLflow start_run（log_params + per-epoch metrics）
    5. 訓練迴圈（含 grad_clip / early stopping on val IC）
    6. 訓練結束 → 載入 best ckpt → test set 一次性評估
    7. 上傳 best/final ckpt + val/test predictions.csv 進 MLflow artifacts

Early Stopping：monitor = val IC（越大越好）；patience = config.training.patience
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# MLflow 3.x 預設拒絕 file:// 後端；此專案明確選用本地 mlruns/，
# 故在 import 前設定 opt-out。可由環境變數覆寫。
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate
from src.models import VALID_ARCHITECTURES, build_model
from src.models.multiplex_gnn import MAGNET
from src.train.evaluator import evaluate
from src.train.losses import build_criterion
from src.train.utils import (
    batch_to_device,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# 工具：友善 run slug + INDEX.csv 維護
# ---------------------------------------------------------------------------

RUNS_ROOT = Path("runs")

INDEX_COLS = [
    "slug", "run_id", "tag", "start_time", "end_time", "status",
    "n_epochs", "best_epoch", "best_val_IC",
    "test_IC", "test_RankIC", "test_ICIR", "test_MSE",
]


def _make_run_slug(start_time_ms: int, tag: str, runs_root: Path) -> str:
    """{YYYYMMDD}_{HHMM}_{tag}[_${n}]，避開 runs_root 已存在的目錄名。"""
    dt = datetime.fromtimestamp(start_time_ms / 1000.0)
    base = f"{dt.strftime('%Y%m%d_%H%M')}_{tag}"
    slug = base
    i = 2
    runs_root.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in runs_root.iterdir() if p.is_dir()}
    while slug in existing:
        slug = f"{base}_{i}"
        i += 1
    return slug


def _iso(ms: int) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000.0).isoformat(timespec="seconds")


def _append_index_row(row: dict, runs_root: Path = RUNS_ROOT) -> None:
    """把一列 append 進 runs/INDEX.csv；不存在則先寫 header。"""
    runs_root.mkdir(parents=True, exist_ok=True)
    path = runs_root / "INDEX.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        if write_header:
            w.writeheader()
        # 只寫已知欄位，其餘忽略
        w.writerow({k: row.get(k, "") for k in INDEX_COLS})


# ---------------------------------------------------------------------------
# 工具：把巢狀 config 攤平成 mlflow.log_params 可吃的 flat dict
# ---------------------------------------------------------------------------

def _flatten(d: dict, parent: str = "", sep: str = ".") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# 單一 epoch：訓練
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     torch.nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device:    torch.device,
    grad_clip: float,
    log_every: int,
    epoch:     int,
) -> dict:
    model.train()

    total_sum    = 0.0
    mse_sum      = 0.0
    rank_sum     = 0.0
    align_sum    = 0.0
    variance_sum = 0.0
    n_samples    = 0

    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        y_hat, extras = model(batch)
        y = batch["y"]

        # 走 model.compute_loss：所有模型皆委派給內部 criterion（與外部
        # criterion 同 cfg 構建、行為等價），並允許模型附加自身的正則項
        # （M8 weak links 的 L1 稀疏懲罰）
        loss, comps = model.compute_loss(y_hat, y, extras)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        B = y.size(0)
        total_sum    += float(loss.item()) * B
        mse_sum      += comps["mse"]   * B
        rank_sum     += comps["rank"]  * B
        align_sum    += comps["align"] * B
        variance_sum += comps.get("variance", 0.0) * B
        n_samples    += B

        if (step + 1) % max(1, log_every) == 0:
            avg = total_sum / max(1, n_samples)
            print(
                f"  [epoch {epoch:03d} step {step+1:04d}/{len(loader):04d}] "
                f"loss_total={avg:.5f}  "
                f"(mse={comps['mse']:.5f} rank={comps['rank']:.5f} "
                f"align={comps['align']:.5f} var={comps.get('variance', 0.0):.5f})"
            )

    return {
        "loss_total":    total_sum    / max(1, n_samples),
        "loss_mse":      mse_sum      / max(1, n_samples),
        "loss_rank":     rank_sum     / max(1, n_samples),
        "loss_align":    align_sum    / max(1, n_samples),
        "loss_variance": variance_sum / max(1, n_samples),
        "epoch_time_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主訓練函數
# ---------------------------------------------------------------------------

def train(
    config_path: str = "configs/base.yaml",
    epochs:      Optional[int] = None,
    tag:         str = "full",
    mse_weight:        Optional[float] = None,
    rank_weight:       Optional[float] = None,
    variance_weight:   Optional[float] = None,
    lr_override:       Optional[float] = None,
    early_stop_metric: Optional[str]   = None,
    use_scheduler:     bool            = True,
    beta_n_factors:    Optional[int]   = None,
    per_target_head:   Optional[bool]  = None,
    architecture:      Optional[str]   = None,
    patience_override: Optional[int]   = None,
    seed_override:     Optional[int]   = None,
    smooth_window_override: Optional[int] = None,
    residual_alpha:    Optional[float] = None,
    cs_demean:         Optional[str]   = None,
    raw_skip:          Optional[str]   = None,
    raw_skip_mode:     Optional[str]   = None,
    input_norm:        Optional[str]   = None,
    input_norm_scope:  Optional[str]   = None,
    beta_layer:        Optional[bool]  = None,
    beta_init_std:     Optional[float] = None,
    beta_ablate:       Optional[str]   = None,
    graph_ablate:      Optional[str]   = None,
    optimizer_name:    Optional[str]   = None,
    coupling_init_other: Optional[float] = None,
    gat_layers:        Optional[int]   = None,
    lambda_sparse:     Optional[float] = None,
    t_history:         Optional[int]   = None,
    features:          Optional[list]  = None,
    save_every_epoch:  bool            = False,
) -> str:
    """
    主訓練流程。

    Args:
        config_path:       configs/base.yaml
        epochs:            若提供，覆寫 config.training.max_epochs（sanity 用）
        tag:               MLflow run name（"sanity" | "full" | "opt_pN_..." | ...）
        mse_weight:        若提供，覆寫 cfg.loss_weights.mse
        rank_weight:       若提供，覆寫 cfg.loss_weights.rank
        variance_weight:   若提供，覆寫 cfg.loss_weights.variance
        lr_override:       若提供，覆寫 cfg.training.lr
        early_stop_metric: 若提供（"IC"|"ICIR"），覆寫 cfg.training.early_stop_metric
        coupling_init_other: 若提供，覆寫 cfg.model.coupling.init_other（稠密 A 的非配對
                           位置初始值）。預設 None -> 程式內取 1/n1。只在
                           --architecture magnet_dense_a 下有作用。
        raw_skip_mode:     若提供，覆寫 cfg.model.raw_skip.mode（add / concat）。
        input_norm:        若提供，覆寫 cfg.model.lstm.input_norm（none / batchnorm）。
        optimizer_name:    若提供，覆寫 cfg.training.optimizer（adam / adamw）。
                           add 是 A-2a（已測，耦合點資訊未增加）；
                           concat 是 A-2a'，proj 讓出幾維給原始特徵，保留可分性。
        raw_skip:          若提供，覆寫 cfg.model.raw_skip。可為 none / l1 / l2 / both，
                           控制是否把原始特徵（最後一步）跳接到耦合點（階段 A-2a）。
        cs_demean:         若提供，覆寫 cfg.model.cs_demean。可為 none / l1 / l2 / both，
                           控制耦合前是否對節點維度去均值（階段 A-7）。
        residual_alpha:    若提供，覆寫 cfg.model.residual.alpha（magnet_intermediate
                           的 initial residual；掃 alpha 用，免得每個值開一份 config）
        gat_layers:        若提供，覆寫 cfg.model.gat.num_layers（跳數 = 層數；掃
                           過度平滑用。注意 num_layers=1 時最後一層強制
                           concat=False/heads=1，故 num_heads 失效）
        lambda_sparse:     若提供，覆寫 cfg.model.weak_links.lambda_sparse。
                           base.yaml / tw50.yaml 沒有 weak_links 區塊，直接用
                           --architecture magnet_weak_free 會靜默吃到程式預設的
                           1e-3；那個值在 tw50 下會把 1,493 條候選邊全部釘死
                           （資料梯度中位數 3.7e-04 < lambda），要 lambda=0
                           必須明講

    Returns:
        mlflow_run_id (str)
    """
    # ── 讀 config ──────────────────────────────────────────────────
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ── CLI 覆寫（None 表示不動；mutate cfg 讓下游 build_criterion / lr / 快照都生效）
    _overrides: list[str] = []
    if mse_weight is not None:
        cfg["loss_weights"]["mse"] = float(mse_weight)
        _overrides.append(f"mse={mse_weight}")
    if rank_weight is not None:
        cfg["loss_weights"]["rank"] = float(rank_weight)
        _overrides.append(f"rank={rank_weight}")
    if variance_weight is not None:
        cfg["loss_weights"]["variance"] = float(variance_weight)
        _overrides.append(f"variance={variance_weight}")
    if lr_override is not None:
        cfg["training"]["lr"] = float(lr_override)
        _overrides.append(f"lr={lr_override}")
    if patience_override is not None:
        cfg["training"]["patience"] = int(patience_override)
        _overrides.append(f"patience={patience_override}")
    if seed_override is not None:
        cfg["training"]["seed"] = int(seed_override)
        _overrides.append(f"seed={seed_override}")
    if residual_alpha is not None:
        # 覆寫在 config_snapshot.yaml 之前套用，快照裡看得到實際用的值
        cfg.setdefault("model", {}).setdefault("residual", {})["alpha"] = float(residual_alpha)
        _overrides.append(f"residual_alpha={residual_alpha}")
    if coupling_init_other is not None:
        # 覆寫在 config_snapshot.yaml 之前套用，快照裡看得到實際用的值
        cfg.setdefault("model", {}).setdefault("coupling", {})["init_other"] = float(coupling_init_other)
        _overrides.append(f"coupling_init_other={coupling_init_other}")
    if raw_skip is not None:
        # 覆寫在 config_snapshot.yaml 之前套用，快照裡看得到實際用的值
        choices = {"none": (False, False), "l1": (True, False),
                   "l2": (False, True),    "both": (True, True)}
        if raw_skip not in choices:
            raise ValueError(
                f"--raw-skip 需為 {sorted(choices)} 之一，當前為 {raw_skip!r}")
        l1, l2 = choices[raw_skip]
        blk = cfg.setdefault("model", {}).setdefault("raw_skip", {})
        blk["l1"], blk["l2"] = l1, l2
        _overrides.append(f"raw_skip={raw_skip}")
    if raw_skip_mode is not None:
        if raw_skip_mode not in ("add", "concat"):
            raise ValueError(
                f"--raw-skip-mode 需為 add 或 concat，當前為 {raw_skip_mode!r}")
        cfg.setdefault("model", {}).setdefault("raw_skip", {})["mode"] = raw_skip_mode
        _overrides.append(f"raw_skip_mode={raw_skip_mode}")
    if input_norm is not None:
        # 覆寫在 config_snapshot.yaml 之前套用，快照裡看得到實際用的值
        if input_norm not in ("none", "batchnorm"):
            raise ValueError(
                f"--input-norm 需為 none 或 batchnorm，當前為 {input_norm!r}")
        cfg.setdefault("model", {}).setdefault("lstm", {})["input_norm"] = input_norm
        _overrides.append(f"input_norm={input_norm}")
    if input_norm_scope is not None:
        if input_norm_scope not in ("shared", "per_layer"):
            raise ValueError(
                f"--input-norm-scope 需為 shared 或 per_layer，"
                f"當前為 {input_norm_scope!r}")
        cfg.setdefault("model", {}).setdefault("lstm", {})["input_norm_scope"] = \
            input_norm_scope
        _overrides.append(f"input_norm_scope={input_norm_scope}")
    if beta_layer is not None:
        cfg.setdefault("model", {}).setdefault("weak_links", {})["beta_layer"] = \
            bool(beta_layer)
        _overrides.append(f"beta_layer={beta_layer}")
    if beta_init_std is not None:
        cfg.setdefault("model", {}).setdefault("weak_links", {})["beta_init_std"] = \
            float(beta_init_std)
        _overrides.append(f"beta_init_std={beta_init_std}")
    if beta_ablate is not None:
        # 消融：關掉 beta 層的某一項。identity=①、factor=②、residual=③
        keys = {"identity": "beta_use_identity", "factor": "beta_use_factor",
                "residual": "beta_use_residual"}
        blk = cfg.setdefault("model", {}).setdefault("weak_links", {})
        for name in beta_ablate.split(","):
            name = name.strip()
            if name not in keys:
                raise ValueError(
                    f"--beta-ablate 只接受 {sorted(keys)} 的逗號組合，"
                    f"當前為 {name!r}")
            blk[keys[name]] = False
        _overrides.append(f"beta_ablate={beta_ablate}")
    if graph_ablate is not None:
        ok = {"none", "empty_l1", "empty_l2", "empty_both",
              "rand_l1", "rand_l2", "rand_both"}
        if graph_ablate not in ok:
            raise ValueError(
                f"--graph-ablate 需為 {sorted(ok)} 之一，當前為 {graph_ablate!r}")
        cfg.setdefault("model", {}).setdefault("gat", {})["graph_ablate"] = graph_ablate
        _overrides.append(f"graph_ablate={graph_ablate}")
    if optimizer_name is not None:
        if optimizer_name not in ("adam", "adamw"):
            raise ValueError(
                f"--optimizer 需為 adam 或 adamw，當前為 {optimizer_name!r}")
        cfg.setdefault("training", {})["optimizer"] = optimizer_name
        _overrides.append(f"optimizer={optimizer_name}")
    if cs_demean is not None:
        # 覆寫在 config_snapshot.yaml 之前套用，快照裡看得到實際用的值。
        # 用字串而非兩個 bool 旗標：掃描時 --cs-demean both 比
        # --cs-demean-l1 --cs-demean-l2 少一個「只開了一半」的出錯面。
        choices = {"none": (False, False), "l1": (True, False),
                   "l2": (False, True),    "both": (True, True)}
        if cs_demean not in choices:
            raise ValueError(
                f"--cs-demean 需為 {sorted(choices)} 之一，當前為 {cs_demean!r}")
        l1, l2 = choices[cs_demean]
        cfg.setdefault("model", {})["cs_demean"] = {"l1": l1, "l2": l2}
        _overrides.append(f"cs_demean={cs_demean}")
    if gat_layers is not None:
        if gat_layers < 1:
            raise ValueError(f"--gat-layers 需 >= 1，當前為 {gat_layers}")
        cfg["model"]["gat"]["num_layers"] = int(gat_layers)
        _overrides.append(f"gat_layers={gat_layers}")
    if lambda_sparse is not None:
        if lambda_sparse < 0:
            raise ValueError(f"--lambda-sparse 需 >= 0，當前為 {lambda_sparse}")
        cfg["model"].setdefault("weak_links", {})["lambda_sparse"] = float(lambda_sparse)
        _overrides.append(f"lambda_sparse={lambda_sparse}")
    if t_history is not None:
        # 同時決定 LSTM 的步數與 MultiplexDataset 取幾天，兩者都讀這一格
        if t_history < 1:
            raise ValueError(f"--t-history 需 >= 1，當前為 {t_history}")
        cfg["model"]["lstm"]["T_history"] = int(t_history)
        _overrides.append(f"T_history={t_history}")
    if features is not None:
        # input_dim 維持宣告的原始欄數，只有 SharedLSTM 的 input_size 會變
        cfg["model"]["lstm"]["feature_subset"] = list(features)
        _overrides.append(f"features={list(features)}")
    if smooth_window_override is not None:
        cfg["training"]["early_stop_smooth_window"] = int(smooth_window_override)
        _overrides.append(f"smooth_window={smooth_window_override}")
    if early_stop_metric is not None:
        cfg["training"]["early_stop_metric"] = early_stop_metric
        _overrides.append(f"early_stop_metric={early_stop_metric}")
    if beta_n_factors is not None:
        cfg.setdefault("model", {}).setdefault("weak_links", {})["beta_n_factors"] = beta_n_factors
        _overrides.append(f"beta_n_factors={beta_n_factors}")
    if per_target_head:
        cfg.setdefault("model", {}).setdefault("prediction_head", {})["per_target"] = True
        _overrides.append("prediction_head.per_target=True")
    if architecture is not None:
        if architecture not in VALID_ARCHITECTURES:
            raise ValueError(
                f"未知 --architecture={architecture!r}；可選：{VALID_ARCHITECTURES}"
            )
        cfg.setdefault("model", {})["architecture"] = architecture
        _overrides.append(f"architecture={architecture}")
    if _overrides:
        print(f"[cli-override] {', '.join(_overrides)}")

    t_cfg  = cfg["training"]
    mf_cfg = cfg["mlflow"]
    d_cfg  = cfg["data"]

    seed       = int(t_cfg.get("seed", 42))
    max_epochs = int(epochs) if epochs is not None else int(t_cfg.get("max_epochs", 100))
    patience   = int(t_cfg.get("patience", 10))
    batch_size = int(t_cfg.get("batch_size", 32))
    lr         = float(t_cfg.get("lr", 1e-3))
    weight_dec = float(t_cfg.get("weight_decay", 1e-4))
    num_workers = int(t_cfg.get("num_workers", 0))
    pin_memory  = bool(t_cfg.get("pin_memory", False))
    grad_clip   = float(t_cfg.get("grad_clip", 1.0))
    log_every   = int(t_cfg.get("log_every_n_steps", 10))
    min_delta   = float(t_cfg.get("min_delta", 0.0))

    set_seed(seed)
    device = get_device(t_cfg.get("device", "auto"))
    print(f"[train] device={device} | tag={tag} | max_epochs={max_epochs} | patience={patience}")

    # ── DataLoaders ────────────────────────────────────────────────
    snap_dir = d_cfg["snapshot_dir"]
    feat_dir = d_cfg["features_dir"]

    train_ds = MultiplexDataset(snapshot_dir=snap_dir, features_dir=feat_dir,
                                 T=cfg["model"]["lstm"]["T_history"],
                                 split="train", config_path=config_path)
    val_ds   = MultiplexDataset(snapshot_dir=snap_dir, features_dir=feat_dir,
                                 T=cfg["model"]["lstm"]["T_history"],
                                 split="val", config_path=config_path)
    test_ds  = MultiplexDataset(snapshot_dir=snap_dir, features_dir=feat_dir,
                                 T=cfg["model"]["lstm"]["T_history"],
                                 split="test", config_path=config_path)

    common_loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,            # 時序資料禁洗牌
        collate_fn=multiplex_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train_ds, **common_loader_kwargs)
    val_loader   = DataLoader(val_ds,   **common_loader_kwargs)
    test_loader  = DataLoader(test_ds,  **common_loader_kwargs)

    # ── Model / Optimizer / Criterion ─────────────────────────────
    model = build_model(cfg).to(device)
    print(f"[model] architecture={cfg['model'].get('architecture', 'magnet')} "
          f"({sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params)")
    criterion = build_criterion(cfg).to(device)
    # optimizer：adam（預設）| adamw。見 configs/base.yaml training.optimizer。
    opt_name = str(t_cfg.get("optimizer", "adam")).lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_dec)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_dec)
    else:
        raise ValueError(f"training.optimizer 需為 adam 或 adamw，當前為 {opt_name!r}")
    print(f"[optim] {opt_name} lr={lr} weight_decay={weight_dec}")

    # ── 評估用的獨立模型（見 base.yaml training.eval_device）────────
    # 所有 evaluate() 都走 eval_device（預設 cpu），因為 MPS 的 scatter-add
    # 聚合順序不固定：同一份權重連評 5 次 test IC 全距 0.0169，而 CPU 是
    # 位元相同。記錄下來的 metric 必須是可重現的那一個。
    #
    # 用另一個 model 實例而不是把 model 搬來搬去：後者會讓 optimizer 的
    # exp_avg 還在 MPS、param 已在 CPU，下一次 step() 就炸。
    # 非持久 buffer（pair_src / has_pair / feat_idx）由 cfg 決定，
    # eval_model 用同一份 cfg 建，本來就已經對了，不需要也不會被 copy。
    eval_device = get_device(t_cfg.get("eval_device", "cpu"))
    eval_model = build_model(cfg).to(eval_device)
    eval_criterion = build_criterion(cfg).to(eval_device)
    print(f"[eval] eval_device={eval_device} (metrics 由此產生，須可重現)")

    def _eval_ready() -> torch.nn.Module:
        """把訓練中的權重同步到 eval_model，回傳可直接評估的模型。"""
        eval_model.load_state_dict(model.state_dict())
        return eval_model

    # M5 Step 4: CosineAnnealingWarmRestarts（可用 --no-scheduler 關閉做 ablation）
    # T_0=10 → 第一個 cycle 10 個 epoch；T_mult=2 → 之後每 cycle 長度加倍
    # 動機：opt_p2/p7-p10 觀察到 LR=1e-3 持續訓練，模型在 epoch 2-8 衝高 val IC 後就崩
    #       cosine warm restart 讓 LR 在 cycle 末期降低 → 收斂更穩
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2,
        )
        print(f"[scheduler] CosineAnnealingWarmRestarts(T_0=10, T_mult=2)")
    else:
        scheduler = None
        print(f"[scheduler] disabled (--no-scheduler)")

    # M5 Step 4: 從 cfg 讀 early stop monitor（CLI 已可透過 --early-stop-metric 覆寫）
    # IC 為預設；ICIR 用於避免「單天 lucky IC 但訊號不穩」的假性峰值
    es_metric = str(t_cfg.get("early_stop_metric", "IC")).upper()
    if es_metric not in ("IC", "ICIR"):
        print(f"[warn] unknown early_stop_metric={es_metric}, fallback to IC")
        es_metric = "IC"
    # M8 route 1: 選點穩定化——monitor 用 trailing 移動平均（window=1 即現行為）。
    # 動機：val IC 在相鄰 epoch 間 ±0.07 振盪、val loss 平坦，單 epoch 峰值
    # 選點是高變異估計器（見 docs/m8_seed_robustness.md）。
    smooth_w = max(1, int(t_cfg.get("early_stop_smooth_window", 1)))
    smooth_note = f" | smooth_window={smooth_w}" if smooth_w > 1 else ""
    print(f"[early-stop] monitor = val/{es_metric}  (NaN 時 fallback val/IC){smooth_note}")

    # ── MLflow setup ──────────────────────────────────────────────
    mlflow.set_tracking_uri(mf_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mf_cfg.get("experiment_name", "MAGNET_M4_baseline"))

    with mlflow.start_run(run_name=tag) as run:
        run_id = run.info.run_id

        # ── 建立友善 slug + run 資料夾 ───────────────────────────
        # 集中式佈局：runs/<slug>/{checkpoints,predictions,figures}/
        slug = _make_run_slug(run.info.start_time, tag, RUNS_ROOT)
        run_dir = RUNS_ROOT / slug
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)

        # MLflow 端設 tag，方便反向查找
        mlflow.set_tag("magnet.slug", slug)

        print(f"[mlflow] run_id={run_id}")
        print(f"[runs]   slug={slug}  →  {run_dir}/")

        # 凍結訓練當下的 config 快照（M5/M6 對照、論文重現必備）
        with open(run_dir / "config_snapshot.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        # log all hyperparams
        params_flat = _flatten({
            "model":        cfg["model"],
            "loss_weights": cfg["loss_weights"],
            "align_loss":   cfg["align_loss"],
            "training":     {k: v for k, v in t_cfg.items()},
        })
        # 覆寫 epochs（CLI 可能改）
        params_flat["training.max_epochs"] = max_epochs
        params_flat["tag"] = tag
        # mlflow 限制 value 長度，整批 log
        mlflow.log_params({k: str(v) for k, v in params_flat.items()})

        # ── 訓練迴圈 ───────────────────────────────────────────────
        best_monitor  = -float("inf")   # monitor 值（依 es_metric 為 IC 或 ICIR；smooth_w>1 時為其移動平均）
        best_val_ic   = -float("inf")   # 對應 epoch 的 IC（保留供 INDEX/log 寫入）
        best_epoch    = -1
        patience_cnt  = 0
        monitor_hist: list[float] = []  # M8 route 1: trailing MA 用的 monitor 歷史
        best_ckpt_path  = run_dir / "checkpoints" / "best.pt"
        final_ckpt_path = run_dir / "checkpoints" / "final.pt"

        for epoch in range(max_epochs):
            train_stats = train_one_epoch(
                model=model, loader=train_loader, optimizer=optimizer,
                criterion=criterion, device=device,
                grad_clip=grad_clip, log_every=log_every, epoch=epoch,
            )

            val_stats = evaluate(_eval_ready(), val_loader, eval_device,
                                 criterion=eval_criterion)
            val_ic   = val_stats["IC"]
            val_icir = val_stats["ICIR"]

            # MLflow log
            mlflow.log_metrics({
                "train/loss_total":    train_stats["loss_total"],
                "train/loss_mse":      train_stats["loss_mse"],
                "train/loss_rank":     train_stats["loss_rank"],
                "train/loss_align":    train_stats["loss_align"],
                "train/loss_variance": train_stats["loss_variance"],
                "train/epoch_time_sec": train_stats["epoch_time_sec"],
                "val/loss_total":      val_stats.get("loss_total",    float("nan")),
                "val/loss_mse":        val_stats.get("loss_mse",      float("nan")),
                "val/loss_rank":       val_stats.get("loss_rank",     float("nan")),
                "val/loss_align":      val_stats.get("loss_align",    float("nan")),
                "val/loss_variance":   val_stats.get("loss_variance", float("nan")),
                "val/MSE":             val_stats["MSE"],
                "val/MAE":             val_stats["MAE"],
                "val/IC":           val_stats["IC"],
                "val/RankIC":       val_stats["RankIC"],
                "val/ICIR":         val_stats["ICIR"] if not np.isnan(val_stats["ICIR"]) else 0.0,
                "val/RankICIR":     val_stats["RankICIR"] if not np.isnan(val_stats["RankICIR"]) else 0.0,
                "learning_rate":    optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_stats['loss_total']:.5f} | "
                f"val_loss={val_stats.get('loss_total', float('nan')):.5f} "
                f"val_IC={val_ic:.4f} val_RankIC={val_stats['RankIC']:.4f} "
                f"({train_stats['epoch_time_sec']:.1f}s)"
            )

            # M8 路線 A 實驗 3（Figure 1）：逐 epoch 存權重供事後 test 軌跡評估
            if save_every_epoch:
                save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                                model, None, epoch=epoch, best_val_ic=val_ic)

            # Early stopping on configurable metric（越大越好）
            # 選 monitor 值：ICIR NaN 時退回 IC
            if es_metric == "ICIR" and not np.isnan(val_icir):
                monitor_val = val_icir
            else:
                monitor_val = val_ic

            # M8 route 1: trailing 移動平均（開頭窗口不足時取現有歷史）
            monitor_hist.append(monitor_val)
            if smooth_w > 1:
                monitor_val = float(np.mean(monitor_hist[-smooth_w:]))

            improved = (not np.isnan(monitor_val)) and (monitor_val > best_monitor + min_delta)
            if improved:
                best_monitor = monitor_val
                best_val_ic  = val_ic           # 紀錄該 epoch 的 IC（INDEX 用）
                best_epoch   = epoch
                patience_cnt = 0
                save_checkpoint(best_ckpt_path, model, optimizer,
                                epoch=epoch, best_val_ic=best_val_ic,
                                extras={"val_stats": {k: v for k, v in val_stats.items()
                                                        if k not in ("predictions", "daily_IC", "daily_RankIC")}})
                print(f"    ↑ new best val_{es_metric}={best_monitor:.4f} "
                      f"(val_IC={best_val_ic:.4f}, epoch {epoch}) → saved")
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"[early stopping] val_{es_metric} no improvement for "
                          f"{patience} epochs, stopping at epoch {epoch}")
                    break

            # M5 Step 4: 推進 LR scheduler（cosine warm restart）
            if scheduler is not None:
                scheduler.step()

        # 存 final
        save_checkpoint(final_ckpt_path, model, optimizer,
                        epoch=epoch, best_val_ic=best_val_ic, extras={"final": True})

        # ── 載入 best → test 一次性評估 ───────────────────────────
        if best_ckpt_path.exists():
            load_checkpoint(best_ckpt_path, model, optimizer=None, map_location=device)
            print(f"[test] loaded best ckpt (epoch={best_epoch}, val_IC={best_val_ic:.4f})")
        else:
            print("[test] WARNING: best ckpt 不存在（IC 從未為正），用 final 評估")

        test_stats = evaluate(
            _eval_ready(), test_loader, eval_device, criterion=eval_criterion,
            eval_cfg=cfg.get("evaluation"),
        )
        print(
            f"[test] IC={test_stats['IC']:.4f} RankIC={test_stats['RankIC']:.4f} "
            f"ICIR={test_stats['ICIR']:.4f} MSE={test_stats['MSE']:.5f} "
            f"R2={test_stats['R2']:.5f}"
        )
        if "Sharpe" in test_stats:
            print(
                f"[test] Sharpe={test_stats['Sharpe']:.3f} "
                f"hit_rate={test_stats['hit_rate']:.3f} "
                f"(pre-cost, n_days={test_stats['pf_n_days']})"
            )

        # log test metrics
        mlflow.log_metrics({
            "test/IC":       test_stats["IC"],
            "test/RankIC":   test_stats["RankIC"],
            "test/ICIR":     test_stats["ICIR"] if not np.isnan(test_stats["ICIR"]) else 0.0,
            "test/RankICIR": test_stats["RankICIR"] if not np.isnan(test_stats["RankICIR"]) else 0.0,
            "test/MSE":      test_stats["MSE"],
            "test/MAE":      test_stats["MAE"],
            "test/RMSE":     test_stats["RMSE"],
            "test/R2":       test_stats["R2"],
            "test/R2_zero":  test_stats["R2_zero"],
            "best_epoch":    float(best_epoch),
            "best_val_IC":   best_val_ic,
        })

        if "Sharpe" in test_stats:
            mlflow.log_metrics({
                "test/Sharpe":         test_stats["Sharpe"],
                "test/mean_daily_pnl": test_stats["mean_daily_pnl"],
                "test/std_daily_pnl":  test_stats["std_daily_pnl"],
                "test/hit_rate":       test_stats["hit_rate"],
                "test/ann_return":     test_stats["ann_return"],
            })

        # ── Predictions：本地寫入 runs/<slug>/predictions/，並上傳 MLflow ──
        test_pred_path = run_dir / "predictions" / "test_predictions.csv"
        val_pred_path  = run_dir / "predictions" / "val_predictions.csv"
        test_stats["predictions"].to_csv(test_pred_path, index=False)
        val_pred_dump = evaluate(_eval_ready(), val_loader, eval_device,
                                 criterion=eval_criterion)["predictions"]
        val_pred_dump.to_csv(val_pred_path, index=False)

        if bool(mf_cfg.get("log_artifacts", True)):
            mlflow.log_artifact(str(test_pred_path), artifact_path="predictions")
            mlflow.log_artifact(str(val_pred_path),  artifact_path="predictions")
            if best_ckpt_path.exists():
                mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
            mlflow.log_artifact(str(final_ckpt_path), artifact_path="checkpoints")
            mlflow.log_artifact(str(run_dir / "config_snapshot.yaml"))

        # ── 自我驗證：best.pt 必須重現剛剛記下的 test IC ──────────
        # 這是 2026-08-16 之前那批 run 壞掉的地方——metric 在 MPS 上算，
        # 換一個 process 重載 best.pt 再評就對不上（全距 0.0169）。
        # 現在每個 run 自己檢查一次，1.3 秒，對不上就當場停。
        roundtrip_ic = None
        if best_ckpt_path.exists():
            verify_model = build_model(cfg).to(eval_device)
            load_checkpoint(best_ckpt_path, verify_model, optimizer=None,
                            map_location=eval_device)
            roundtrip_ic = float(evaluate(
                verify_model, test_loader, eval_device,
                eval_cfg=cfg.get("evaluation"),
            )["IC"])
            drift = abs(roundtrip_ic - float(test_stats["IC"]))
            if drift > 1e-4:
                raise RuntimeError(
                    f"checkpoint 無法重現自己的 test IC："
                    f"記錄 {float(test_stats['IC']):+.6f} vs 重載 {roundtrip_ic:+.6f} "
                    f"(差 {drift:.2e} > 1e-4)。eval_device={eval_device}；"
                    f"若為 mps 請改回 cpu——MPS 的 scatter-add 不可重現。"
                )
            print(f"[verify] best.pt 重現 test IC {roundtrip_ic:+.6f} "
                  f"(差 {drift:.2e}) OK")

        # ── 寫 meta.json + 補 INDEX.csv 一列 ─────────────────────
        end_ms = int(time.time() * 1000)
        meta = {
            "slug":       slug,
            "run_id":     run_id,
            "tag":        tag,
            "start_time": _iso(run.info.start_time),
            "end_time":   _iso(end_ms),
            "status":     "FINISHED",
            "n_epochs":   epoch + 1,
            "best_epoch": best_epoch,
            "best_val_IC": float(best_val_ic) if best_val_ic > -float("inf") else None,
            # metric 是在哪個 device 上算的。cpu 以外的值代表數字不保證可重現。
            "eval_device": str(eval_device),
            "test_IC_roundtrip": roundtrip_ic,
            "test_metrics": {
                "IC":     float(test_stats["IC"]),
                "RankIC": float(test_stats["RankIC"]),
                "ICIR":   float(test_stats["ICIR"]) if not np.isnan(test_stats["ICIR"]) else None,
                "MSE":    float(test_stats["MSE"]),
                "MAE":    float(test_stats["MAE"]),
                "RMSE":   float(test_stats["RMSE"]),
            },
        }
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        _append_index_row({
            "slug":        slug,
            "run_id":      run_id,
            "tag":         tag,
            "start_time":  _iso(run.info.start_time),
            "end_time":    _iso(end_ms),
            "status":      "FINISHED",
            "n_epochs":    epoch + 1,
            "best_epoch":  best_epoch,
            "best_val_IC": round(float(best_val_ic), 6) if best_val_ic > -float("inf") else "",
            "test_IC":     round(float(test_stats["IC"]),     6),
            "test_RankIC": round(float(test_stats["RankIC"]), 6),
            "test_ICIR":   round(float(test_stats["ICIR"]),   6) if not np.isnan(test_stats["ICIR"]) else "",
            "test_MSE":    round(float(test_stats["MSE"]),    8),
        })

        print(f"\n[runs] meta.json + INDEX.csv 已更新 → {run_dir}/")
        return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAGNET 訓練主程式")
    p.add_argument("--config", type=str, default="configs/base.yaml",
                   help="YAML 路徑（預設 configs/base.yaml）")
    p.add_argument("--epochs", type=int, default=None,
                   help="覆寫 cfg.training.max_epochs（sanity 用 3）")
    p.add_argument("--tag", type=str, default="full",
                   help="MLflow run name / runs/ slug 後綴（建議 opt_pN_<desc>）")
    # M5 sprint：超參覆寫（None = 沿用 yaml 設定）
    p.add_argument("--mse-weight",      type=float, default=None,
                   help="覆寫 cfg.loss_weights.mse")
    p.add_argument("--rank-weight",     type=float, default=None,
                   help="覆寫 cfg.loss_weights.rank")
    p.add_argument("--variance-weight", type=float, default=None,
                   help="覆寫 cfg.loss_weights.variance")
    p.add_argument("--lr",              type=float, default=None,
                   help="覆寫 cfg.training.lr")
    p.add_argument("--early-stop-metric", choices=["IC", "ICIR"], default=None,
                   help="覆寫 cfg.training.early_stop_metric（IC 預設）")
    p.add_argument("--no-scheduler", action="store_true",
                   help="關閉 CosineAnnealingWarmRestarts（用於 Step 4 ablation）")
    p.add_argument("--architecture",
                   choices=list(VALID_ARCHITECTURES), default=None,
                   help="覆寫 cfg.model.architecture（M6 Stage 0 ablation 用）")
    p.add_argument("--patience", type=int, default=None,
                   help="覆寫 cfg.training.patience（M7 hyperparameter grid 用）")
    p.add_argument("--seed", type=int, default=None,
                   help="覆寫 cfg.training.seed（M8 multi-seed robustness 用）")
    p.add_argument("--smooth-window", type=int, default=None,
                   help="覆寫 cfg.training.early_stop_smooth_window（M8 route 1 選點穩定化；1=現行為）")
    p.add_argument("--residual-alpha", type=float, default=None,
                   help="覆寫 cfg.model.residual.alpha（magnet_intermediate 的 "
                        "initial residual；0=無殘差，即原行為）")
    p.add_argument("--coupling-init-other", type=float, default=None,
                   help="覆寫 cfg.model.coupling.init_other（稠密 A 的非配對位置初始值）。"
                        "不給則用 1/n1；給 0 等於從現行 MAGNET 的結構出發。"
                        "只在 --architecture magnet_dense_a 下有作用。")
    p.add_argument("--raw-skip", type=str, default=None,
                   choices=["none", "l1", "l2", "both"],
                   help="覆寫 cfg.model.raw_skip（階段 A-2a）。把原始特徵最後一步"
                        "跳接到耦合點；量到的耦合點上限由 +0.0448 升到 +0.0666。"
                        "預設 none = 原行為。")
    p.add_argument("--raw-skip-mode", type=str, default=None,
                   choices=["add", "concat"],
                   help="覆寫 cfg.model.raw_skip.mode。add=階段 A-2a（已測無效）；"
                        "concat=A-2a'，proj 讓出 concat_dim 維給原始特徵，"
                        "保留兩條路徑的可分性（拼接上限 +0.0662 vs 相加 +0.0446）。")
    p.add_argument("--input-norm", type=str, default=None,
                   choices=["none", "batchnorm"],
                   help="覆寫 cfg.model.lstm.input_norm。batchnorm=在 LSTM 之前"
                        "逐特徵跨節點對齊尺度。動機：原始尺度下 RSI_14 的 std 是"
                        "log_return 的 471 倍，初始化時 4 成閘門已飽和，且 "
                        "weight_decay 1e-3 > log_return 方向的資料曲率 4.97e-4。"
                        "預設 none = 原行為。")
    p.add_argument("--input-norm-scope", type=str, default=None,
                   choices=["shared", "per_layer"],
                   help="覆寫 cfg.model.lstm.input_norm_scope。shared=兩層共用一個"
                        "FeatureNorm（預設，殘餘市場間尺度差約 1.53 倍）；"
                        "per_layer=每層各一個，與 raw_skip 的做法一致。")
    p.add_argument("--beta-layer", action="store_true", default=None,
                   help="開啟 model.weak_links.beta_layer（階段 P1）。每檔台股一個"
                        "可學的 ADR 權重 alpha_j 與市場因子暴露 gamma_j，候選邊改吃"
                        "殘差。目標是 rank-1 beta 模型的 +0.0738。")
    p.add_argument("--graph-ablate", type=str, default=None,
                   choices=["none", "empty_l1", "empty_l2", "empty_both",
                            "rand_l1", "rand_l2", "rand_both"],
                   help="層內圖 A₁/A₂ 的消融。empty=只留 self-loop（拿掉訊息傳遞，"
                        "保留 GAT 參數）；rand=保留邊數與 edge_attr、只打亂端點"
                        "（檢驗相關係數挑出的結構是否有資訊）。")
    p.add_argument("--beta-ablate", type=str, default=None,
                   help="關掉 beta 層的某幾項（逗號分隔）："
                        "identity=① ADR 特異訊號、factor=② 市場因子暴露、"
                        "residual=③ 殘差聚合。例：--beta-ablate identity")
    p.add_argument("--beta-init-std", type=float, default=None,
                   help="覆寫 model.weak_links.beta_init_std（alpha/gamma 的初始"
                        "離散度）。設為 0 會讓 beta 層在初始化點逐位元退化成現行 MAGNET。")
    p.add_argument("--optimizer", type=str, default=None,
                   choices=["adam", "adamw"],
                   help="覆寫 cfg.training.optimizer。adamw 用解耦 weight decay，"
                        "可分離「正規化不足」與「衰減耦合」兩個機制。"
                        "預設 adam = 原行為。")
    p.add_argument("--cs-demean", type=str, default=None,
                   choices=["none", "l1", "l2", "both"],
                   help="覆寫 cfg.model.cs_demean（階段 A-7）。耦合前對節點維度"
                        "去均值，拆掉表示塌縮（實測 L1 餘弦 +0.9997 -> +0.1616）。"
                        "預設 none = 原行為。")
    p.add_argument("--gat-layers", type=int, default=None,
                   help="覆寫 cfg.model.gat.num_layers（= 圖上的跳數；掃過度平滑用）")
    p.add_argument("--lambda-sparse", type=float, default=None,
                   help="覆寫 cfg.model.weak_links.lambda_sparse。tw50.yaml 無此區塊，"
                        "不指定會吃到程式預設 1e-3（在 tw50 下等同凍結全部候選邊）")
    p.add_argument("--t-history", type=int, default=None,
                   help="覆寫 cfg.model.lstm.T_history（回看天數；同時影響 dataset）")
    p.add_argument("--features", nargs="*", default=None,
                   help="覆寫 cfg.model.lstm.feature_subset，例如 --features log_return"
                        "（預設全取 9 欄；只影響 LSTM 的 input_size，input_dim 不變）")
    p.add_argument("--beta-n-factors", type=int, default=None,
                   help="覆寫 cfg.model.weak_links.beta_n_factors。"
                        "k>1 時 ② 改為 k 個學出來的因子。")
    p.add_argument("--per-target-head", action="store_true", default=None,
                   help="覆寫 cfg.model.prediction_head.per_target=True。"
                        "每檔台股一條讀出向量；缺口見 proposal §34.2 缺陷 2。")
    p.add_argument("--save-every-epoch", action="store_true",
                   help="逐 epoch 存 checkpoint（M8 Figure 1 軌跡分析用）")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_id = train(
        config_path=args.config,
        epochs=args.epochs,
        tag=args.tag,
        mse_weight=args.mse_weight,
        rank_weight=args.rank_weight,
        variance_weight=args.variance_weight,
        lr_override=args.lr,
        early_stop_metric=args.early_stop_metric,
        use_scheduler=not args.no_scheduler,
        architecture=args.architecture,
        patience_override=args.patience,
        seed_override=args.seed,
        smooth_window_override=args.smooth_window,
        residual_alpha=args.residual_alpha,
        cs_demean=args.cs_demean,
        raw_skip=args.raw_skip,
        raw_skip_mode=args.raw_skip_mode,
        input_norm=args.input_norm,
        input_norm_scope=args.input_norm_scope,
        beta_layer=args.beta_layer,
        beta_init_std=args.beta_init_std,
        beta_ablate=args.beta_ablate,
        graph_ablate=args.graph_ablate,
        per_target_head=args.per_target_head,
        beta_n_factors=args.beta_n_factors,
        optimizer_name=args.optimizer,
        coupling_init_other=args.coupling_init_other,
        gat_layers=args.gat_layers,
        lambda_sparse=args.lambda_sparse,
        t_history=args.t_history,
        features=args.features,
        save_every_epoch=args.save_every_epoch,
    )
    print(f"\nDone. MLflow run_id = {run_id}")
    print(f"   查看：  cat runs/INDEX.csv  |  ls runs/")
