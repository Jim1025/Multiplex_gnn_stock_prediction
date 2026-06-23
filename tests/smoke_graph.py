"""
smoke_graph.py — graph_builder.py MVP 的關鍵行為驗證

涵蓋使用者要求的 5 個驗證項目：
  ① 圖快照產出數量正確（≈1640 張）
  ② A12 嚴格對角、無跨公司洩漏
  ③ 相關係數窗口內無 target_date 及之後的資料
  ④ 9 維節點特徵正確讀取
  ⑤ HeteroData 結構符合 PyG 規範，可被 GAT 直接吃
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, "/Users/lichengjun/Desktop/Multiplex_gnn_stock_prediction/src/dataset")
from graph_builder import (
    MultiplexGraphBuilder,
    NODE_FEATURE_COLS, N_FEATURES,
    EDGE_L1_TYPE, EDGE_L2_TYPE, EDGE_A12_TYPE,
)


# ════════════════════════════════════════════════════════════
# 準備合成資料（模擬 7 組配對的 features.py 產出）
# ════════════════════════════════════════════════════════════

ROOT = Path("/tmp/smoke_graph")
if ROOT.exists():
    shutil.rmtree(ROOT)
ADR_DIR = ROOT / "data/features/adr"
TW_DIR  = ROOT / "data/features/tw"
ADR_DIR.mkdir(parents=True)
TW_DIR.mkdir(parents=True)
OUT_DIR = ROOT / "data/graphs/snapshots"


PAIR_MAP = {
    "TSM":   "2330",
    "UMC":   "2303",
    "ASX":   "3711",
    "CHT":   "2412",
    "IMOS":  "8150",
    "AUOTY": "2409",
    "HNHPF": "2317",
}

N_TRADING_DAYS = 1700  # 模擬約 7 年的 TWSE 交易日
START_DATE     = "2018-01-02"


def make_features_csv(path: Path, seed: int, base_close: float = 100.0):
    """產生符合 features.py 格式的合成 CSV（9 維 + 4 個追溯欄位）"""
    np.random.seed(seed)
    dates = pd.bdate_range(START_DATE, periods=N_TRADING_DAYS)
    close = base_close + np.cumsum(np.random.normal(0, 1, N_TRADING_DAYS))

    df = pd.DataFrame({
        "Open":   close,
        "High":   close * 1.01,
        "Low":    close * 0.99,
        "Close":  close,
        "Volume": np.random.randint(1000, 5000, N_TRADING_DAYS),
    }, index=dates)
    df.index.name = "Date"

    # 9 維特徵
    df["log_return"]   = np.log(df["Close"] / df["Close"].shift(1))
    df["RSI_14"]       = np.random.uniform(20, 80, N_TRADING_DAYS)
    df["MACD"]         = np.random.normal(0, 0.5, N_TRADING_DAYS)
    df["MACD_signal"]  = np.random.normal(0, 0.5, N_TRADING_DAYS)
    df["MACD_hist"]    = np.random.normal(0, 0.3, N_TRADING_DAYS)
    df["BB_pos"]       = np.random.uniform(0, 1, N_TRADING_DAYS)
    df["MA5_dev"]      = np.random.normal(0, 0.02, N_TRADING_DAYS)
    df["MA20_dev"]     = np.random.normal(0, 0.05, N_TRADING_DAYS)
    df["log_volume_z"] = np.random.normal(0, 1, N_TRADING_DAYS)

    # 4 個追溯欄位
    df["is_imputed"]              = False
    df["gap_length"]              = 0
    df["imputation_source_date"]  = pd.NaT
    df["is_long_gap"]             = False

    # 故意設一個 long_gap 列在中段，讓 ADR 端 TSM 第 500 列為 long_gap
    if seed == 1:   # TSM
        df.iloc[500, df.columns.get_loc("is_long_gap")] = True
        # long_gap 列的特徵應為 NaN（features.py 的行為）
        for col in NODE_FEATURE_COLS:
            df.iloc[500, df.columns.get_loc(col)] = np.nan

    df.to_csv(path)


# 用不同 seed 製作 7 組配對的特徵
for i, (adr, tw) in enumerate(PAIR_MAP.items()):
    make_features_csv(ADR_DIR / f"{adr}.csv", seed=i*2 + 1)
    make_features_csv(TW_DIR  / f"{tw}.csv",  seed=i*2 + 2)


# ════════════════════════════════════════════════════════════
# 建立 GraphBuilder 並建構序列
# ════════════════════════════════════════════════════════════

print("=" * 70)
print("Graph Builder MVP - Smoke Test")
print("=" * 70)

gb = MultiplexGraphBuilder(
    pair_map=PAIR_MAP,
    adr_dir=str(ADR_DIR),
    tw_dir=str(TW_DIR),
    corr_window=60,
    corr_threshold=0.3,
)

# 建構整個序列（從第 60 個交易日開始才有暖機資料）
all_meta = gb.build_sequence(
    start_date=pd.Timestamp(START_DATE),
    end_date=pd.Timestamp("2024-12-31"),
    out_dir=str(OUT_DIR),
)


# ════════════════════════════════════════════════════════════
# 驗證項目
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("驗證結果")
print("=" * 70)

# 統計
n_total   = len(all_meta)
n_success = sum(1 for m in all_meta if m.n_l1_nodes > 0)
saved_files = sorted(OUT_DIR.glob("graph_*.pt"))
n_files   = len(saved_files)


# === 驗證 1：圖快照產出數量正確 ≈ 1640 張 ===
# N_TRADING_DAYS = 1700，扣除暖機期 60 = 1640
print(f"\n【驗證 1】快照數量")
print(f"  總目標日候選: {n_total}")
print(f"  成功建構快照: {n_success}")
print(f"  序列化檔案數: {n_files}")
expected = N_TRADING_DAYS - 60
assert n_success == expected, \
    f"期望 {expected} 張快照，實際 {n_success}"
assert n_files == n_success, \
    f"成功建構 {n_success} 但儲存 {n_files}"
print(f"  PASS: 共 {n_success} 張快照（預期 {expected}）")


# === 驗證 2：A12 嚴格對角、無跨公司洩漏 ===
print(f"\n【驗證 2】A12 嚴格對角")
sample_path = saved_files[100]   # 取中段一張快照
data = torch.load(sample_path, weights_only=False)

a12 = data[EDGE_A12_TYPE].edge_index
assert a12.shape == (2, 7), f"A12 shape 錯誤：{a12.shape}（應為 [2,7]）"

# 檢查每一條邊：src == dst（嚴格對角）
for k in range(a12.shape[1]):
    src, dst = int(a12[0, k]), int(a12[1, k])
    assert src == dst, \
        f"A12 第 {k} 條邊 src={src} ≠ dst={dst}（跨公司洩漏！）"

# 檢查覆蓋所有節點
src_set = set(a12[0].tolist())
dst_set = set(a12[1].tolist())
assert src_set == dst_set == set(range(7)), \
    f"A12 未覆蓋全部 7 個節點：src={src_set}, dst={dst_set}"
print(f"  PASS: A12 = {a12.tolist()}，嚴格對角且覆蓋 7 節點")


# === 驗證 3：相關係數視窗內無 target_date 及之後的資料 ===
print(f"\n【驗證 3】時序窗口無 Look-ahead")
# 取一張中段快照來檢查
test_idx = 200
sample_meta = [m for m in all_meta if m.n_l1_nodes > 0][test_idx]
data = torch.load(saved_files[test_idx], weights_only=False)

target_date  = pd.Timestamp(data.target_date)
window_start = pd.Timestamp(data.window_start)
window_end   = pd.Timestamp(data.window_end)

print(f"  target_date  = {target_date.date()}")
print(f"  window_start = {window_start.date()}")
print(f"  window_end   = {window_end.date()}")

# 防線 1：window_end 必須嚴格早於 target_date
assert window_end < target_date, \
    f"window_end {window_end.date()} 不嚴格早於 target {target_date.date()}"

# 防線 2：window 跨度 = corr_window = 60
ws_pos = pd.Series(0, index=pd.bdate_range(START_DATE, periods=N_TRADING_DAYS)).index.get_loc(window_start)
we_pos = pd.Series(0, index=pd.bdate_range(START_DATE, periods=N_TRADING_DAYS)).index.get_loc(window_end)
window_size = we_pos - ws_pos + 1
assert window_size == 60, f"視窗跨度 {window_size} ≠ 60"

# 防線 3：直接讀回 ADR CSV，確認用於相關係數的資料皆 < target_date
df_tsm = pd.read_csv(ADR_DIR / "TSM.csv", index_col="Date", parse_dates=True)
window_data = df_tsm.loc[window_start:window_end]
assert window_data.index.max() < target_date, \
    f"視窗內含 target_date 之後的資料"

print(f"  PASS: 視窗 [{window_start.date()}, {window_end.date()}] "
      f"嚴格早於 target {target_date.date()}，跨度 60 工作日")


# === 驗證 4：9 維節點特徵正確讀取 ===
print(f"\n【驗證 4】9 維節點特徵")
data = torch.load(saved_files[100], weights_only=False)

# 節點數與特徵維度
assert data["adr"].x.shape == (7, 9), \
    f"ADR 特徵 shape={data['adr'].x.shape}（應為 [7,9]）"
assert data["tw"].x.shape == (7, 9), \
    f"TW 特徵 shape={data['tw'].x.shape}（應為 [7,9]）"

# 特徵 dtype
assert data["adr"].x.dtype == torch.float32
assert data["tw" ].x.dtype == torch.float32

# 特徵值不應全 0（否則代表沒讀到資料）
assert data["adr"].x.abs().sum() > 0, "ADR 特徵全為 0，疑似讀取失敗"
assert data["tw" ].x.abs().sum() > 0, "TW 特徵全為 0，疑似讀取失敗"

# 追溯欄位 mask 存在
assert hasattr(data["adr"], "is_imputed")
assert hasattr(data["adr"], "is_long_gap")
assert data["adr"].is_imputed.dtype == torch.bool
assert data["adr"].is_long_gap.dtype == torch.bool
print(f"  PASS: ADR x={tuple(data['adr'].x.shape)}, "
      f"TW x={tuple(data['tw'].x.shape)}")
print(f"        含 is_imputed / is_long_gap mask（追溯政策 §8.1）")


# === 驗證 5：HeteroData 結構符合 PyG 規範，可被 GAT 直接吃 ===
print(f"\n【驗證 5】HeteroData 結構與 GAT 相容性")
data = torch.load(saved_files[100], weights_only=False)

# 5a：是 HeteroData 物件
assert isinstance(data, HeteroData), f"型別錯誤：{type(data)}"

# 5b：三類邊都存在
assert EDGE_L1_TYPE  in data.edge_types, "L1 邊類型缺失"
assert EDGE_L2_TYPE  in data.edge_types, "L2 邊類型缺失"
assert EDGE_A12_TYPE in data.edge_types, "A12 邊類型缺失"

# 5c：邊 index 維度合法（[2, n_edges]）
for et in [EDGE_L1_TYPE, EDGE_L2_TYPE, EDGE_A12_TYPE]:
    ei = data[et].edge_index
    assert ei.dim() == 2 and ei.size(0) == 2, \
        f"邊 {et} edge_index shape 錯誤：{ei.shape}"
    assert ei.dtype == torch.long, \
        f"邊 {et} edge_index dtype 應為 long，實際 {ei.dtype}"

# 5d：邊權重存在於 L1/L2，且維度合法
for et in [EDGE_L1_TYPE, EDGE_L2_TYPE]:
    ea = data[et].edge_attr
    n_edges = data[et].edge_index.shape[1]
    assert ea.shape == (n_edges, 1), \
        f"邊 {et} edge_attr shape={ea.shape}（應為 [{n_edges},1]）"
    # 邊權重 = |ρ|，應在 [0, 1]
    if n_edges > 0:
        assert ea.min() >= 0 and ea.max() <= 1, \
            f"邊權重 {ea.min():.3f}~{ea.max():.3f} 超出 [0,1]"

# 5e：y 標籤存在於 TW 節點
assert hasattr(data["tw"], "y"), "缺少預測目標 y"
assert data["tw"].y.shape == (7,), f"y shape 錯誤：{data['tw'].y.shape}"

# 5f：嘗試模擬 GAT 的 forward pass（不實際訓練，只驗證結構合法）
from torch_geometric.nn import GATConv

# L1 GAT
gat_l1 = GATConv(in_channels=9, out_channels=16, heads=2, edge_dim=1)
out_l1 = gat_l1(
    x=data["adr"].x,
    edge_index=data[EDGE_L1_TYPE].edge_index,
    edge_attr=data[EDGE_L1_TYPE].edge_attr,
)
assert out_l1.shape == (7, 32), f"L1 GAT 輸出 shape={out_l1.shape}"

# L2 GAT
gat_l2 = GATConv(in_channels=9, out_channels=16, heads=2, edge_dim=1)
out_l2 = gat_l2(
    x=data["tw"].x,
    edge_index=data[EDGE_L2_TYPE].edge_index,
    edge_attr=data[EDGE_L2_TYPE].edge_attr,
)
assert out_l2.shape == (7, 32), f"L2 GAT 輸出 shape={out_l2.shape}"

print(f"  PASS: HeteroData 結構合法")
print(f"        L1 邊數={data[EDGE_L1_TYPE].edge_index.shape[1]} (含權重)")
print(f"        L2 邊數={data[EDGE_L2_TYPE].edge_index.shape[1]} (含權重)")
print(f"        A12 邊數={data[EDGE_A12_TYPE].edge_index.shape[1]} (純對角)")
print(f"        GATConv forward 通過：[7,9] → [7,32]")


# === 額外驗證：long_gap 列特徵被正確處理 ===
print(f"\n【附加驗證】long_gap 處理")
# TSM seed=1 的第 500 列被設為 long_gap，對應日期
tsm_dates = pd.bdate_range(START_DATE, periods=N_TRADING_DAYS)
long_gap_date = tsm_dates[500]

# 找對應的快照（target_date 的前一日是該 long_gap 日）
# 即 target_date = long_gap_date 之後第 1 個交易日
target_idx = 501
target_d = tsm_dates[target_idx]

# 找到對應的 .pt
matching = [f for f in saved_files if f.stem == f"graph_{target_d.strftime('%Y-%m-%d')}"]
if matching:
    data = torch.load(matching[0], weights_only=False)
    # TSM 是 ADR 的第 0 個節點
    is_lg = data["adr"].is_long_gap[0].item()
    assert is_lg, f"TSM 在 {long_gap_date.date()} 應標記為 long_gap"
    print(f"  PASS: long_gap 旗標正確傳遞至圖節點")
else:
    print(f"  SKIP: 找不到 target={target_d.date()} 的快照")

print("\n" + "=" * 70)
print("所有 5 項關鍵驗證通過")
print("=" * 70)