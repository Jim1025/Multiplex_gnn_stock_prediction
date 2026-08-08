# MAGNET — 專案工作規範

Multi-market ADR-Guided NETwork：以美股 ADR 的隔夜資訊預測台股次日報酬的橫截面排序。

---

## 硬性規範

1. **資料層預設不可動**——`src/dataset/` 下的 `pipeline.py` / `features.py` /
   `graph_builder.py` / `config.py`。例外程序見下節。
2. **超參數一律從 `configs/base.yaml` 讀**,`.py` 內禁止 hard-code。
   純函式可有必要的結構性預設值（例如由 F 推導 2F），但可調參數不行。
3. **亂數種子同時固定**:`torch.manual_seed(42)` 與 `np.random.seed(42)` 兩者皆須設定。
4. **圖卷積使用 PyG 的 `GATv2Conv`**,不得用舊版 `GATConv`。
5. **Walk-forward split,禁止隨機切**。分割點由 `configs/base.yaml` 的
   `data.split` 指定。
6. **`shuffle=False`**——時序資料禁止洗牌,DataLoader 與訓練迴圈皆是。
7. **不使用 emoji**——程式碼、文件、對話輸出皆不得出現。

---

## 資料層變更例外條款

資料層預設不可動,但允許在**同時滿足以下四項**時變更:

1. **使用者在對話中明確授權**該次變更。事後追認不算。
2. **變更理由與影響範圍寫進程式碼註解**,而不只寫在 commit 訊息裡——
   讀原始碼的人必須能就地看懂為什麼這樣做。
3. **變更後執行 `.venv/bin/python scripts/freeze_k7.py --verify`**:
   - 通過 → 代表 k=7 模型輸入未受影響,可直接繼續
   - 未通過 → 必須先確認差異是預期的,**重新凍結**（`--emit`）後才能繼續。
     不得以「差異很小」帶過。
4. **commit 訊息標明「經授權變更資料層」**,並列出驗證結果。

### 已授權的變更紀錄

| 日期 | 檔案 / 位置 | 變更 | 驗證結果 |
|---|---|---|---|
| 2026-08-08 | `pipeline.py::_step1_outliers` 1a | 價格水準 IQR 由「裁切」改為「僅診斷計數」。原實作對趨勢資產把 Close 壓成常數,導致 `log_return ≡ 0`、**預測目標被破壞**（2308 有 79/246 個測試日 y ≡ 0）;且全序列分位數構成 look-ahead | `freeze_k7.py --verify` 17/17 通過;1,643 個 graph snapshot 內容 0 改變;重跑 early-fusion 與凍結基準位元相同 |

### 已知殘留項（尚未處理）

- `pipeline.py::_step1_outliers` 1b：成交量 IQR 裁切仍使用全序列分位數,
  同樣有輕微 look-ahead。未修正是因為成交量大致平穩、無趨勢股封頂問題,
  且變更會擾動 `log_volume_z` 而使凍結基準必須重做。
  處置:擴充實驗結束後再修,或於論文直接揭露。

---

## 環境

- **Python 環境為 `.venv`**,不是系統 python3。`mlflow` 只裝在 `.venv`。
  一律用 `.venv/bin/python` 執行訓練與腳本。
- 訓練:`.venv/bin/python -m src.train.train --architecture <arch> --tag <slug>`
- 測試:`.venv/bin/python -m pytest tests/ -q`

---

## 基準與 universe

- **k=7 凍結基準**:git tag `k7-frozen`。內容見 `docs/frozen_k7_results.md`,
  機器可讀版 `docs/frozen_k7_manifest.json`。
  驗證:`.venv/bin/python scripts/freeze_k7.py --verify`
  - `baseline_lstm` 與 `baseline_early_fusion` 走純 LSTM 路徑,同 seed **位元確定**
    → 以 SHA-256 逐位元比對,這是硬證據
  - GNN 路徑在 MPS 下非位元確定 → 只能比對 test_IC 是否落在 ±3σ_run（±0.081）內
- **擴充後 universe**:`configs/universe/universe_tw_2019.json`
  （TW 50 檔 / US 30 檔,恆等配對 7 對,配對率 14.0%）
- `data/graphs/snapshots/` 是 k=7 基準的一部分,**擴充後的快照請寫入新目錄**
  （如 `snapshots_tw50/`）,否則凍結驗證無法執行。

---

## 報告紀律

- 實驗結果**照實報告**,包含不利於現行架構者。第 0 階段的 early-fusion
  對照就是一例:它勝過 MAGNET,已完整記錄並附上事先登記的可證偽預測。
- 任何單一 run 的數字都要附**變異來源與雜訊帶**。M9 已證明 run（非 seed）
  才是變異單位,σ_run ≈ 0.027;k=7 的橫截面連 Sharpe 都有 σ ≈ 1.0 的虛無帶。
- 事先登記的預測寫在實驗執行**之前**,作為未事後編造解釋的證據。
