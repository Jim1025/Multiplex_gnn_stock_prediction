#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# backup_mlruns.sh — 備份 MLflow 訓練紀錄、模型權重、預測 CSV
#
# 用法：
#   bash scripts/backup_mlruns.sh                       # 預設備份到 backups/
#   bash scripts/backup_mlruns.sh /path/to/backup/dir   # 指定目錄
#   BACKUP_DIR=/Volumes/SSD/magnet bash scripts/backup_mlruns.sh
#
# 產出：
#   <backup_dir>/magnet_mlruns_YYYY-MM-DD_HHMM.tar.gz
#   壓縮內容：mlruns/ + checkpoints/ + predictions/ + configs/base.yaml
#
# 安全性：
#   - 只讀，不刪除任何原始檔案
#   - 若同名備份已存在，會加上 _2 / _3 …序號
#   - 結束時印出 sha256 校驗碼與大小
# ---------------------------------------------------------------------------

set -euo pipefail

# ── 解析參數 ───────────────────────────────────────────────────────
DEFAULT_BACKUP_DIR="backups"
BACKUP_DIR="${1:-${BACKUP_DIR:-$DEFAULT_BACKUP_DIR}}"

# 切換到 project root（腳本位於 scripts/，往上一層）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── 預檢 ───────────────────────────────────────────────────────────
if [[ ! -d "mlruns" ]]; then
    echo "❌ 找不到 mlruns/，請在 project root 跑此腳本" >&2
    exit 1
fi

# 統計待備份內容
N_RUNS=$(find mlruns -mindepth 2 -maxdepth 2 -type d ! -name ".trash" | wc -l | tr -d ' ')
if [[ "$N_RUNS" == "0" ]]; then
    echo "⚠️  mlruns/ 內沒有任何 run，仍會備份空目錄。"
fi

# ── 決定輸出檔名 ───────────────────────────────────────────────────
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +"%Y-%m-%d_%H%M")
BASE_NAME="magnet_mlruns_${TIMESTAMP}"
OUT_FILE="${BACKUP_DIR}/${BASE_NAME}.tar.gz"

# 同名衝突時加序號
i=2
while [[ -e "$OUT_FILE" ]]; do
    OUT_FILE="${BACKUP_DIR}/${BASE_NAME}_${i}.tar.gz"
    i=$((i + 1))
done

# ── 收集要備份的路徑（只挑存在的）──────────────────────────────────
INCLUDE=()
for p in mlruns checkpoints predictions configs/base.yaml; do
    if [[ -e "$p" ]]; then
        INCLUDE+=("$p")
    fi
done

echo "─────────────────────────────────────────────"
echo "📦 MAGNET MLflow 備份"
echo "─────────────────────────────────────────────"
echo "Project root : $PROJECT_ROOT"
echo "Backup dir   : $BACKUP_DIR"
echo "Output file  : $OUT_FILE"
echo "MLflow runs  : $N_RUNS"
echo "Include      : ${INCLUDE[*]}"
echo "─────────────────────────────────────────────"

# ── 打包（排除暫存與 .DS_Store）────────────────────────────────────
tar \
    --exclude='.DS_Store' \
    --exclude='mlruns/.trash' \
    --exclude='__pycache__' \
    -czf "$OUT_FILE" \
    "${INCLUDE[@]}"

# ── 產出資訊 ───────────────────────────────────────────────────────
SIZE=$(du -h "$OUT_FILE" | cut -f1)
if command -v shasum >/dev/null 2>&1; then
    SHA=$(shasum -a 256 "$OUT_FILE" | cut -d' ' -f1)
else
    SHA="(shasum 未安裝，略過)"
fi

echo ""
echo "✅ 備份完成"
echo "  檔案 : $OUT_FILE"
echo "  大小 : $SIZE"
echo "  sha256: $SHA"
echo ""
echo "💡 還原指令："
echo "    tar -xzf \"$OUT_FILE\" -C /path/to/restore/dir"
