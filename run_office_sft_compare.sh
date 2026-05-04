#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/output/office_rqkmeans_plus/sft/logs}"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"

MODES=(
  original_full
  original_freeze_new_tokens
  new_lora
)

for mode in "${MODES[@]}"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_ROOT/${mode}_${ts}.log"
  echo "[$(date)] start $mode log=$log"
  bash "$REPO_ROOT/sft_office_rqkmeans_plus.sh" "$mode" 2>&1 | tee "$log"
  echo "[$(date)] done $mode"
done

echo "[$(date)] all Office_Products rqkmeans_plus SFT runs finished"
