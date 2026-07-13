#!/bin/bash
set -euo pipefail
cd /lerobot-ros
LOG=/lerobot-ros/logs/hf_upload_rama.log
folders=(
  data/stir_bar
  data/funnel_insert_reactor
  data/pick_vial_20260524
  data/bimanual_pick_20260624
  data/dose_solid
  data/blue_dispense_20260617
  data/septum_insert_reactor
  data/liquid_pouring
)
echo "=== HF upload started $(date -Iseconds) ===" | tee -a "$LOG"
for f in "${folders[@]}"; do
  echo "--- Uploading $f $(date -Iseconds) ---" | tee -a "$LOG"
  if lerobot-ros-backup "$f" 2>&1 | tee -a "$LOG"; then
    echo "[OK] $f $(date -Iseconds)" | tee -a "$LOG"
  else
    echo "[FAIL] $f $(date -Iseconds)" | tee -a "$LOG"
    exit 1
  fi
done
echo "=== All uploads complete $(date -Iseconds) ===" | tee -a "$LOG"
