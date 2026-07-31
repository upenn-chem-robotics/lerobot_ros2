#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="${LOG:-/lerobot-ros/logs/hf_backup_smrithi.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

DATASETS=(
  cap_20260717
  needle_20260629_old
  needle_20260707_success
  salt_20260725
  uncap_20260713
)

for d in "${DATASETS[@]}"; do
  path="data/smrithi/${d}"
  echo ""
  echo "========================================"
  echo "$(date -Iseconds) START upload: $path"
  echo "========================================"
  if lerobot-ros-backup --verify "$path" 2>/dev/null; then
    echo "$(date -Iseconds) SKIP $path (already complete on Hub)"
    continue
  fi
  lerobot-ros-backup "$path"
  if lerobot-ros-backup --verify "$path"; then
    echo "$(date -Iseconds) OK $path fully mirrored"
  else
    echo "$(date -Iseconds) WARNING $path still incomplete after upload"
    exit 1
  fi
done

echo ""
echo "$(date -Iseconds) ALL BACKUPS COMPLETE"
