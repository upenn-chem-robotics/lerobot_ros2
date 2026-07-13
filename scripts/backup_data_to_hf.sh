#!/usr/bin/env bash
# Back up experiment folders under DATA_ROOT to Hugging Face.
#
# Usage:
#   ./scripts/backup_data_to_hf.sh              # upload all
#   ./scripts/backup_data_to_hf.sh --verify     # verify only
#   ./scripts/backup_data_to_hf.sh --dry-run    # show repo mapping
#
# On host machine:
#   DATA_ROOT=/home/rama/data ./scripts/backup_data_to_hf.sh
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/rama/data}"

declare -A FOLDERS=(
  [bimanual_pick_20260624]=lerobot-data-smrithi-bimanual_pick_20260624
  [blue_dispense_20260617]=lerobot-data-smrithi-blue_dispense_20260617
  [dose_solid]=lerobot-data-rama-dose_solid
  [funnel_insert_reactor]=lerobot-data-rama-funnel_insert_reactor
  [liquid_pouring]=lerobot-data-rama-liquid_pouring
  [pick_vial_20260524]=lerobot-data-smrithi-pick_vial_20260524
  [septum_insert_reactor]=lerobot-data-rama-septum_insert_reactor
  [stir_bar]=lerobot-data-rama-stir_bar
)

MODE=upload
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --verify) MODE=verify ;;
    --dry-run) MODE=dry-run ;;
    *) EXTRA+=("$arg") ;;
  esac
done

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "DATA_ROOT not found: $DATA_ROOT" >&2
  exit 1
fi

for name in "${!FOLDERS[@]}"; do
  repo="${FOLDERS[$name]}"
  path="$DATA_ROOT/$name"
  echo "========================================"
  echo "$MODE: $path -> $repo"
  echo "========================================"
  if [[ ! -d "$path" ]]; then
    echo "SKIP: $path does not exist" >&2
    continue
  fi
  case "$MODE" in
    verify)
      lerobot-ros-backup --verify --repo "$repo" "$path" "${EXTRA[@]}"
      ;;
    dry-run)
      lerobot-ros-backup --dry-run --repo "$repo" "$path" "${EXTRA[@]}"
      ;;
    upload)
      lerobot-ros-backup --repo "$repo" "$path" "${EXTRA[@]}"
      ;;
  esac
  echo ""
done

echo "Done."
