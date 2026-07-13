#!/usr/bin/env bash
# Back up experiment folders from the Samsung T5 USB stick to Hugging Face.
# Run on the host machine with the USB mounted at USB_ROOT.
#
# Usage:
#   ./scripts/backup_usb_upenn.sh              # upload all folders
#   ./scripts/backup_usb_upenn.sh --verify     # verify only
#   ./scripts/backup_usb_upenn.sh --dry-run    # show repo mapping
set -euo pipefail

USB_ROOT="${USB_ROOT:-/media/rama/Samsung_T5/upenn}"

# local folder name on USB -> HF repo name (without username prefix)
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

if [[ ! -d "$USB_ROOT" ]]; then
  echo "USB root not found: $USB_ROOT" >&2
  echo "Mount the Samsung T5 and/or set USB_ROOT=/path/to/upenn" >&2
  exit 1
fi

for name in "${!FOLDERS[@]}"; do
  repo="${FOLDERS[$name]}"
  path="$USB_ROOT/$name"
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
