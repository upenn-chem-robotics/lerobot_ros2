#!/usr/bin/env bash
# Delete every lerobot-data backup repo on Hugging Face, then re-upload from USB.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXTRA_REPOS=(
  lerobot-data-smrithi-bimanual_pick_20260624
  lerobot-data-smrithi-bimanual_pick_20260624_success
  lerobot-data-smrithi-blue_dispense_20260617
  lerobot-data-smrithi-needle_20260629
  lerobot-data-smrithi-needle_20260706
)

echo "=== Step 1: delete all HF backup repos ==="
args=(--delete-all --yes)
for repo in "${EXTRA_REPOS[@]}"; do
  args+=(--extra-repo "$repo")
done
PYTHONPATH=src python -m lerobot_ros2.cli.backup "${args[@]}"

echo ""
echo "=== Step 2: upload USB data ==="
exec ./scripts/backup_usb_upenn.sh "$@"
