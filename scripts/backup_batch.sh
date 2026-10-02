#!/usr/bin/env bash
set -euo pipefail
CONFIG="${HF_BACKUP_CONFIG:-/config/hf-backup.yaml}"
exec lerobot-ros-backup --config "$CONFIG" batch "$@"
