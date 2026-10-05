#!/usr/bin/env bash
set -euo pipefail

# Conda activation is required for ROS 2 environment hooks and shared libraries.
# RoboStack activation hooks reference variables that may be unset outside a
# conda-build process. Temporarily disable nounset while Conda runs its hooks.
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate lerobot
set -u

if [[ "${1:-}" == "bash" ]] && [[ ! -t 0 ]]; then
  echo "warning: bash started without a TTY; use docker compose run --rm robot bash" >&2
fi
exec "$@"
