#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -f SECURITY.md
test -f CONTRIBUTING.md
docker compose config --quiet
if git ls-files | grep -Eq '^(data|outputs|config.local|third_party)/'; then
  echo "private/runtime paths are tracked" >&2; exit 1
fi
docker build --target runtime -t lerobot-ros2:release-check .
docker run --rm lerobot-ros2:release-check python -m pip check
echo "release check passed"
