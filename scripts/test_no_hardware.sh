#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-$ROOT/release-logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/no-hardware-tests-$STAMP.log}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-lerobot-ros2:no-hardware-test}"
DEV_IMAGE="${DEV_IMAGE:-lerobot-ros2:no-hardware-test-dev}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-$((100 + 10#$(date -u +%S)))}"
SKIP_BUILD="${SKIP_BUILD:-0}"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_FILE") 2>&1

COMPOSE_TEST_ROOT=""
cleanup() {
  local status=$?
  if [[ -n "$COMPOSE_TEST_ROOT" ]]; then rm -rf "$COMPOSE_TEST_ROOT"; fi
  printf '\nLog: %s\n' "$LOG_FILE"
  if (( status == 0 )); then
    echo 'NO-HARDWARE INTEGRATION TESTS: PASS'
  else
    echo 'NO-HARDWARE INTEGRATION TESTS: FAIL'
  fi
  exit "$status"
}
trap cleanup EXIT

echo "Started: $(date -u --iso-8601=seconds)"
echo "Runtime image: $RUNTIME_IMAGE"
echo "Development image: $DEV_IMAGE"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"

docker compose config --quiet

if [[ "$SKIP_BUILD" != "1" ]]; then
  docker build --progress=plain --target runtime -t "$RUNTIME_IMAGE" .
  docker build --progress=plain --target dev -t "$DEV_IMAGE" .
fi

docker run --rm "$RUNTIME_IMAGE" python -m pip check
docker run --rm "$RUNTIME_IMAGE" ros2 doctor --report

# Verify that Compose runs the released image as the invoking host user and that
# files created through bind mounts retain that user's numeric ownership.
COMPOSE_TEST_ROOT="$(mktemp -d)"
mkdir -p "$COMPOSE_TEST_ROOT"/{data,config,cache/huggingface,cache/torch}
cp config/gello.example.yaml "$COMPOSE_TEST_ROOT/config/gello.yaml"
LEROBOT_ROS_IMAGE="${RUNTIME_IMAGE%:*}" \
IMAGE_TAG="${RUNTIME_IMAGE##*:}" \
LEROBOT_HOST_UID="$(id -u)" \
LEROBOT_HOST_GID="$(id -g)" \
LEROBOT_DATA="$COMPOSE_TEST_ROOT/data" \
LEROBOT_CONFIG="$COMPOSE_TEST_ROOT/config" \
LEROBOT_CACHE="$COMPOSE_TEST_ROOT/cache" \
docker compose run --rm tools sh -eu -c \
  'test "$(id -u)" = "$LEROBOT_HOST_UID"; test "$(id -g)" = "$LEROBOT_HOST_GID"; touch /data/ownership-test'
test "$(stat -c %u "$COMPOSE_TEST_ROOT/data/ownership-test")" = "$(id -u)"
test "$(stat -c %g "$COMPOSE_TEST_ROOT/data/ownership-test")" = "$(id -g)"

# The tests are mounted read-only but execute against packages installed in the image.
docker run --rm \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  -e ROS_LOCALHOST_ONLY=1 \
  -v "$ROOT/tests:/validation/tests:ro" \
  -w /validation \
  "$DEV_IMAGE" \
  pytest -q -o cache_dir=/tmp/pytest-cache \
    tests/integration/test_ros_graph_smoke.py \
    tests/integration/test_console_scripts.py \
    tests/test_action_history_diffusion.py
