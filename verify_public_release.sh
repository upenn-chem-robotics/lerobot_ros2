#!/usr/bin/env bash
set -uo pipefail

# Full public-release verification. Run from the repository root.
# It continues after failures so one log captures the complete state.

ROOT="$(pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${RELEASE_LOG_DIR:-$ROOT/release-logs}"
LOG_FILE="$LOG_DIR/public-release-verification-$STAMP.log"
SUMMARY_FILE="$LOG_DIR/public-release-verification-$STAMP.summary"
IMAGE="${RELEASE_IMAGE:-lerobot-ros2:release-check-$STAMP}"
DEV_IMAGE="${RELEASE_DEV_IMAGE:-lerobot-ros2:dev-check-$STAMP}"
CONFIG_DIR="$LOG_DIR/config-$STAMP"
DATA_DIR="$LOG_DIR/data-$STAMP"
KEEP_IMAGES="${KEEP_RELEASE_IMAGES:-1}"
TRIVY_TIMEOUT="${TRIVY_TIMEOUT:-20m}"
COLD_BUILD="${COLD_RELEASE_BUILD:-0}"
NOTIFY_ON_FINISH="${NOTIFY_ON_FINISH:-1}"
FAILURES=0
WARNINGS=0

mkdir -p "$LOG_DIR" "$CONFIG_DIR" "$DATA_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'Public release verification\n'
printf 'Started (UTC): %s\n' "$(date -u -Is)"
printf 'Repository: %s\n' "$ROOT"
printf 'Runtime image: %s\n' "$IMAGE"
printf 'Development image: %s\n' "$DEV_IMAGE"
printf 'Log: %s\n' "$LOG_FILE"
printf 'Cold build: %s\n' "$COLD_BUILD"
printf 'Keep images: %s\n\n' "$KEEP_IMAGES"

cleanup() {
  local rc=$?
  trap - EXIT
  if [[ "$KEEP_IMAGES" != "1" ]] && command -v docker >/dev/null 2>&1; then
    docker image rm -f "$IMAGE" "$DEV_IMAGE" >/dev/null 2>&1 || true
  fi
  {
    printf 'PUBLIC RELEASE VERIFICATION SUMMARY\n'
    printf 'Completed (UTC): %s\n' "$(date -u -Is)"
    printf 'Failures: %d\n' "$FAILURES"
    printf 'Warnings: %d\n' "$WARNINGS"
    printf 'Log: %s\n' "$LOG_FILE"
    if (( FAILURES == 0 )); then
      printf 'Result: PASS\n'
    else
      printf 'Result: FAIL\n'
    fi
  } | tee "$SUMMARY_FILE"
  printf '\nSummary: %s\n' "$SUMMARY_FILE"
  if [[ "$NOTIFY_ON_FINISH" == "1" ]]; then
    result="PASS"; (( FAILURES > 0 )) && result="FAIL"
    message="Result: ${result}; failures: ${FAILURES}; warnings: ${WARNINGS}"
    if command -v send-notify >/dev/null 2>&1; then
      send-notify "lerobot-ros2 verification" "$message" >/dev/null 2>&1 || true
    elif command -v notify-send >/dev/null 2>&1; then
      notify-send "lerobot-ros2 verification" "$message" >/dev/null 2>&1 || true
    fi
  fi
  if (( FAILURES > 0 )); then exit 1; fi
  exit "$rc"
}
trap cleanup EXIT

section() {
  printf '\n================================================================================\n'
  printf '%s\n' "$1"
  printf '================================================================================\n'
}

run() {
  local name="$1"; shift
  section "$name"
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if "$@"; then
    printf '[PASS] %s\n' "$name"
  else
    local rc=$?
    printf '[FAIL] %s (exit %d)\n' "$name" "$rc"
    FAILURES=$((FAILURES + 1))
    return "$rc"
  fi
  return 0
}

warn_run() {
  local name="$1"; shift
  section "$name"
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if "$@"; then
    printf '[PASS] %s\n' "$name"
  else
    local rc=$?
    printf '[WARN] %s (exit %d)\n' "$name" "$rc"
    WARNINGS=$((WARNINGS + 1))
  fi
}

require_command() {
  if command -v "$1" >/dev/null 2>&1; then
    printf '[PASS] command available: %s -> %s\n' "$1" "$(command -v "$1")"
  else
    printf '[FAIL] required command missing: %s\n' "$1"
    FAILURES=$((FAILURES + 1))
  fi
}

section 'Host and repository metadata'
uname -a || true
printf '\n'
require_command git
require_command docker
require_command python3
if command -v git >/dev/null 2>&1; then
  git rev-parse --show-toplevel || true
  git rev-parse HEAD || true
  git status --short || true
fi
if command -v docker >/dev/null 2>&1; then
  docker version || true
  docker compose version || true
fi

run 'Repository root check' bash -lc 'test -f Dockerfile && test -f compose.yaml && test -f pyproject.toml && test -d src && test -d tests'
run 'No forbidden runtime paths tracked' bash -lc "! git ls-files | grep -E '^(data|outputs|config.local|third_party)/'"
run 'No unresolved merge markers' bash -lc "! git grep -nE '^(<<<<<<<|=======|>>>>>>>)' -- . ':!verify_public_release.sh'"
run 'Required public-release files exist' bash -lc 'for f in LICENSE SECURITY.md CONTRIBUTING.md MIGRATION.md environment.yml requirements.lock.txt dependencies.env docker/entrypoint.sh docs/release-process.md examples/gello.yaml examples/hf-backup.yaml .gitleaksignore GITLEAKS_FINDINGS_REVIEW.md; do test -e "$f" || { echo "missing: $f"; exit 1; }; done'

run 'Python, TOML, and YAML static validation' python3 - <<'PY'
from pathlib import Path
import ast
import tomllib
import yaml
root = Path('.')
for path in list((root / 'src').rglob('*.py')) + list((root / 'tests').rglob('*.py')):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
for path in [
    root / 'pyproject.toml',
    root / 'packages/lerobot_policy_strided_diffusion/pyproject.toml',
    root / 'packages/lerobot_policy_action_history_diffusion/pyproject.toml',
]:
    tomllib.loads(path.read_text(encoding='utf-8'))
for path in [
    root / 'environment.yml', root / 'compose.yaml', root / 'examples/gello.yaml',
    root / 'examples/hf-backup.yaml', root / '.github/workflows/ci.yml',
]:
    yaml.safe_load(path.read_text(encoding='utf-8'))
print('Static parsing passed.')
PY

run 'Docker Compose validation' docker compose config --quiet

RUNTIME_READY=0
DEV_READY=0
BUILD_CACHE_ARGS=()
if [[ "$COLD_BUILD" == "1" ]]; then BUILD_CACHE_ARGS+=(--no-cache); fi
if run 'Docker runtime image build' docker build "${BUILD_CACHE_ARGS[@]}" --progress=plain --target runtime -t "$IMAGE" .; then
  RUNTIME_READY=1
fi
if run 'Docker development image build' docker build "${BUILD_CACHE_ARGS[@]}" --progress=plain --target dev -t "$DEV_IMAGE" .; then
  DEV_READY=1
fi

if [[ -f examples/gello.yaml ]]; then
  cp examples/gello.yaml "$CONFIG_DIR/gello.yaml"
fi
chmod -R a+rwX "$DATA_DIR" || true

if (( RUNTIME_READY )); then
  run 'Runtime pip dependency check' docker run --rm "$IMAGE" python -m pip check
  run 'Runtime ROS doctor report' docker run --rm "$IMAGE" ros2 doctor --report
  run 'Runtime application preflight' docker run --rm -e GELLO_CONFIG=/config/gello.yaml -v "$CONFIG_DIR:/config:ro" -v "$DATA_DIR:/data" "$IMAGE" lerobot-ros-doctor --skip-hardware --skip-ros-graph
  run 'Console entry-point import smoke test' docker run --rm "$IMAGE" python -c "import importlib; from importlib.metadata import entry_points; eps=[e for e in entry_points(group='console_scripts') if e.module.startswith('lerobot_ros2')]; assert eps; [importlib.import_module(e.module) for e in eps]; print('Imported', len(eps), 'console scripts')"
  run 'Policy distribution discovery' docker run --rm "$IMAGE" python -c "from importlib.metadata import distributions; from packaging.utils import canonicalize_name; n={canonicalize_name(d.metadata['Name']) for d in distributions() if d.metadata['Name']}; e={canonicalize_name(x) for x in ('lerobot-policy-strided-diffusion','lerobot-policy-action-history-diffusion')}; print('missing',e-n); assert not e-n"
else
  printf '[SKIP] Runtime checks: runtime image build failed\n'
fi

if (( DEV_READY )); then
  run 'Pytest in development image' docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$ROOT:/workspace:ro" -w /workspace "$DEV_IMAGE" pytest -o cache_dir=/tmp/pytest-cache
  run 'Ruff in development image' docker run --rm -v "$ROOT:/workspace:ro" -w /workspace "$DEV_IMAGE" ruff check --no-cache .
else
  printf '[SKIP] Development checks: development image build failed\n'
fi

run_gitleaks() {
  if gitleaks git --help >/dev/null 2>&1; then
    gitleaks git --redact --no-banner --report-format json --report-path "$LOG_DIR/gitleaks-$STAMP.json"
  elif gitleaks detect --help >/dev/null 2>&1; then
    gitleaks detect --source . --redact --no-banner --report-format json --report-path "$LOG_DIR/gitleaks-$STAMP.json"
  else
    echo 'Unsupported Gitleaks CLI' >&2
    gitleaks version || true
    gitleaks --help || true
    return 2
  fi
}

if command -v gitleaks >/dev/null 2>&1; then
  run 'Gitleaks full-history scan' run_gitleaks
else
  warn_run 'Gitleaks full-history scan through Docker' docker run --rm -v "$ROOT:/repo" -w /repo zricethezav/gitleaks:latest detect --source /repo --redact --no-banner --report-format json --report-path "/repo/release-logs/gitleaks-$STAMP.json"
fi

if (( RUNTIME_READY && FAILURES == 0 )); then
  if command -v trivy >/dev/null 2>&1; then
    run 'Trivy image scan' trivy image --scanners vuln --timeout "$TRIVY_TIMEOUT" --ignore-unfixed --severity HIGH,CRITICAL --exit-code 1 "$IMAGE"
  else
    warn_run 'Trivy image scan through Docker' docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest image --scanners vuln --timeout "$TRIVY_TIMEOUT" --ignore-unfixed --severity HIGH,CRITICAL --exit-code 1 "$IMAGE"
  fi
  section 'Image inspection'
  docker image inspect "$IMAGE" || { printf '[FAIL] image inspection\n'; FAILURES=$((FAILURES + 1)); }
  docker history --no-trunc "$IMAGE" || { printf '[FAIL] image history\n'; FAILURES=$((FAILURES + 1)); }
else
  printf '[SKIP] Trivy and image inspection: resolve earlier verification failures first\n'
fi

printf '\nVerification collection complete.\n'
