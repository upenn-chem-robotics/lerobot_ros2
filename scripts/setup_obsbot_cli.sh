#!/usr/bin/env bash
#
# Build the OBSBOT Meet SE control CLI (imaviso/obsbot-meetse-cli).
#
# Our cameras (front + both wrists) are OBSBOT Meet SE AI webcams. Their
# auto-framing / gesture zoom is NOT a standard v4l2 control, so it can't be
# pinned from config/gello.yaml. It has to be turned off through OBSBOT's own
# SDK. This script vendors and builds the community CLI that wraps that SDK;
# scripts/obsbot_lock_cameras.sh then uses the built binary to disable
# auto-framing on every camera before recording.
#
# Usage:
#   scripts/setup_obsbot_cli.sh                 # clone + build into third_party/
#   OBSBOT_CLI_DIR=/opt/obsbot scripts/setup_obsbot_cli.sh
#
# After a successful build the script prints the binary path. Export it so the
# lock script can find it:
#   export OBSBOT_CLI=/lerobot-ros/third_party/obsbot-meetse-cli/target/release/obsbot-cli
#
set -euo pipefail

REPO_URL="https://github.com/imaviso/obsbot-meetse-cli"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${OBSBOT_CLI_DIR:-$PROJECT_ROOT/third_party/obsbot-meetse-cli}"

echo ">> OBSBOT CLI destination: $DEST"

if [[ -d "$DEST/.git" ]]; then
  echo ">> Repo already present, pulling latest"
  git -C "$DEST" pull --ff-only
else
  mkdir -p "$(dirname "$DEST")"
  echo ">> Cloning $REPO_URL"
  git clone --depth 1 "$REPO_URL" "$DEST"
fi

cd "$DEST"

# The SDK is proprietary and bundled in the repo under sdk/. bindgen (via
# build.rs) needs libclang, and the linker needs the SDK's .so at build and
# run time. Nix is the upstream-supported path; plain cargo works too if the
# host already has clang + a C++ toolchain.
build_ok=0
if command -v nix >/dev/null 2>&1; then
  echo ">> Building via 'nix develop' (upstream-recommended)"
  if nix develop --command cargo build --release; then
    build_ok=1
  else
    echo "!! nix develop build failed; falling back to plain cargo" >&2
  fi
fi

if [[ "$build_ok" -eq 0 ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: neither a working 'nix' build nor 'cargo' is available." >&2
    echo "       Install Rust (https://rustup.rs) and libclang, or install Nix" >&2
    echo "       with flakes enabled, then re-run this script." >&2
    exit 1
  fi
  echo ">> Building via plain 'cargo build --release'"
  cargo build --release
  build_ok=1
fi

BIN="$DEST/target/release/obsbot-cli"
if [[ ! -x "$BIN" ]]; then
  # nix develop + cargo build (no --release passthrough on some setups) may
  # land in debug/; accept either.
  if [[ -x "$DEST/target/debug/obsbot-cli" ]]; then
    BIN="$DEST/target/debug/obsbot-cli"
  else
    echo "ERROR: build reported success but no obsbot-cli binary was found." >&2
    exit 1
  fi
fi

echo
echo ">> Build complete: $BIN"
echo ">> Sanity check (lists connected OBSBOT devices):"
"$BIN" list || echo "   (no devices found now — that's fine if cameras aren't plugged in)"
echo
echo ">> Add this to your shell / launch env so the lock script finds the binary:"
echo "   export OBSBOT_CLI=$BIN"
