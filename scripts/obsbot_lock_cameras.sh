#!/usr/bin/env bash
#
# Lock every connected OBSBOT Meet SE for data collection.
#
# WHY: the Meet SE's AI auto-framing continuously drives an internal zoom
# (v4l2 shows zoom_continuous jumping out of range, e.g. 245) as the gripper
# moves near the lens. That focal-length drift corrupts the geometry a
# diffusion policy relies on for mm-accurate needle insertion. Auto-framing is
# an OBSBOT SDK feature, not a v4l2 control, so config/gello.yaml can't touch
# it — this script does, via the CLI built by scripts/setup_obsbot_cli.sh.
#
# What it does to EVERY detected OBSBOT device (no per-camera mapping needed,
# because these settings are identical for all of them):
#   - media mode  -> Normal   (disables Auto-Framing AND background modes)
#   - HDR         -> Off       (HDR retimes/reframes; keep exposure stable)
#   - digital zoom-> 1.0x      (fixed, widest field of view)
# Optionally, with --focus N, also pins manual focus (0-100) on every device.
# By default focus is left to the v4l2 pipeline (config/gello.yaml's
# focus_absolute), which already applies and verifies per camera at record time.
#
# Run this at the START of every recording / deploy session, AFTER the cameras
# are plugged in and BEFORE `lerobot-ros-record` / `lerobot-ros-deploy`.
#
# This script and config/gello.yaml are complements, not alternatives: gello.yaml
# owns every standard V4L2 control, this owns the SDK-only ones. See the
# "gello.yaml vs obsbot-cli" section of README.md for the full split.
#
# Usage:
#   scripts/obsbot_lock_cameras.sh
#   scripts/obsbot_lock_cameras.sh --focus 35
#   scripts/obsbot_lock_cameras.sh --zoom 1.0 --cli /path/to/obsbot-cli
#
set -euo pipefail

ZOOM="1.0"
FOCUS=""          # empty => don't touch focus (leave it to v4l2/gello.yaml)
CLI="${OBSBOT_CLI:-}"

usage() { sed -n '2,32p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --focus) FOCUS="${2:?--focus needs a value 0-100}"; shift 2;;
    --zoom)  ZOOM="${2:?--zoom needs a value 1.0-4.0}"; shift 2;;
    --cli)   CLI="${2:?--cli needs a path}"; shift 2;;
    -h|--help) usage 0;;
    *) echo "Unknown arg: $1" >&2; usage 1;;
  esac
done

# Resolve the CLI binary if not given via $OBSBOT_CLI.
if [[ -z "$CLI" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  for cand in \
    "$PROJECT_ROOT/third_party/obsbot-meetse-cli/target/release/obsbot-cli" \
    "$PROJECT_ROOT/third_party/obsbot-meetse-cli/target/debug/obsbot-cli"; do
    [[ -x "$cand" ]] && CLI="$cand" && break
  done
fi

if [[ -z "$CLI" || ! -x "$CLI" ]]; then
  echo "ERROR: obsbot-cli not found. Build it first:" >&2
  echo "       scripts/setup_obsbot_cli.sh" >&2
  echo "   then re-run, or pass --cli /path/to/obsbot-cli (or set \$OBSBOT_CLI)." >&2
  exit 1
fi

echo ">> Using CLI: $CLI"
echo ">> Enumerating OBSBOT devices..."
LIST_OUT="$("$CLI" list || true)"
echo "$LIST_OUT"

# Parse serials from lines like: "Device 0: OBSBOT Meet SE (SN: ABC123)"
mapfile -t SERIALS < <(printf '%s\n' "$LIST_OUT" | grep -oP 'SN:\s*\K[^)]+' | sed 's/[[:space:]]*$//')

if [[ "${#SERIALS[@]}" -eq 0 ]]; then
  echo "ERROR: no OBSBOT serials parsed from 'list'. Are the cameras plugged in" >&2
  echo "       and is your user in the 'video' group?" >&2
  exit 1
fi

echo ">> Locking ${#SERIALS[@]} device(s): ${SERIALS[*]}"
fail=0
for sn in "${SERIALS[@]}"; do
  echo
  echo "--- $sn ---"
  # Disable auto-framing (and background modes) by forcing Normal media mode.
  "$CLI" mode   --sn "$sn" normal        || { echo "  !! mode normal failed"  >&2; fail=1; }
  # HDR off for stable exposure/geometry.
  "$CLI" hdr    --sn "$sn" off           || { echo "  !! hdr off failed"      >&2; fail=1; }
  # Fixed digital zoom.
  "$CLI" camera --sn "$sn" --zoom "$ZOOM" || { echo "  !! zoom set failed"     >&2; fail=1; }
  # Optional manual focus lock.
  if [[ -n "$FOCUS" ]]; then
    "$CLI" camera --sn "$sn" --focus-auto false --focus "$FOCUS" \
      || { echo "  !! focus set failed" >&2; fail=1; }
  fi
  # Read back media mode as verification.
  echo "  verify:"
  "$CLI" info --sn "$sn" | sed -n 's/^/    /p' | grep -E 'Media Mode|HDR|Serial' || true
done

echo
if [[ "$fail" -ne 0 ]]; then
  echo ">> Completed WITH ERRORS — check messages above before recording." >&2
  exit 1
fi
echo ">> All devices report Media Mode: Normal. Auto-framing/gesture zoom disabled."
echo ">> Tip: while moving the gripper near the front cam, watch it stay put:"
echo "     watch -n0.5 'v4l2-ctl -d /dev/video2 --get-ctrl=zoom_continuous,zoom_absolute'"
