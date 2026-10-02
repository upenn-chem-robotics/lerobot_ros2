#!/usr/bin/env bash
# Optional, separately licensed OBSBOT CLI bootstrap.
set -euo pipefail
: "${OBSBOT_CLI_REPOSITORY:?Set the reviewed upstream URL}"
: "${OBSBOT_CLI_COMMIT:?Set a reviewed full commit SHA}"
[[ "$OBSBOT_CLI_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "OBSBOT_CLI_COMMIT must be a 40-character SHA" >&2; exit 2; }
root="${OBSBOT_CLI_ROOT:-third_party/obsbot-cli}"
mkdir -p "$(dirname "$root")"
[[ -d "$root/.git" ]] || git clone --filter=blob:none "$OBSBOT_CLI_REPOSITORY" "$root"
git -C "$root" fetch --depth 1 origin "$OBSBOT_CLI_COMMIT"
git -C "$root" checkout --detach FETCH_HEAD
printf 'Checked out OBSBOT CLI at %s\n' "$(git -C "$root" rev-parse HEAD)"
printf 'Follow reviewed upstream build instructions; the proprietary SDK is not vendored.\n'
