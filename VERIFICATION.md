# Verification delta

Extract this archive at the repository root after the public-release delta.
Run:

```bash
./sh/verify_public_release.sh
```

The script writes a complete timestamped log and a compact summary under `release-logs/`. It continues after individual failures so the log captures all checks. The directory is ignored through `.gitignore.verification`; append that file to `.gitignore` or keep the generated logs untracked.

## Hardware-free integration validation

Run `./sh/test_no_hardware.sh` beside `sh/verify_public_release.sh` to build the runtime and development images and collect a timestamped log under `release-logs/`. The script validates dependency consistency, the ROS installation, an actual ROS 2 publish/subscribe round trip, execution of every installed `lerobot_ros2` console script through `--help`, and the action-history policy forward-path tests.

To reuse images that were already built, set `SKIP_BUILD=1` together with `RUNTIME_IMAGE` and `DEV_IMAGE`. These tests intentionally do not claim hardware, checkpoint, or external-dataset coverage.
