# Workflows

> **Page scope:** This page owns command-oriented procedures. If you are still deciding what operation you need, start with [Choose a capability](capabilities.md).

All commands below run inside the released containers. Start by inspecting the installed parser because flags are versioned with the command:

```bash
docker compose run --rm tools lerobot-ros-export --help
```

## Diagnose the runtime

```bash
docker compose run --rm tools   lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

With the hardware override configured, run the full preflight before any motion:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot   lerobot-ros-doctor
```

## Probe cameras

Use the camera probe before writing camera identifiers into local configuration:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot   lerobot-ros-probe-cameras --help
```

Record stable `/dev/v4l/by-id` paths, not transient `/dev/videoN` numbers.

## Record a dataset

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot   lerobot-ros-record --help
```

Before recording, validate camera streams, operator input, ROS topics, the destination under `/data`, free space, and the no-motion state. After recording, allow the process to finalize files before stopping containers or powering down hardware.

## Inspect and transform datasets

The release provides separate commands so each transformation can be run and reviewed independently:

```bash
docker compose run --rm tools lerobot-ros-canonicalize --help
docker compose run --rm tools lerobot-ros-reorient --help
docker compose run --rm tools lerobot-ros-downsample --help
docker compose run --rm tools lerobot-ros-export --help
```

Use a new output location for destructive or lossy transformations. Validate episode counts, timestamps, frame rate, observation keys, action dimensions, camera orientation, and metadata before replacing a source dataset.

## Back up datasets

```bash
docker compose run --rm tools lerobot-ros-backup --help
```

Keep authentication outside committed files. Verify the destination repository and dataset identity before upload, and retain a local copy until the remote artifact has been checked.

## Train a policy

```bash
docker compose run --rm gpu lerobot-ros-train --help
```

The repository includes two policy plugins:

- `strided_diffusion`: uniformly spaced observation history; configure `DATASET_FPS` and `STRIDE_SECONDS`; predicted actions remain contiguous.
- `action_history_diffusion`: conditions on prior commanded actions through an MLP; `n_action_history=0` matches the stock DiffusionPolicy conditioning path, while dropout is available to reduce dependence on action history.

Record the dataset identity, configuration, code commit, dependency pins, random seed, and output checkpoint for every training run.

## Deploy a checkpoint

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot   lerobot-ros-deploy --help
```

Deployment crosses the observation-to-command boundary. Load and inspect the checkpoint without motion first. Confirm observation names and shapes, action dimensions, normalization metadata, control frequency, limits, emergency stop, and workspace clearance before enabling commands.

## DAgger

The repository separates collection/deployment and training commands:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot   lerobot-ros-dagger --help
docker compose run --rm gpu lerobot-ros-train-dagger --help
```

Treat every DAgger iteration as a new dataset and checkpoint lineage. Preserve the policy version, intervention source, episode metadata, and validation outcome.

## Visualization

```bash
docker compose run --rm tools lerobot-ros-app --help
```

The visualization application depends on the visualization extras included in the release image. Do not expose the application on an untrusted network or place credentials in its arguments.
