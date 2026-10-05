# Choose and run a workflow

Use this page to select an operation, confirm its prerequisites, and run the corresponding command. If terms such as *episode*, *policy*, or *checkpoint* are unfamiliar, begin with [Concepts](concepts.md).

> Replace angle-bracket placeholders before running a command. Paths beginning with `/data` refer to the host `data/` directory mounted into the container.

## Quick chooser

| Goal | Workflow | Scope | Required input |
|---|---|---|---|
| Check the installed application | [Run the software-only preflight](#run-the-software-only-preflight) | Software-only | Pulled `tools` image |
| Inspect a dataset | [Visualize a dataset](#visualize-a-dataset) | Software-only | Existing LeRobot dataset |
| Create videos or timeline images | [Export dataset media](#export-dataset-media) | Software-only | Existing dataset |
| Confirm camera identities | [Probe configured cameras](#probe-configured-cameras) | Camera access | Camera configuration and hardware mappings |
| Collect demonstrations | [Record demonstrations](#record-demonstrations) | Robot-capable | Configured cameras, operator input, and robot interface |
| Normalize dataset structure | [Canonicalize a dataset](#canonicalize-a-dataset) | Software-only | Existing dataset |
| Correct camera orientation | [Reorient camera observations](#reorient-camera-observations) | Software-only | Existing dataset |
| Reduce dataset frame rate | [Downsample a dataset](#downsample-a-dataset) | Software-only | Existing dataset |
| Preserve an artifact externally | [Back up a dataset or checkpoint directory](#back-up-a-dataset-or-checkpoint-directory) | Network access | Artifact, destination configuration, and authentication |
| Produce a policy checkpoint | [Train a policy](#train-a-policy) | GPU-required | Compatible dataset and NVIDIA runtime |
| Run a policy on the robot | [Deploy a checkpoint](#deploy-a-checkpoint) | Robot-capable | Compatible checkpoint and completed acceptance checks |
| Collect policy corrections | [Collect DAgger data](#collect-dagger-data) | Robot-capable | Deployable checkpoint and intervention interface |
| Retrain with corrections | [Train from DAgger data](#train-from-dagger-data) | GPU-required | Compatible DAgger dataset and NVIDIA runtime |

The commands are independent building blocks. An experiment does not need to use every workflow or follow the table from top to bottom.

## Run the software-only preflight

**Scope:** software-only. **Requires:** the published `tools` image and local mounts. **Produces:** a runtime and configuration report; it does not validate a dataset, checkpoint, GPU, camera, ROS graph, or robot.

```bash
docker compose pull tools
docker compose run --rm tools \
  lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

Use this to separate packaging or configuration failures from later dataset, learning, or hardware problems.

## Visualize a dataset

**Scope:** software-only. **Requires:** an existing dataset below `/data`.

```bash
docker compose run --rm --service-ports tools \
  lerobot-ros-app \
  --dataset_dir /data/<dataset> \
  --port 7860
```

Open `http://localhost:7860` on the same workstation. Do not use `--share` on an untrusted network.

## Export dataset media

**Scope:** software-only. **Requires:** an existing dataset below `/data`.

```bash
docker compose run --rm tools \
  lerobot-ros-export \
  --dataset-dir /data/<dataset> \
  --output-dir /data/<dataset>/exports/grid_media
```

This writes grid videos and timeline images without replacing the source dataset.

## Start and verify the robot stack

**Scope:** robot-capable. **Requires:** the Humble `ur_robotiq` environment, site robot configuration, and the safety checklist.

The released robot runtime must be running before camera-enabled recording, deployment, or DAgger. Verify the state topics, command topics, GELLO services, controller state, and shared `ROS_DOMAIN_ID` before using the deployed hardware.

Follow [Bimanual UR3 and Robotiq integration](robot-integration.md) for the supported launch commands and interface checks. Do not treat `lerobot-ros-doctor --skip-ros-graph` as a robot integration check.

> **STOP: hardware boundary**
> The next workflows can access cameras or command a robot. Read [Hardware and safety](hardware-and-safety.md), replace every `REPLACE_*` value, review `compose.hardware.yaml`, verify stable device paths, complete no-motion checks, clear the workspace, and confirm emergency-stop access.

## Probe configured cameras

**Scope:** camera access. **Requires:** completed local configuration and hardware override.

Complete `config.local/gello.yaml` and `compose.hardware.yaml` first, then run:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot \
  lerobot-ros-probe-cameras \
  --config /config/gello.yaml \
  --output-dir /data/camera_probes
```

Use the generated camera reports to confirm stable device identity and settings before recording.

## Record demonstrations

**Scope:** robot-capable. **Requires:** cameras, operator inputs, completed hardware configuration, and controlled safety checks.

After the no-motion and safety checks:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot \
  lerobot-ros-record \
  --name <dataset-name> \
  --task "<task description>" \
  --config /config/gello.yaml
```

Use `--left` or `--right` only for a single-arm recording. Keep partial or interrupted recordings separate until their metadata and episode finalization have been checked.

## Canonicalize a dataset

**Scope:** software-only. **Requires:** an existing dataset; writes a new destination.

Write to a new destination:

```bash
docker compose run --rm tools \
  lerobot-ros-canonicalize \
  --src /data/<source-dataset> \
  --dst /data/<canonical-dataset> \
  --task-name "<task description>"
```

Do not point `--dst` at the source dataset.

## Reorient camera observations

**Scope:** software-only. **Requires:** an existing dataset; writes a new destination.

```bash
docker compose run --rm tools \
  lerobot-ros-reorient \
  --src /data/<source-dataset> \
  --dst /data/<reoriented-dataset> \
  --camera <camera-name>:<episode-spec>
```

Use a camera short name reported by `lerobot-ros-reorient --src /data/<source-dataset> --report`. The episode specification can be a single episode, a comma-separated list, a range, or `all`. The destination must differ from the source.

## Downsample a dataset

**Scope:** software-only. **Requires:** an existing dataset; writes a new destination.

```bash
docker compose run --rm tools \
  lerobot-ros-downsample \
  --src /data/<source-dataset> \
  --dst /data/<downsampled-dataset>
```

Keep the source unchanged until the destination has been inspected.

## Back up a dataset or checkpoint directory

**Scope:** network access. **Requires:** an artifact, destination configuration, and external authentication.

Configure `config.local/hf-backup.yaml`, authenticate outside committed files, then run:

```bash
docker compose run --rm tools \
  lerobot-ros-backup /data/<dataset-or-checkpoint-directory>
```

Retain the local copy until the destination repository and artifact contents have been checked.

## Train a policy

**Scope:** GPU-required for the documented path. **Requires:** a compatible dataset and NVIDIA Container Toolkit.

```bash
docker compose run --rm gpu \
  lerobot-ros-train \
  --dataset.repo_id=local/<dataset-name> \
  --dataset.root=/data/<dataset> \
  --policy.type=diffusion \
  --output_dir=/data/<experiment>/deploy/<run-name> \
  --steps=40000
```

Record the dataset identity, configuration, released image tag, random seed, and output checkpoint. Add `--push` only after the backup mapping has been reviewed.

## Deploy a checkpoint

**Scope:** robot-capable. **Requires:** a compatible checkpoint and controlled acceptance.

Deployment can command hardware. Complete all no-motion checks first:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot \
  lerobot-ros-deploy \
  --policy /data/<experiment>/deploy/<run-name>/checkpoints/last/pretrained_model \
  --config /config/gello.yaml \
  --output-dir /data/<experiment>/deployments
```

Before enabling commands, confirm observation names and shapes, action dimensions, normalization metadata, control frequency, joint limits, workspace clearance, and emergency-stop access.

## Collect DAgger data

**Scope:** robot-capable. **Requires:** a deployable checkpoint, intervention interface, and controlled acceptance.

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot \
  lerobot-ros-dagger \
  --name <dagger-dataset-name> \
  --task "<task description>" \
  --policy /data/<experiment>/deploy/<run-name>/checkpoints/last/pretrained_model \
  --config /config/gello.yaml
```

Treat every DAgger iteration as a new dataset and checkpoint lineage. Preserve the policy version, intervention source, episode metadata, and validation outcome.

## Train from DAgger data

**Scope:** GPU-required for the documented path. **Requires:** a compatible DAgger dataset and NVIDIA Container Toolkit.

```bash
docker compose run --rm gpu \
  lerobot-ros-train-dagger \
  --dataset.repo_id=local/<dagger-dataset-name> \
  --dataset.root=/data/<dagger-dataset> \
  --policy.type=diffusion \
  --output_dir=/data/<experiment>/deploy/<dagger-run-name> \
  --steps=40000
```

The exact policy configuration must match the dataset schema and the intended deployment interface.
