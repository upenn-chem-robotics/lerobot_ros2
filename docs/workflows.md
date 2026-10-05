# Workflows

Use the released image and matching checkout throughout a workflow. If a documented command is missing or its interface differs, first verify the release tag rather than patching the container.

Each workflow below states its scope, prerequisites, and command. Definitions of *episode*, *policy*, and *checkpoint* are provided in [Concepts](concepts.md).

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
| Run a policy on the robot | [Deploy a validated checkpoint](#deploy-a-validated-checkpoint) | Robot-capable | Compatible checkpoint and completed acceptance checks |
| Collect policy corrections | [Collect DAgger data](#collect-dagger-data) | Robot-capable | Deployable checkpoint and intervention interface |
| Retrain with corrections | [Train from DAgger data](#train-from-dagger-data) | GPU-required | Compatible DAgger dataset and NVIDIA runtime |

## Run the software-only preflight

**Scope:** software-only. **Requires:** the published `tools` image and local mounts. **Produces:** a runtime and configuration report; it does not validate a dataset, checkpoint, GPU, camera, ROS graph, or robot.

```bash
docker compose pull tools
docker compose run --rm tools \
  lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

Run this check before dataset, learning, or hardware diagnostics so packaging and mount failures are reported separately.

## Dataset workflows

Use these operations only after the source has been opened and inspected. Always write to a new destination and inspect that destination before training.

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

This writes grid videos and timeline images without replacing the source dataset. Use them for rapid visual review and comparison, not as a substitute for checking metadata and numerical signals. Confirm that the exports cover the intended episodes and camera streams.

## Start and verify the robot stack

**Scope:** robot-capable. **Requires:** the Humble `ur_robotiq` environment, site robot configuration, and the safety checklist.

The released robot runtime must be running before camera-enabled recording, deployment, or DAgger. Verify the state topics, command topics, GELLO services, controller state, and shared `ROS_DOMAIN_ID` before using the deployed hardware.

Follow [Bimanual UR3 and Robotiq integration](robot-integration.md) for the supported launch commands and interface checks. Do not treat `lerobot-ros-doctor --skip-ros-graph` as a robot integration check.

!!! warning "Hardware boundary"
    The following workflows can access cameras or command a robot. Complete [Hardware and safety](hardware-and-safety.md), resolve every `REPLACE_*` value, review `compose.hardware.yaml`, verify stable device paths, complete no-motion checks, clear the workspace, and confirm emergency-stop access.

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

**Recording readiness gate:** continue only when camera identities and orientation are confirmed, pedal or operator input is confirmed, left/right state association is unambiguous, both GELLO nodes have been verified in mode `0`, the destination is understood, the workspace is clear, emergency-stop access is confirmed, and one operator controls the transition.

After that gate:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot \
  lerobot-ros-record \
  --name <dataset-name> \
  --task "<task description>" \
  --config /config/gello.yaml
```

Use `--left` or `--right` only for a single-arm recording. Keep partial or interrupted recordings separate until their metadata and episode finalization have been checked.

## Canonicalize a dataset

**Use when:** the source fields, ordering, shapes, or conventions are understood and must be converted to the project schema. **Do not use when:** the meaning of a source field is unknown.

**Scope:** software-only. **Requires:** an existing dataset; writes a new destination.

```bash
docker compose run --rm tools \
  lerobot-ros-canonicalize \
  --src /data/<source-dataset> \
  --dst /data/<canonical-dataset> \
  --task-name "<task description>"
```

Canonicalization may change field names, ordering, shapes, or conventions to match the project schema. Do not point `--dst` at the source dataset. Before using the result, compare episode counts, observation fields, action dimensions, joint ordering, gripper convention, timestamps, and task metadata with the source.

## Reorient camera observations

**Use when:** a known camera is rotated and the required deployment orientation is known. **Do not use when:** camera identity or the affected episodes are uncertain.

**Scope:** software-only. **Requires:** an existing dataset; writes a new destination.

```bash
docker compose run --rm tools \
  lerobot-ros-reorient \
  --src /data/<source-dataset> \
  --dst /data/<reoriented-dataset> \
  --camera <camera-name>:<episode-spec>
```

Use a camera short name reported by `lerobot-ros-reorient --src /data/<source-dataset> --report`. The episode specification can be a single episode, a comma-separated list, a range, or `all`. Run `lerobot-ros-reorient --help` for the orientation operations supported by the installed release; preview the selected camera and episodes before applying one. The destination must differ from the source, and the transformed images must be checked against the orientation expected at deployment.

## Downsample a dataset

**Use when:** a specific lower temporal rate is required and short events have been reviewed. **Do not use merely to reduce size without checking contacts, brief actions, and policy FPS requirements.

**Scope:** software-only. **Requires:** an existing dataset; writes a new destination.

```bash
docker compose run --rm tools \
  lerobot-ros-downsample \
  --src /data/<source-dataset> \
  --dst /data/<downsampled-dataset>
```

The command derives its resampling behavior from the installed command and dataset metadata; inspect `lerobot-ros-downsample --help` for the options and defaults of the released version rather than assuming a target rate from this example. Afterward, verify the destination FPS, timestamps, episode lengths, video alignment, and retention of short contacts or actions. Keep the source unchanged until the destination has been inspected.

## Back up a dataset or checkpoint directory

**Scope:** network access. **Requires:** an artifact, destination configuration, and external authentication.

Configure `config.local/hf-backup.yaml`, authenticate outside committed files, then run:

```bash
docker compose run --rm tools \
  lerobot-ros-backup /data/<dataset-or-checkpoint-directory>
```

Retain the local copy until the destination repository and artifact contents have been checked.

## Train a policy

**Scope:** GPU-required for the documented path. **Requires:** a dataset that has passed inspection and NVIDIA Container Toolkit.

Before the full run:

1. Open the dataset with [Visualize a dataset](#visualize-a-dataset).
2. Record its observation fields, action dimensions, and FPS.
3. Confirm that the selected policy accepts that schema.
4. Start with stock `diffusion` unless spaced observation history or previous-command history addresses a specific task requirement described in [Concepts](concepts.md#project-specific-policy-variants).
5. Inspect the installed command before launching a long run:

```bash
docker compose run --rm gpu lerobot-ros-train --help
```

Use the released command's supported options to perform the shortest practical loading test before the full run. The loading test should reach dataset loading and model initialization and write only to a disposable output directory.

```bash
docker compose run --rm gpu \
  lerobot-ros-train \
  --dataset.repo_id=local/<dataset-name> \
  --dataset.root=/data/<dataset> \
  --policy.type=diffusion \
  --output_dir=/data/<experiment>/deploy/<run-name> \
  --steps=40000
```

**Training output gate:** the dataset loads with the expected fields, the model initializes, the chosen output directory receives the run artifacts, and the saved policy can be reloaded with its recorded schema.

The shown `--steps=40000` is an example run length, not evidence that the policy has converged or is deployable. Record the dataset identity, configuration, released image tag, random seed, and output checkpoint. Confirm that the saved policy reloads with the recorded observation and action schemas. Add `--push` only after the backup mapping has been reviewed.

## Validate a checkpoint without motion

**Scope:** no-motion compatibility check. **Requires:** a saved checkpoint, its training record, and the current configuration.

Before starting any robot-capable deployment command, compare:

- checkpoint observation names and shapes with the configured cameras and state fields;
- checkpoint action dimensions and ordering with the configured joints and grippers;
- normalization metadata and action convention;
- training FPS and history settings with the intended runtime;
- policy type and plugins with the released image.

Use the released command interfaces and the site's no-motion procedure to load the checkpoint without enabling command publication. A successful file load is only the start of this check.

**Checkpoint compatibility verified:** every item above has a recorded match and the live state association has already passed the ROS no-motion checks.

!!! danger "Do not proceed"
    Do not deploy when metadata is missing, a dimension is merely assumed, camera roles differ, joint ordering is uncertain, or the checkpoint requires a policy plugin absent from the released image.

## Deploy a validated checkpoint

**Scope:** robot-capable. **Requires:** checkpoint compatibility verified, ROS interface verified, and physical acceptance completed.

Deployment can command hardware. Immediately before running the command, confirm the recorded checkpoint-compatibility result, current controller state, joint limits, workspace clearance, emergency-stop access, and responsible operator. Then run:

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

Treat every DAgger iteration as a new dataset and checkpoint lineage. Preserve the policy version, intervention source, episode metadata, and validation outcome. Before training, inspect how policy commands and human interventions are represented in the dataset's action-source metadata; do not assume that an intervention field has the same meaning across dataset versions.

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

The exact policy configuration must match the dataset schema and the intended deployment interface. Inspect the installed command's sampling and weighting options with `lerobot-ros-train-dagger --help`, and record the selected treatment of policy-generated and intervention data with the resulting checkpoint.
