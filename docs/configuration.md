# Configuration

> **Page scope:** This is the reference for configuration files, paths, variables, mounts, devices, and secrets. Tutorials should link here instead of duplicating configuration details.

## Minimum required edits

Keep `config/gello.example.yaml` as a reference. Copy it to `config.local/gello.yaml` and change only values required by the workflow.

| Workflow | Minimum local inputs |
|---|---|
| Dataset visualization or transformation | Dataset below host `data/`; no robot configuration. |
| GPU training | Dataset path and repository ID; working NVIDIA runtime. |
| Camera probing | Camera entries plus stable mappings in `compose.hardware.yaml`. |
| Recording | Camera, operator-input, ROS, arm, and gripper values used by the site. |
| Deployment or DAgger | Recording inputs, compatible checkpoint, and completed no-motion checks. |

Validate the merged Compose file and packaged application before hardware access:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml config
docker compose run --rm tools \
  lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

Remove every `REPLACE_*` value before camera or robot workflows. This preflight is not physical acceptance.

## Configuration model

Generic copy-and-edit templates live in `config/` and end in `*.example.yaml`. Supported configurations under `profiles/` use their runtime filenames and are copied into the documented ignored runtime locations. Do not run hardware workflows directly from a generic example file.

Committed files are examples and defaults. Local configuration belongs under `config.local/` and is mounted read-only at `/config`. Datasets and outputs belong under the host `data/` directory and are mounted at `/data`.

```bash
mkdir -p config.local data
cp profiles/ur_robotiq_bimanual/gello.yaml config.local/gello.yaml
```

`profiles/ur_robotiq_bimanual/gello.yaml` is the supported lab profile. `config/gello.example.yaml` is the generic annotated template for other robots and sites. The profile commits stable ROS and dataset decisions; ignored local files hold machine-specific device identifiers and site details.

Replace every `REPLACE_*` value before running a hardware workflow. Never commit credentials, private datasets, participant data, robot network details, hardware inventories, or local device identifiers.

## Important paths

- `config.local/`: ignored site-specific configuration
- `/config`: read-only configuration inside containers
- `data/`: ignored host dataset location
- `/data`: dataset and output location inside containers
- `release-logs/`: ignored verification output
- named Hugging Face and Torch caches: reusable downloaded artifacts

## Environment variables

The Compose services use these primary variables:

- `GELLO_CONFIG`: path to the active configuration inside containers, normally `/config/gello.yaml`
- `HF_HOME`: Hugging Face cache location
- `TORCH_HOME`: Torch cache location
- `UID` and `GID`: host identity used for non-root container files
- ROS variables such as `ROS_DOMAIN_ID` when isolating a graph

Set host identity before running Compose:

```bash
export UID="$(id -u)" GID="$(id -g)"
```

## Hardware override

Keep hardware access outside the base Compose file. Create an ignored `compose.hardware.yaml` and extend only the `robot` service. Use stable device paths:

```yaml
services:
  robot:
    devices:
      - /dev/v4l/by-id/REPLACE_CAMERA:/dev/camera0
      - /dev/input/by-id/REPLACE_PEDAL:/dev/pedal
```

Run the merged configuration as follows:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml config
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot   lerobot-ros-doctor
```

Never use `privileged: true` as a substitute for identifying required devices.

## Configuration review checklist

Before recording or actuation:

1. Confirm that all `REPLACE_*` values are gone.
2. Confirm that device mappings resolve to the intended physical devices.
3. Confirm the ROS domain, topic names, service names, and namespaces.
4. Confirm the data directory and available capacity.
5. Confirm checkpoint and policy configuration without motion.
6. Keep secrets outside YAML committed to the repository.

## Supported profile and generic template

Use `profiles/ur_robotiq_bimanual/` for the lab robot. Its `compatibility.yaml` records the expected Humble robot image and ROS contract. Copy profile files into ignored local state rather than editing the committed profile during operation.

Use `config/gello.example.yaml` and `config/compose.hardware.example.yaml` when adapting a different robot or site. Their `REPLACE_*` values are intentional and must be resolved as part of defining that integration.

The profile's `rotation_arm`, `subtask_arms`, `wrap_joints`, `unwrap`, and `recording.hz` are dataset semantics owned by `lerobot_ros2`. Robot addresses, calibration, base poses, controllers, ToolComm, and GELLO serial identifiers remain owned by the external robot deployment.

See [UR Robotiq bimanual profile](robot-integration.md) for the lab workflow.
