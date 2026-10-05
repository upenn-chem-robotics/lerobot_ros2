# Configuration

> **Page scope:** This is the reference for configuration files, paths, variables, mounts, devices, and secrets. Tutorials should link here instead of duplicating configuration details.

## Configuration model

All committed configuration templates live in `config/` and end in `*.example.yaml`. Copy a template to its documented ignored runtime location before editing it; do not run hardware workflows directly from an example file.

Committed files are examples and defaults. Local configuration belongs under `config.local/` and is mounted read-only at `/config`. Datasets and outputs belong under the host `data/` directory and are mounted at `/data`.

```bash
mkdir -p config.local data
cp examples/gello.yaml config.local/gello.yaml
```

`examples/gello.yaml` is the small comment-free copy used for onboarding and software-only checks. `config/gello.example.yaml` is the annotated reference template; `config/hf-backup.example.yaml` and `config/compose.hardware.example.yaml` are the other committed examples.

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
