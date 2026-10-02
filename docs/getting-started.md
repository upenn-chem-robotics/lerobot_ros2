# Getting started

Use Linux x86-64 with Docker Engine and Compose v2. GPU workflows also require the NVIDIA Container Toolkit. Do not install ROS or Conda on the host.

```bash
mkdir -p config.local data
cp examples/gello.yaml config.local/gello.yaml
export UID="$(id -u)" GID="$(id -g)"
docker compose build tools
docker compose run --rm tools
```

Replace every `REPLACE_*` value, copy `examples/compose.override.hardware.yaml` to `compose.hardware.yaml`, and map stable device paths. Set numeric group IDs with `stat -c '%g' DEVICE`, then run the full preflight. Never use `privileged: true` to hide device permission errors.
