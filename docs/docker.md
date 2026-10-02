# Docker runtime

The `runtime` target is non-root and wheel-installed. `dev` adds pytest and Ruff and bind-mounts source. Standard paths are `/data`, `/config`, `/cache/huggingface`, `/cache/torch`, and development-only `/workspace`.

Compose services are `tools` for CPU work, `gpu` for CUDA work, `robot` for host ROS networking and explicit devices, and `dev` for tests. The entrypoint activates Conda and uses `exec`; `tini` handles subprocess reaping.

The robot service uses host networking on Linux. Match `ROS_DOMAIN_ID` to the robot graph. If host networking is forbidden, provide and validate an explicit DDS configuration rather than assuming bridged multicast works.

```bash
docker compose run --rm gpu python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```
