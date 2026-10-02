# Troubleshooting

Start with `lerobot-ros-doctor` and fix required checks top to bottom.

- Device permission: map the exact path and numeric supplemental group; do not use privileged mode.
- No ROS nodes: verify `ROS_DOMAIN_ID`, host networking, interfaces, firewall, and DDS settings inside `robot`.
- CUDA unavailable: verify host `nvidia-smi`, then NVIDIA Container Toolkit and the in-container Torch check.
- Camera drift: inspect `v4l2-ctl --all`, disable firmware automation, and reapply controls after reopen.
- Wrong ownership: rebuild with host UID/GID and repair existing host files once.
- Plugin discovery: run `python -m pip check` and inspect installed `lerobot-policy-*` distributions.
