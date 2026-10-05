# UR Robotiq bimanual profile

This profile contains the stable `lerobot_ros2` settings for the lab's released ROS 2 Humble `ur_robotiq` runtime. The robot runtime is an external dependency; normal users run its image and do not modify its repository or installed files.

Copy the profile into ignored local state:

```bash
mkdir -p config.local data
cp profiles/ur_robotiq_bimanual/gello.yaml config.local/gello.yaml
cp profiles/ur_robotiq_bimanual/compose.hardware.yaml compose.hardware.yaml
```

Replace only machine-local placeholders:

```bash
grep -R "REPLACE_" config.local/gello.yaml compose.hardware.yaml
```

No output is expected before camera or robot workflows. Keep robot addresses, calibration files, device identifiers, GELLO serial identifiers, and site network details outside the public repository.

The expected image and ROS interfaces are recorded in `compatibility.yaml`. Verify them against the robot runtime release used by the lab before operation.
