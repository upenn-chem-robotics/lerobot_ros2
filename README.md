# lerobot-ros2

Recording, deploy, export and visualization utilities for UR3e bimanual
teleoperation on top of [huggingface/lerobot](https://github.com/huggingface/lerobot).

## Layout

```
lerobot-ros2/
├── config/gello.yaml           Example teleop / camera config
├── src/lerobot_ros2/           Main package (import `lerobot_ros2`)
│   ├── helper.py               Cameras, arms, wrap joints, ROS helpers, ...
│   ├── visualizer.py           OpenCV live preview
│   ├── strided_history.py      Inference-side runner for strided-diffusion
│   ├── data_loader.py          Parquet + video dataset reader
│   ├── video_exporter.py       Timeline / grid video export
│   ├── config_paths.py         `--config` / $GELLO_CONFIG resolution
│   └── cli/                    Console entry points (see below)
├── packages/
│   └── lerobot_policy_strided_diffusion/   Sibling distribution (lerobot plugin)
└── tests/
```

## Install

```bash
# One-shot dev setup (after `pip install -e huggingface/lerobot`)
pip install -e .
pip install -e packages/lerobot_policy_strided_diffusion
```

The strided-diffusion policy plugin has to be a separate distribution
because `lerobot.utils.import_utils.register_third_party_plugins` discovers
plugins by enumerating installed distributions whose name starts with
`lerobot_policy_`.

## Config

All CLIs require a teleop/camera config. Point them at one via either:

```bash
export GELLO_CONFIG=/path/to/gello.yaml
# or per-invocation:
lerobot-ros-record --config /path/to/gello.yaml ...
```

There is no filesystem default — the installed package does not know where
your config lives.

## Console entry points

| Command                | Description                                 |
|------------------------|---------------------------------------------|
| `lerobot-ros-record`   | Record teleoperation datasets               |
| `lerobot-ros-deploy`   | Deploy a trained ACT / Diffusion policy     |
| `lerobot-ros-export`   | Export dataset grid videos / timelines      |
| `lerobot-ros-probe-cameras` | Snapshot v4l2 controls per camera     |
| `lerobot-ros-downsample` | Rewrite + downsample a LeRobot v3 dataset |
| `lerobot-ros-app`      | Gradio dataset visualizer                   |
