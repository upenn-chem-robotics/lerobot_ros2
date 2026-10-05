# Visualize your first dataset

> **Scope:** This tutorial starts from an existing LeRobot dataset. It is software-only and does not access cameras or command a robot. If you do not have a dataset, follow the recording path on [Start here](index.md#i-am-new-and-do-not-have-a-dataset).

## Prerequisites

You need the installed release image and an existing LeRobot dataset below the host `data/` directory. No Hugging Face authentication is required unless the dataset must first be downloaded from a private or gated repository. For example, host `data/pick_place` appears in the container as `/data/pick_place`.

## 1. Run the software-only preflight

```bash
docker compose pull tools
docker compose run --rm tools \
  lerobot-ros-doctor --skip-hardware --skip-ros-graph
```

This checks the packaged application and local mounts. It does not validate the dataset itself.

## 2. Open the dataset

```bash
docker compose run --rm --service-ports tools \
  lerobot-ros-app --dataset_dir /data/pick_place --port 7860
```

Replace `/data/pick_place` with your path below `/data`, then open `http://localhost:7860` on the same workstation.

## 3. Inspect before processing or training

Check that episodes and frames load; camera identity and orientation are correct; timestamps and task boundaries are plausible; observations and actions have the expected dimensions; and partial episodes are identified. Keep the source unchanged and write repairs or transformations to a new destination.

## Stop before hardware

Before camera probing, recording, deployment, or DAgger, read [Hardware and safety](hardware-and-safety.md), replace every `REPLACE_*` value, create `compose.hardware.yaml`, and complete the no-motion checks.
