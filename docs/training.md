# Training

Use the `gpu` service so training matches deployment. Store datasets and checkpoints under `/data`; caches use named volumes.

```bash
docker compose run --rm gpu lerobot-ros-train --help
docker compose run --rm gpu lerobot-ros-train-dagger --help
```

Record the image tag, Git commit, `dependencies.env`, configuration, dataset revision, seed, and command. Policy plugins stay separate because LeRobot discovers `lerobot_policy_*` distributions.
