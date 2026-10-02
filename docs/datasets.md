# Dataset operations

Use `tools` for CPU transformations. Preserve raw inputs, choose explicit outputs, inspect one episode before a batch, and load results before deleting intermediates.

```bash
docker compose run --rm tools lerobot-ros-export --help
docker compose run --rm tools lerobot-ros-canonicalize --help
docker compose run --rm tools lerobot-ros-reorient --help
docker compose run --rm tools lerobot-ros-downsample --help
```

A zero exit code is not sufficient. Verify indices, timestamps, dimensions, camera keys, video decoding, and metadata consistency.
