# Recording

Run the full doctor, verify camera orientation and controls, inspect arm state, test discard/reset behavior, and choose a fresh output under `/data`.

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm robot lerobot-ros-record --config /config/gello.yaml --help
```

Afterward, verify episodes, frame counts, timestamps, state/action dimensions, videos, and metadata. Preserve raw data and write transformations to a new destination unless in-place behavior is explicitly documented.
