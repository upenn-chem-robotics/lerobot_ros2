# Deployment

Deployment controls physical equipment. Validate the checkpoint, observation schema, camera geometry, command topics, workspace, limits, emergency stop, and clean SIGINT shutdown without publishing motion first. Then perform a reduced-speed rollout with an operator at the emergency stop.

```bash
docker compose -f compose.yaml -f compose.hardware.yaml run --rm gpu lerobot-ros-deploy --config /config/gello.yaml --help
```

This stack is not a safety-rated or real-time control layer.
