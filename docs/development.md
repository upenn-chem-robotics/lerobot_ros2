# Development

```bash
export UID="$(id -u)" GID="$(id -g)"
docker compose build dev
docker compose run --rm dev ruff check .
docker compose run --rm dev pytest
```

Runtime installs wheels; editable source is confined to `dev`. Separate unit, ROS integration, and hardware acceptance tests. New CLIs must be registered, import without hardware, expose `--help`, and have a smoke test.
