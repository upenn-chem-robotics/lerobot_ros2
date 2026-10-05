# Development

> **Audience:** Contributors and advanced users who need to modify source or documentation. Normal users should use the published-image path in [Installation](installation.md).

## Repository structure

- `src/lerobot_ros2/`: application and CLI implementation
- `packages/`: separately distributed policy plugins
- `tests/`: unit and hardware-free integration tests
- `scripts/`: release, validation, camera-support, backup, and container-entrypoint scripts
- `examples/`: minimal comment-free onboarding configuration
- `config/*.example.yaml`: annotated public configuration templates
- `docs/`: public user and contributor documentation

The runtime image installs wheels. Editable or source-mounted execution is reserved for the `dev` service so release behavior is tested against installed distributions.

## Development environment

```bash
export UID="$(id -u)" GID="$(id -g)"
docker compose build dev
docker compose run --rm dev pytest
docker compose run --rm dev ruff check .
```

Run dependency validation in the runtime image:

```bash
docker compose run --rm tools python -m pip check
```

## CLI commands

Console entry points are declared in `pyproject.toml`. A CLI change is complete only when:

- the module imports in the runtime image
- `--help` exits successfully
- required dependencies are present in the correct environment
- tests cover parsing and the behavior changed
- the relevant workflow section is updated

## Policy plugins

Policy plugins are separately built distributions under `packages/`. Keep plugin versions aligned with the main release and verify distribution discovery in the installed image. Tests should target the problem addressed by each policy rather than relying on arbitrary component ablations.

For `strided_diffusion`, preserve the distinction between spaced observation history and contiguous predicted actions. For `action_history_diffusion`, test action-history dimensions, dropout behavior, and the `n_action_history=0` compatibility path.

## Documentation

The public documentation intentionally uses a small number of task-oriented pages. Add content to an existing page unless a genuinely separate user journey requires a new top-level page.

Build documentation strictly:

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

A documentation change should include working internal links, commands that match installed entry points, explicit prerequisites, expected outputs, and failure or safety boundaries.

## Contribution boundaries

Do not commit local configuration, data, outputs, logs, checkpoints, tokens, robot addresses, private inventories, or participant information. Use sanitized examples and keep site-specific material under ignored paths.
