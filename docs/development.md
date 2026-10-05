# Development

> **Audience:** Contributors modifying source code or documentation. Published releases are installed as described in [Installation](installation.md).

## Repository structure

- `src/lerobot_ros2/`: application and CLI implementation
- `packages/`: separately distributed policy plugins
- `tests/`: unit and hardware-free integration tests
- `scripts/`: release, validation, camera-support, backup, and container-entrypoint scripts
- `config/*.example.yaml`: annotated public configuration templates
- `docs/`: public user and contributor documentation

The runtime image installs wheels. Editable or source-mounted execution is reserved for the `dev` service so release behavior is tested against installed distributions. Before changing code, identify the owning command or package, reproduce the current behavior in `dev`, and determine which installed runtime image must ultimately contain the change. Do not validate only through an editable source mount when the released path uses installed wheels.

## First development checkout

Use a full clone rather than the sparse release checkout:

```bash
git clone https://github.com/upenn-chem-robotics/lerobot_ros2.git
cd lerobot_ros2
git switch -c <branch-name>
```

Before editing, build the development service and run the baseline checks. After editing, rerun the verification flow that owns the changed files.

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

For `strided_diffusion`, preserve the distinction between spaced observation history and contiguous predicted actions. Test temporal indexing, beginning-of-episode padding, input shapes, serialization, and rejection of inconsistent FPS configuration. For `action_history_diffusion`, test action-history dimensions, dropout behavior, serialization, and the `n_action_history=0` compatibility path. Behavioral comparisons should be tied to the partial-observability problem the variant is intended to address rather than treated as arbitrary component ablations.

## Documentation

Add material to the page that owns the relevant workflow. Create a top-level page only for a separate workflow.

Build documentation strictly:

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

A documentation change should include working internal links, commands that match installed entry points, explicit prerequisites, expected outputs, and failure or safety boundaries.

## Contribution boundaries

Do not commit local configuration, data, outputs, logs, checkpoints, tokens, robot addresses, private inventories, or participant information. Use sanitized examples and keep site-specific material under ignored paths.

## Quick verification flows

Before opening a pull request, run the verification flow that covers the changed components. Hardware-facing changes also require the controlled checks in [Hardware and safety](hardware-and-safety.md).

### Documentation-only change

```bash
python -m mkdocs build --strict
```

This catches broken navigation, invalid Markdown configuration, and unresolved documentation links.

### Python or CLI change

```bash
docker compose run --rm dev ruff check .
docker compose run --rm dev pytest
```

Use this for application logic, command-line interfaces, configuration parsing, and unit-tested dataset operations.

### Container or dependency change

```bash
bash scripts/test_no_hardware.sh
```

This builds the runtime and development images, runs `pip check`, runs the ROS doctor report, and executes the repository's no-hardware integration selection. To reuse images that were already built with the script's expected tags, set `SKIP_BUILD=1`.

### Public bundle or image change

```bash
bash scripts/verify_public_release.sh
```

Use this when changing the Dockerfile, Compose files, packaging metadata, release bundle contents, entrypoint, dependency locks, or published-image behavior. Review the generated logs and resolve every failure before publishing. Retain the logs with the exact source and image identifiers to which the verification applies.

### Hardware-facing change

First run the Python/CLI and container flows above. Then follow [Hardware and safety](hardware-and-safety.md) in a controlled workspace. Verify configuration and observation interfaces without motion before enabling commands, and repeat the relevant camera, operator-input, limit, emergency-stop, and shutdown checks after changes to controllers, firmware, calibration, networking, checkpoints, or safety configuration.
