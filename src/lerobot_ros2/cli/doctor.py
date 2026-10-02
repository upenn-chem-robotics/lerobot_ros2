"""Read-only preflight checks for the supported Docker runtime."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import yaml

from lerobot_ros2.config_paths import resolve_config_path


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _import_check(module: str) -> Check:
    try:
        importlib.import_module(module)
        return Check(f"import {module}", True, "available")
    except Exception as exc:  # diagnostics must report the actual loader failure
        return Check(f"import {module}", False, f"{type(exc).__name__}: {exc}")


def _config_check(path: Path) -> tuple[Check, dict[str, Any]]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("top-level YAML value must be a mapping")
        return Check("configuration", True, str(path)), loaded
    except Exception as exc:
        return Check("configuration", False, f"{type(exc).__name__}: {exc}"), {}


def _device_checks(config: dict[str, Any]) -> list[Check]:
    devices = sorted({value for value in _walk_strings(config) if value.startswith("/dev/")})
    if not devices:
        return [Check("hardware devices", True, "no /dev paths declared", required=False)]
    return [Check(f"device {device}", Path(device).exists(), "present" if Path(device).exists() else "missing") for device in devices]


def _ros_graph_check() -> Check:
    if not shutil.which("ros2"):
        return Check("ROS graph", False, "ros2 executable is not on PATH")
    try:
        result = subprocess.run(
            ["ros2", "node", "list"], capture_output=True, text=True, timeout=10, check=False
        )
    except Exception as exc:
        return Check("ROS graph", False, f"{type(exc).__name__}: {exc}")
    detail = result.stdout.strip() or result.stderr.strip() or "reachable; no nodes discovered"
    return Check("ROS graph", result.returncode == 0, detail)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Config path; defaults to $GELLO_CONFIG")
    parser.add_argument("--skip-hardware", action="store_true", help="Do not check declared /dev paths")
    parser.add_argument("--skip-ros-graph", action="store_true", help="Import ROS but do not query discovery")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks: list[Check] = []

    try:
        config_path = resolve_config_path(args.config)
        config_check, config = _config_check(config_path)
    except SystemExit as exc:
        config_check, config = Check("configuration", False, str(exc)), {}
    checks.append(config_check)

    for module in ("rclpy", "sensor_msgs", "std_srvs", "rcl_interfaces", "lerobot", "lerobot_ros2"):
        checks.append(_import_check(module))

    for package in ("lerobot-ros2", "lerobot-policy-strided-diffusion", "lerobot-policy-action-history-diffusion"):
        try:
            checks.append(Check(f"distribution {package}", True, version(package)))
        except PackageNotFoundError:
            checks.append(Check(f"distribution {package}", False, "not installed"))

    data_root = Path(os.environ.get("LEROBOT_DATA", "/data"))
    checks.append(Check("data directory", data_root.is_dir() and os.access(data_root, os.W_OK), str(data_root)))

    if not args.skip_hardware:
        checks.extend(_device_checks(config))
    if not args.skip_ros_graph:
        checks.append(_ros_graph_check())

    width = max(len(check.name) for check in checks)
    for check in checks:
        label = "PASS" if check.ok else ("WARN" if not check.required else "FAIL")
        print(f"{label:4}  {check.name:<{width}}  {check.detail}")

    failures = [check for check in checks if check.required and not check.ok]
    if failures:
        print(f"\n{len(failures)} required check(s) failed.", file=sys.stderr)
        return 1
    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
