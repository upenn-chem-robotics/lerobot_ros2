"""Exercise installed lerobot-ros2 console scripts beyond import-only validation."""

from __future__ import annotations

import os
import subprocess
from importlib.metadata import entry_points


def test_every_console_script_constructs_help() -> None:
    scripts = sorted(
        entry_point.name
        for entry_point in entry_points(group="console_scripts")
        if entry_point.module.startswith("lerobot_ros2.cli")
    )
    assert scripts, "no lerobot_ros2 console scripts were installed"

    failures: list[str] = []
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    for script in scripts:
        try:
            result = subprocess.run(
                [script, "--help"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{script}: timed out")
            continue
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip()[-2000:]
            failures.append(f"{script}: exit {result.returncode}\n{output}")

    assert not failures, "console script help failures:\n\n" + "\n\n".join(failures)
