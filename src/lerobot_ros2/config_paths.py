"""Resolve the path to the ``gello.yaml`` teleop/camera config.

Callers must provide the path explicitly, either through a ``--config`` CLI
argument or by setting ``$GELLO_CONFIG``. There is deliberately no filesystem
fallback: the installed package does not know where the user's config tree
lives, and we would rather fail loudly than silently load the wrong file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


CONFIG_ENV_VAR = "GELLO_CONFIG"


def resolve_config_path(cli_value: Optional[str] = None) -> Path:
    """Return the absolute path of the config file to load.

    Precedence: explicit ``cli_value`` > ``$GELLO_CONFIG``. If neither is set,
    or the resolved path does not exist, ``SystemExit`` is raised with a
    message telling the user how to fix it.
    """
    value = cli_value or os.environ.get(CONFIG_ENV_VAR)
    if not value:
        raise SystemExit(
            "No config specified. Pass --config PATH or set "
            f"${CONFIG_ENV_VAR}=/path/to/gello.yaml."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Config file not found: {path}")
    return path
