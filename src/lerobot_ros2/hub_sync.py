"""Mirror local dataset / checkpoint folders to private Hugging Face repos.

This implements the "raw mirror" backup strategy described in
``config/hf_backup.yaml``: a folder under ``data/`` is uploaded verbatim (its
internal ``meta/``, ``data/``, ``videos/``, ``deploy/`` tree preserved) into a
private HF *dataset* repo, so it can be deleted locally and restored later with
``hf download <repo> --repo-type dataset --local-dir <path>``.

The module is imported by the record / dagger / train CLIs to push new data
automatically after a run. ``huggingface_hub`` is imported lazily so importing
this module never hard-fails an environment that doesn't have it installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# config/hf_backup.yaml lives at the repo root; this file is at
# <repo>/src/lerobot_ros2/hub_sync.py, so parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "hf_backup.yaml"

# Env var to globally disable auto-push (e.g. ``LEROBOT_HF_PUSH=0``).
PUSH_ENV_VAR = "LEROBOT_HF_PUSH"


@dataclass(frozen=True)
class BackupConfig:
    hf_user: str
    repo_prefix: str
    repo_type: str
    private: bool
    data_root: Path  # absolute
    nested_roots: tuple[str, ...]
    targets: tuple[tuple[Path, str], ...]  # (absolute local root, repo name)

    def repo_id(self, repo_name: str) -> str:
        return f"{self.hf_user}/{repo_name}"


@dataclass(frozen=True)
class RepoTarget:
    repo_id: str
    repo_type: str
    private: bool
    path_in_repo: str  # "" means the repo root


def load_config(config_path: str | os.PathLike[str] | None = None) -> BackupConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"HF backup config not found at {path}. Expected config/hf_backup.yaml."
        )
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

    hf_user = str(raw.get("hf_user") or "").strip()
    if not hf_user or hf_user.startswith("<"):
        raise ValueError(
            f"'hf_user' is not set in {path}. Put your Hugging Face username there."
        )

    repo_root = path.resolve().parent.parent  # <repo>/config/.. -> <repo>
    data_root = (repo_root / str(raw.get("data_root", "data"))).resolve()

    targets: list[tuple[Path, str]] = []
    for entry in raw.get("backup_targets", []) or []:
        local = (repo_root / str(entry["local"])).resolve()
        targets.append((local, str(entry["repo"])))

    return BackupConfig(
        hf_user=hf_user,
        repo_prefix=str(raw.get("repo_prefix", "lerobot-data")),
        repo_type=str(raw.get("repo_type", "dataset")),
        private=bool(raw.get("private", True)),
        data_root=data_root,
        nested_roots=tuple(str(r) for r in (raw.get("nested_roots") or [])),
        targets=tuple(targets),
    )


@lru_cache(maxsize=4)
def _cached_config(config_path: str | None) -> BackupConfig:
    return load_config(config_path)


def _sanitize(name: str) -> str:
    """HF repo names allow [A-Za-z0-9._-]; map anything else to '-'."""
    return "".join(c if (c.isalnum() or c in "._-") else "-" for c in name)


def resolve_target(
    local_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
) -> RepoTarget:
    """Map a local folder under ``data/`` to its HF repo + path-in-repo.

    Explicit ``backup_targets`` win (a path inside a target maps to that repo
    with the remainder as ``path_in_repo``). Otherwise the default rule based
    on ``nested_roots`` is applied.
    """
    cfg = _cached_config(str(config_path) if config_path else None)
    abs_path = Path(local_path).resolve()

    # 1. Explicit targets (longest match first so nested targets win).
    for root, repo_name in sorted(cfg.targets, key=lambda t: len(str(t[0])), reverse=True):
        if abs_path == root or root in abs_path.parents:
            rel = abs_path.relative_to(root)
            return RepoTarget(
                repo_id=cfg.repo_id(repo_name),
                repo_type=cfg.repo_type,
                private=cfg.private,
                path_in_repo="" if rel == Path(".") else rel.as_posix(),
            )

    # 2. Default rule, relative to data_root.
    try:
        rel = abs_path.relative_to(cfg.data_root)
    except ValueError as exc:
        raise ValueError(
            f"{abs_path} is not under the configured data_root {cfg.data_root}; "
            f"cannot determine a backup repo. Either move it under data/ or add an "
            f"entry to config/hf_backup.yaml, or run the backup manually."
        ) from exc

    parts = rel.parts
    if not parts:
        raise ValueError(f"Refusing to mirror the entire data_root {cfg.data_root}.")

    depth = 2 if (parts[0] in cfg.nested_roots and len(parts) >= 2) else 1
    repo_root_parts = parts[:depth]
    repo_name = _sanitize("-".join((cfg.repo_prefix, *repo_root_parts)))
    path_in_repo = Path(*parts[depth:]).as_posix() if len(parts) > depth else ""
    return RepoTarget(
        repo_id=cfg.repo_id(repo_name),
        repo_type=cfg.repo_type,
        private=cfg.private,
        path_in_repo=path_in_repo,
    )


def push_enabled(explicit: bool | None = None) -> bool:
    """Resolve whether to push: explicit flag > ``LEROBOT_HF_PUSH`` env > True."""
    if explicit is not None:
        return explicit
    val = os.environ.get(PUSH_ENV_VAR)
    if val is None:
        return True
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def pop_no_push_flag(argv: list[str]) -> bool:
    """Strip ``--no-push`` / ``--no_push`` from ``argv``; return True if present.

    For CLIs that use draccus/other parsers which would reject unknown flags
    (e.g. the training entry points).
    """
    aliases = {"--no-push", "--no_push"}
    found = False
    cleaned: list[str] = []
    for tok in argv:
        if tok in aliases:
            found = True
            continue
        cleaned.append(tok)
    argv[:] = cleaned
    return found


def _hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "huggingface_hub is not installed in this environment. "
            "Install it (pip install huggingface_hub) and authenticate with "
            "`hf auth login` or HF_TOKEN."
        ) from exc
    return HfApi()


def sync_to_hub(
    local_path: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] | None = None,
    large: bool = False,
    create: bool = True,
    dry_run: bool = False,
) -> RepoTarget:
    """Upload ``local_path`` to its mapped private HF repo (incremental).

    ``large=True`` uses ``upload_large_folder`` (resumable, multi-threaded) and
    is intended for the one-time bulk backup of whole experiment folders; it
    requires ``path_in_repo == ""`` (repo root == folder). Otherwise a plain
    ``upload_folder`` is used, which only re-uploads changed/new files and so is
    cheap for repeated post-run syncs.
    """
    abs_path = Path(local_path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"Nothing to upload: {abs_path} does not exist.")

    target = resolve_target(abs_path, config_path=config_path)
    logger.info(
        "HF mirror: %s -> %s%s%s",
        abs_path,
        target.repo_id,
        f" :/{target.path_in_repo}" if target.path_in_repo else "",
        " [dry-run]" if dry_run else "",
    )
    if dry_run:
        return target

    api = _hf_api()
    if create:
        api.create_repo(
            repo_id=target.repo_id,
            repo_type=target.repo_type,
            private=target.private,
            exist_ok=True,
        )

    if large and not target.path_in_repo:
        api.upload_large_folder(
            repo_id=target.repo_id,
            folder_path=str(abs_path),
            repo_type=target.repo_type,
            private=target.private,
        )
    else:
        api.upload_folder(
            repo_id=target.repo_id,
            repo_type=target.repo_type,
            folder_path=str(abs_path),
            path_in_repo=target.path_in_repo or None,
            commit_message=f"Mirror {abs_path.name}",
        )
    logger.info("HF mirror complete: %s", target.repo_id)
    return target


def try_sync_to_hub(
    local_path: str | os.PathLike[str],
    *,
    push: bool | None = None,
    config_path: str | os.PathLike[str] | None = None,
    large: bool = False,
) -> None:
    """Best-effort wrapper for post-run hooks: never raises, logs on failure.

    Respects :func:`push_enabled` (the ``--no-push`` flag / ``LEROBOT_HF_PUSH``).
    """
    if not push_enabled(push):
        logger.info("HF auto-push disabled (LEROBOT_HF_PUSH/--no-push); skipping %s", local_path)
        return
    try:
        sync_to_hub(local_path, config_path=config_path, large=large)
    except Exception as exc:  # noqa: BLE001 - hook must not crash the run
        logger.warning(
            "HF auto-push failed for %s: %s\n"
            "Your local data is intact. Re-run later with: lerobot-ros-backup %s",
            local_path,
            exc,
            local_path,
        )


def verify(
    local_path: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> tuple[bool, list[str]]:
    """Compare local files under ``local_path`` against the mapped HF repo.

    Returns ``(ok, missing)`` where ``missing`` is the list of repo-relative
    paths present locally but absent on the Hub. Use this before deleting a
    local folder.
    """
    abs_path = Path(local_path).resolve()
    target = resolve_target(abs_path, config_path=config_path)
    api = _hf_api()

    prefix = (target.path_in_repo + "/") if target.path_in_repo else ""
    try:
        remote = set(
            api.list_repo_files(repo_id=target.repo_id, repo_type=target.repo_type)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list %s: %s", target.repo_id, exc)
        return False, []

    local_files = [
        prefix + p.relative_to(abs_path).as_posix()
        for p in abs_path.rglob("*")
        if p.is_file()
    ]
    missing = sorted(f for f in local_files if f not in remote)
    ok = not missing
    logger.info(
        "Verify %s vs %s: %d local files, %d missing on Hub",
        abs_path,
        target.repo_id,
        len(local_files),
        len(missing),
    )
    return ok, missing
