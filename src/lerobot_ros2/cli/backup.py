#!/usr/bin/env python
"""Mirror local ``data/`` folders to private Hugging Face dataset repos.

This is the one-time bulk-backup driver *and* the manual sync/verify tool. The
record / dagger / train CLIs reuse :mod:`lerobot_ros2.hub_sync` to push new data
automatically; use this command for the initial backup of existing data, to
re-push a folder, or to verify a repo before deleting it locally.

Examples::

    # Authenticate once (write token from huggingface.co/settings/tokens):
    hf auth login            # or: export HF_TOKEN=hf_xxx

    # One-time backup of every experiment folder listed in config/hf_backup.yaml
    # (resumable; safe to re-run after an interruption):
    lerobot-ros-backup --all

    # Back up / re-push a single folder:
    lerobot-ros-backup data/rama/dose_solid

    # Check a repo is complete before deleting locally:
    lerobot-ros-backup --verify data/rama/dose_solid

    # See where things would go without uploading:
    lerobot-ros-backup --all --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys

from lerobot_ros2 import hub_sync


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="lerobot-ros-backup",
        description="Mirror local data/ folders to private Hugging Face dataset repos.",
    )
    p.add_argument(
        "path",
        nargs="?",
        help="Local folder under data/ to upload (or verify with --verify).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Back up every folder in config/hf_backup.yaml backup_targets.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Compare local files against the Hub instead of uploading.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to hf_backup.yaml (default: config/hf_backup.yaml).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resolved repo mapping(s) without creating repos or uploading.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt for --all.",
    )
    return p.parse_args(argv)


def _verify_one(path: str, config: str | None) -> bool:
    ok, missing = hub_sync.verify(path, config_path=config)
    if ok:
        print(f"[OK] {path} is fully mirrored on the Hub.")
    else:
        print(f"[INCOMPLETE] {path}: {len(missing)} file(s) missing on the Hub.")
        for m in missing[:20]:
            print(f"    missing: {m}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
    return ok


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    cfg = hub_sync.load_config(args.config)

    # Determine the list of local roots to act on.
    if args.all:
        targets = [str(root) for root, _repo in cfg.targets]
        if not targets:
            sys.exit("No backup_targets configured in hf_backup.yaml.")
    elif args.path:
        targets = [args.path]
    else:
        sys.exit("Provide a PATH or use --all. See --help.")

    # Verify mode.
    if args.verify:
        all_ok = True
        for t in targets:
            all_ok &= _verify_one(t, args.config)
        sys.exit(0 if all_ok else 1)

    # Dry-run: just print the mapping.
    if args.dry_run:
        for t in targets:
            tgt = hub_sync.resolve_target(t, config_path=args.config)
            suffix = f" :/{tgt.path_in_repo}" if tgt.path_in_repo else ""
            print(f"{t}  ->  {tgt.repo_id} ({tgt.repo_type}, private={tgt.private}){suffix}")
        return

    # Confirm before a large bulk upload.
    if args.all and not args.yes:
        print(f"About to mirror {len(targets)} folder(s) to private repos under "
              f"'{cfg.hf_user}/'. This can transfer hundreds of GB.")
        resp = input("Continue? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            sys.exit("Aborted.")

    failures = 0
    for t in targets:
        try:
            # ``large=True`` => upload_large_folder (resumable) for whole-folder
            # backups; falls back to upload_folder automatically when the path
            # maps to a sub-path of a repo.
            hub_sync.sync_to_hub(t, config_path=args.config, large=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logging.error("Failed to back up %s: %s", t, exc)

    if failures:
        sys.exit(f"{failures} folder(s) failed. Re-run to resume (uploads are resumable).")
    print("Backup complete.")


if __name__ == "__main__":
    main()
