#!/usr/bin/env python
"""Mirror local ``data/`` folders to private Hugging Face dataset repos.

This is the manual sync/verify/delete tool. Record / dagger / train do **not**
upload by default; use this command (or pass ``--push`` on those CLIs) to back
up data to Hugging Face.

Examples::

    # Authenticate once (write token from huggingface.co/settings/tokens):
    hf auth login            # or: export HF_TOKEN=hf_xxx

    # One-time backup of every experiment folder listed in config/hf_backup.yaml
    # (resumable; safe to re-run after an interruption):
    lerobot-ros-backup --all

    # Back up / re-push a single folder under data/:
    lerobot-ros-backup data/rama/dose_solid

    # Back up a folder outside data/ (e.g. on a USB stick) with an explicit repo:
    lerobot-ros-backup --repo lerobot-data-rama-dose_solid /media/rama/.../dose_solid

    # Check a repo is complete before deleting locally:
    lerobot-ros-backup --verify data/rama/dose_solid

    # Delete a whole HF dataset repo (local files untouched):
    lerobot-ros-backup --delete data/smrithi/pick_vial_20260524

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
        description="Mirror local folders to private Hugging Face dataset repos.",
    )
    p.add_argument(
        "path",
        nargs="?",
        help="Local folder to upload, verify, or delete (mapped via data/ layout or --repo).",
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
        "--delete",
        action="store_true",
        help="Delete the mapped HF dataset repo (local files are not touched).",
    )
    p.add_argument(
        "--delete-all",
        action="store_true",
        help="Delete every HF repo listed in config/hf_backup.yaml backup_targets "
             "(and --extra-repo names). Local files are not touched.",
    )
    p.add_argument(
        "--extra-repo",
        action="append",
        default=[],
        metavar="NAME",
        help="Additional HF repo name to include in --delete-all (repeatable).",
    )
    p.add_argument(
        "--repo",
        metavar="NAME",
        help="HF repo name without username (e.g. lerobot-data-rama-dose_solid). "
             "Required for paths outside data/; optional override otherwise.",
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
        help="Skip the confirmation prompt for --all or --delete.",
    )
    return p.parse_args(argv)


def _resolve_mapping(path: str | None, repo: str | None, config: str | None) -> hub_sync.RepoTarget:
    if repo is not None:
        return hub_sync.resolve_target_from_repo(repo, config_path=config)
    if path is None:
        raise ValueError("Provide PATH or --repo.")
    return hub_sync.resolve_target(path, config_path=config)


def _verify_one(path: str, config: str | None, repo: str | None = None) -> bool:
    ok, missing = hub_sync.verify(path, config_path=config, repo_name=repo)
    label = path if path else f"repo:{repo}"
    if ok:
        print(f"[OK] {label} is fully mirrored on the Hub.")
    else:
        print(f"[INCOMPLETE] {label}: {len(missing)} file(s) missing on the Hub.")
        for m in missing[:20]:
            print(f"    missing: {m}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
    return ok


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    cfg = hub_sync.load_config(args.config)

    if args.repo and args.all:
        sys.exit("Use either --repo or --all, not both.")
    if args.delete and args.delete_all:
        sys.exit("Use either --delete or --delete-all, not both.")

    # Determine the list of local roots to act on.
    if args.delete_all:
        repo_names = [repo for _root, repo in cfg.targets] + list(args.extra_repo)
        targets = [("", name) for name in dict.fromkeys(repo_names)]
    elif args.all:
        targets: list[tuple[str, str | None]] = [(str(root), None) for root, _repo in cfg.targets]
        if not targets:
            sys.exit("No backup_targets configured in hf_backup.yaml.")
    elif args.path or args.repo:
        if args.path is None and not args.delete:
            sys.exit("Provide PATH when using --repo (except with --delete).")
        targets = [(args.path or "", args.repo)]
    else:
        sys.exit("Provide a PATH or use --all. See --help.")

    # Delete mode.
    if args.delete or args.delete_all:
        if not args.yes:
            names = []
            for path, repo in targets:
                tgt = _resolve_mapping(path or None, repo, args.config)
                names.append(tgt.repo_id)
            print("About to DELETE these Hugging Face dataset repos (local files untouched):")
            for name in names:
                print(f"  - {name}")
            resp = input("Continue? [y/N] ").strip().lower()
            if resp not in ("y", "yes"):
                sys.exit("Aborted.")
        failures = 0
        for path, repo in targets:
            try:
                hub_sync.delete_repo(path or None, repo_name=repo, config_path=args.config)
                tgt = _resolve_mapping(path or None, repo, args.config)
                print(f"[DELETED] {tgt.repo_id}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logging.error("Failed to delete %s: %s", path or repo, exc)
        sys.exit(1 if failures else 0)

    # Verify mode.
    if args.verify:
        all_ok = True
        for path, repo in targets:
            if not path:
                sys.exit("--verify requires a local PATH.")
            all_ok &= _verify_one(path, args.config, repo=repo)
        sys.exit(0 if all_ok else 1)

    # Dry-run: just print the mapping.
    if args.dry_run:
        for path, repo in targets:
            tgt = _resolve_mapping(path or None, repo, args.config)
            suffix = f" :/{tgt.path_in_repo}" if tgt.path_in_repo else ""
            src = path or f"(repo {repo})"
            print(f"{src}  ->  {tgt.repo_id} ({tgt.repo_type}, private={tgt.private}){suffix}")
        return

    # Confirm before a large bulk upload.
    if args.all and not args.yes:
        print(f"About to mirror {len(targets)} folder(s) to private repos under "
              f"'{cfg.hf_user}/'. This can transfer hundreds of GB.")
        resp = input("Continue? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            sys.exit("Aborted.")

    failures = 0
    for path, repo in targets:
        if not path:
            sys.exit("Upload requires a local PATH.")
        try:
            hub_sync.sync_to_hub(
                path,
                config_path=args.config,
                repo_name=repo,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logging.error("Failed to back up %s: %s", path, exc)

    if failures:
        sys.exit(f"{failures} folder(s) failed. Re-run to resume (uploads are resumable).")
    print("Backup complete.")


if __name__ == "__main__":
    main()
