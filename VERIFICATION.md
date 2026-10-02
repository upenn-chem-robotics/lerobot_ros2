# Verification delta

Extract this archive at the repository root after the public-release delta.
Run:

```bash
./verify_public_release.sh
```

The script writes a complete timestamped log and a compact summary under `release-logs/`. It continues after individual failures so the log captures all checks. The directory is ignored through `.gitignore.verification`; append that file to `.gitignore` or keep the generated logs untracked.
