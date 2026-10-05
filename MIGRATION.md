# Public-release migration

This release consolidates executable helpers under `scripts/` and committed configuration templates under `config/` using the `*.example.yaml` suffix. Real machine configuration remains under ignored `config.local/`.

After extracting the update at the repository root, run:

```bash
./scripts/cleanup_public_release_layout.sh
```

The cleanup removes obsolete duplicate paths only after checking that the canonical replacements exist. Review its output, then inspect `git status`. Preserve real configuration under `config.local/`; never copy it back into a tracked example.

Audit the complete Git history with Gitleaks because replacing the current tree does not erase earlier private inventory. Review license ownership with all contributors. Create ignored `compose.hardware.yaml` from `config/compose.hardware.example.yaml` and never commit it. `constraints.txt` is only a compatibility redirect; use `environment.yml` and `requirements.lock.txt`.
