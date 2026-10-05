# Public-release migration

The current repository already uses the public-release layout: executable helpers live under `scripts/`, committed configuration templates live under `config/` with the `*.example.yaml` suffix, and real machine configuration belongs under ignored `config.local/`.

For an older checkout, migrate local files manually before updating:

1. Preserve real machine configuration under `config.local/`; never copy it into a tracked example.
2. Remove obsolete duplicate scripts and templates only after confirming that their canonical replacements exist under `scripts/` and `config/`.
3. Create ignored `compose.hardware.yaml` from `config/compose.hardware.example.yaml` and replace every placeholder locally.
4. Inspect `git status` before committing.

Audit the complete Git history with Gitleaks because replacing the current tree does not erase earlier private inventory. Review license ownership with all contributors. `constraints.txt` is only a compatibility redirect; use `environment.yml` and `requirements.lock.txt`.
