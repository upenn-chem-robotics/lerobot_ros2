# Public-release migration

This delta overwrites tracked lab configuration with sanitized examples. Preserve real configuration under ignored `config.local/`. Audit complete Git history with Gitleaks because current-tree replacement does not erase private inventory. Review license ownership with all contributors. Create `compose.hardware.yaml` from the example and never commit it. `constraints.txt` is only a compatibility redirect; use `environment.yml` and `requirements.lock.txt`.
