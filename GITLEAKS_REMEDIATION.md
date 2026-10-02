# Gitleaks remediation

The verification run scanned 38 commits and reported four findings. The bundled log does not expose each finding's rule, path, and commit, so do not add an allowlist yet.

Generate a reviewable report locally:

```bash
gitleaks detect --source . --report-format json --report-path release-logs/gitleaks.json --redact
```

For newer Gitleaks CLIs, use:

```bash
gitleaks git --report-format json --report-path release-logs/gitleaks.json --redact
```

Review `release-logs/gitleaks.json`. Revoke and rotate any real credential before rewriting history. Remove sensitive files or values from every affected commit with `git filter-repo`, force-push the rewritten branches and tags, and have collaborators reclone. Add a narrow `.gitleaks.toml` allowlist only for a verified false positive, scoped by rule and path rather than disabling a detector globally.
