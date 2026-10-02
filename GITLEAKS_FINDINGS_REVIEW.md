# Gitleaks findings review

The redacted report from the `20261002T115921Z` verification run contains four `generic-api-key` findings. All four refer to the same historical commit and the generated Conda export `environment.yml`.

The redacted matches are package assignments:

- line 98: `client=<version>`
- line 125: `keyutils=<version>`
- line 183: `access=<version>`
- line 354: `keyboard-handler=<version>`

These are package names and versions, not credentials. `.gitleaksignore` suppresses only the four exact fingerprints. The generic API-key detector remains enabled for every other file, line, and commit.
