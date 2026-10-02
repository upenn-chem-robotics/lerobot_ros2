# Verification cache controls

Verification now uses Docker's build cache and retains the generated runtime and development images by default.

```bash
./verify_public_release.sh
```

For the final reproducibility gate, force a cold build:

```bash
COLD_RELEASE_BUILD=1 ./verify_public_release.sh
```

To remove verification images automatically when a run ends:

```bash
KEEP_RELEASE_IMAGES=0 ./verify_public_release.sh
```
