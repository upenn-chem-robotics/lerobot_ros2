# Verification and development publication contract

This document is the single verification contract for changes intended for the `devel` branch and the moving `latest` development image. Do not update either reference unless every required check below passes.

The development pair is:

- repository branch: `devel`
- container image: `ghcr.io/upenn-chem-robotics/lerobot-ros2:latest`
- local Mike version: `devel`
- local Mike alias: `latest`

This flow does not publish GitHub Pages. Documentation publication remains manual.

## 1. Verify the current tree

Run these commands from the repository root, in this order:

```bash
pytest -q tests/test_compose_identity.py
bash scripts/test_no_hardware.sh
bash scripts/verify_public_release.sh
```

All three commands are required:

1. `tests/test_compose_identity.py` verifies the Compose runtime-identity and host-owned bind-mount contract.
2. `scripts/test_no_hardware.sh` builds the runtime and development images and exercises the hardware-free integration path, including ownership of files created through `/data`.
3. `scripts/verify_public_release.sh` performs the broader public-image, packaging, dependency, static-analysis, security, and installed-distribution checks.

Both scripts write timestamped output under `release-logs/`. The directory is ignored by Git and must not be committed.

Do not use `SKIP_BUILD=1` for the final verification of a changed image or Compose contract. It is acceptable only when intentionally rerunning tests against explicitly selected, already-built images:

```bash
SKIP_BUILD=1 \
RUNTIME_IMAGE=<runtime-image> \
DEV_IMAGE=<development-image> \
  bash scripts/test_no_hardware.sh
```

These checks do not claim hardware, checkpoint, external-dataset, GPU, camera, or physical-robot validation.

## 2. Preview the development documentation locally

Install the documentation dependencies and build the current working tree as Mike version `devel` with alias `latest`:

```bash
python -m pip install -r requirements-docs.txt
git branch -D docs-test 2>/dev/null || true

DOCS_VERSION=devel \
DOCS_REPOSITORY_REF=devel \
DOCS_IMAGE_TAG=latest \
  mike deploy --branch docs-test --update-aliases devel latest

mike set-default --branch docs-test latest
mike serve --branch docs-test
```

Verify that:

- the version selector shows `devel` with alias `latest`;
- the installation page clones `--branch devel`;
- the generated `.env` example sets `IMAGE_TAG=latest`;
- the host identity variables are `LEROBOT_HOST_UID` and `LEROBOT_HOST_GID`;
- no command exports Bash's read-only `UID` variable.

Do not add `--push`. This preview must remain local.

## 3. Commit and push `devel`

After verification and local documentation review pass:

```bash
git status --short
git diff --check
git add .
git commit -m "Run containers as host user and add devel docs channel"
git push -u origin devel
```

The documentation workflow is manual-only, so pushing `devel` does not publish GitHub Pages.

## 4. Build the traceable development image

Use the existing `latest` image as build cache while building from the verified commit:

```bash
IMAGE=ghcr.io/upenn-chem-robotics/lerobot-ros2
SHA_TAG="devel-$(git rev-parse --short=12 HEAD)"

docker pull "$IMAGE:latest"

docker build --target runtime \
  --cache-from "$IMAGE:latest" \
  -t "$IMAGE:$SHA_TAG" \
  -t "$IMAGE:latest" .
```

The commit-derived tag is the traceable image reference. `latest` is the moving development reference used by the `devel` installation documentation.

## 5. Push the image references

Push the traceable reference first, then move `latest` to the same tested image:

```bash
docker push "$IMAGE:$SHA_TAG"
docker push "$IMAGE:latest"
```

Record the resulting digest in the verification notes or release log. The two tags should identify the same image content immediately after publication.

## Acceptance gate

The `devel` branch and `latest` image may be updated only when all of the following are true:

- the focused identity pytest passes;
- `scripts/test_no_hardware.sh` passes;
- `scripts/verify_public_release.sh` passes;
- the local Mike preview renders `devel` and `latest` correctly;
- `git diff --check` passes;
- the pushed traceable image tag is derived from the pushed `devel` commit;
- `latest` is applied to that same tested image.
