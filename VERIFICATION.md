# Release verification contract

This contract defines the required release process. Perform the steps in order. Do not commit, tag, or push the repository until the working tree, cold-built image, and local documentation have all been verified.

Replace `vX.Y.Z` with the release version selected after verification. The Git tag, container image tag, Mike documentation version, and documented repository reference must use the same value.

## 1. Verify the uncommitted working tree

Start from the repository root with the complete release change present in the working tree.

```bash
git status --short
git diff --check
pytest -q tests/test_compose_identity.py
bash scripts/test_no_hardware.sh
bash scripts/verify_public_release.sh
```

All checks must pass against the exact tree intended for release.

The scripts write timestamped output under `release-logs/`. This directory is ignored by Git and must not be committed.

Do not commit yet. Do not use `SKIP_BUILD=1` for this verification.

These checks do not establish hardware, checkpoint, external-dataset, GPU, camera, or physical-robot validation. Record any additional validation required by the release separately.

## 2. Cold-build an image from that tree

Build the runtime image without layer cache so that the candidate image is derived from the current verified tree rather than cached build output.

```bash
IMAGE=ghcr.io/upenn-chem-robotics/lerobot-ros2
CANDIDATE_TAG=release-candidate-local

docker build --no-cache --pull --target runtime \
  -t "$IMAGE:$CANDIDATE_TAG" .
```

Do not push this candidate yet.

## 3. Reverify the cold-built image

Run the verification suite against the cold-built candidate without rebuilding it:

```bash
SKIP_BUILD=1 \
RUNTIME_IMAGE="$IMAGE:$CANDIDATE_TAG" \
DEV_IMAGE="$IMAGE:$CANDIDATE_TAG" \
  bash scripts/test_no_hardware.sh

RUNTIME_IMAGE="$IMAGE:$CANDIDATE_TAG" \
  bash scripts/verify_public_release.sh
```

If `scripts/verify_public_release.sh` does not consume `RUNTIME_IMAGE`, run its documented image-selection mechanism instead. The release record must make clear which image digest was tested.

Record the local candidate image identifier:

```bash
docker image inspect "$IMAGE:$CANDIDATE_TAG" \
  --format '{{.Id}}'
```

If any check fails, fix the tree and restart from step 1.

## 4. Check the documentation locally

Select a provisional semantic version for the local documentation check. It is not final until the preceding verification and this review pass.

```bash
VERSION=vX.Y.Z

python -m pip install -r requirements-docs.txt
git branch -D docs-test 2>/dev/null || true

DOCS_VERSION="$VERSION" \
DOCS_REPOSITORY_REF="$VERSION" \
DOCS_IMAGE_TAG="$VERSION" \
  mike deploy --branch docs-test --update-aliases \
  "$VERSION" latest

mike serve --branch docs-test
```

Verify that:

- `mkdocs build --strict` succeeds;
- the version selector shows `$VERSION` and the `latest` alias;
- installation commands reference repository tag `$VERSION`;
- generated environment examples use image tag `$VERSION`;
- navigation, internal links, and code examples render correctly;
- host identity variables are `LEROBOT_HOST_UID` and `LEROBOT_HOST_GID`;
- no command exports Bash's read-only `UID` variable.

This is a local preview. Do not add `--push`.

After review:

```bash
git branch -D docs-test
```

If documentation changes are required, update the tree and restart from step 1.

## 5. Decide the release tag

After the tree, cold-built image, and local documentation pass verification, choose the final semantic version:

```bash
VERSION=vX.Y.Z
```

The selected version must not already identify a different release. Use this exact value for the Git tag, image tag, Mike version, repository reference, and documented image tag.

Do not create or push the Git tag yet.

## 6. Tag and push the verified image

Apply the final version tag to the exact local image that passed reverification:

```bash
docker tag "$IMAGE:$CANDIDATE_TAG" "$IMAGE:$VERSION"
docker push "$IMAGE:$VERSION"
```

Do not rebuild between verification and publication.

Record and verify the published digest:

```bash
docker inspect "$IMAGE:$VERSION" --format '{{index .RepoDigests 0}}'
```

The published version tag must refer to the same image content that passed step 3.

## 7. Commit and push the verified tree

Only after the versioned image has been published, commit the exact verified tree:

```bash
git status --short
git diff --check
git add .
git commit -m "Release $VERSION"
```

Confirm that the commit contains only the verified release changes. Then create the release tag and push the commit before the tag:

```bash
git tag -a "$VERSION" -m "Release $VERSION"
git push origin HEAD
git push origin "$VERSION"
```

Pushing or moving a `v*` tag triggers documentation publication. The documentation workflow starts from an empty, local-only Mike branch, rebuilds every `v*` Git tag so the selector and published directories describe the same complete tag set, makes the highest version-sorted tag the site default, and uploads the generated tree directly as a GitHub Pages artifact. It does not create, read, or push a `gh-pages` branch, and it does not pull or verify the container image. The `latest` alias is used only on the temporary local `docs-test` branch. Image verification is completed before the repository and tag are pushed.

## Acceptance gate

A release is accepted only when all of the following are true:

- the uncommitted release tree passed the repository verification suite;
- the runtime image was built with `--no-cache --pull` from that tree;
- the cold-built image passed reverification;
- the documentation was reviewed locally using the final version format;
- a previously unused `vX.Y.Z` version was selected;
- the exact verified image was tagged and pushed as that version without rebuilding;
- the published image digest was recorded;
- the exact verified tree was committed only after image publication;
- the commit was pushed before the Git tag;
- the Git tag matches the image and documentation version exactly.
