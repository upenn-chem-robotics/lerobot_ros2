# Functional verification fixes before Trivy

This update addresses the failures from the interrupted verification run:

- installs Gradio because `lerobot-ros-app` is a registered runtime command and imports it
- removes import-time mutation of OpenCV's installed `site-packages/cv2` directory
- canonicalizes Python distribution names before policy-plugin comparison
- mounts source read-only for tests and lint
- redirects pytest cache to `/tmp` and disables Ruff cache
- runs Trivy only when all earlier required checks, including Gitleaks, have passed

The Docker build cache and generated images remain enabled and retained by default.
