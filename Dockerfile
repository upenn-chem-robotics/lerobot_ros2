# syntax=docker/dockerfile:1.7
ARG MINIFORGE_IMAGE=condaforge/miniforge3:26.3.2-3
FROM ${MINIFORGE_IMAGE} AS environment
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates libgl1 libglib2.0-0 v4l-utils tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN conda install -n base -y \
      "setuptools>=78.1.1,<81" "urllib3>=2.8,<3" "msgpack-python>=1.2.1,<2" && \
    conda clean -afy
COPY environment.yml requirements.lock.txt dependencies.env ./
RUN conda config --system --set channel_priority strict && \
    conda env create --file environment.yml && \
    conda clean -afy
RUN set -a && source dependencies.env && set +a && \
    conda run -n lerobot python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 --requirement requirements.lock.txt && \
    conda run -n lerobot python -m pip install --no-deps \
      "lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}"

FROM environment AS wheel-builder
RUN conda run -n lerobot python -m pip install --no-cache-dir "build>=1.2,<2"
COPY pyproject.toml README.md ./
COPY src ./src
COPY packages ./packages
RUN conda run -n lerobot python -m build --wheel --outdir /wheels . && \
    conda run -n lerobot python -m build --wheel --outdir /wheels packages/lerobot_policy_strided_diffusion && \
    conda run -n lerobot python -m build --wheel --outdir /wheels packages/lerobot_policy_action_history_diffusion

FROM environment AS runtime
ARG APP_UID=1000
ARG APP_GID=1000
RUN set -eux; \
    existing_group="$(getent group "${APP_GID}" | cut -d: -f1 || true)"; \
    if [ -z "${existing_group}" ]; then \
      groupadd --gid "${APP_GID}" app; \
      existing_group=app; \
    elif [ "${existing_group}" != app ]; then \
      groupmod --new-name app "${existing_group}"; \
      existing_group=app; \
    fi; \
    existing_user="$(getent passwd "${APP_UID}" | cut -d: -f1 || true)"; \
    if [ -z "${existing_user}" ]; then \
      useradd --uid "${APP_UID}" --gid "${existing_group}" --create-home app; \
    else \
      if [ "${existing_user}" != app ]; then usermod --login app "${existing_user}"; fi; \
      usermod --gid "${existing_group}" --home /home/app --move-home app; \
    fi
COPY --from=wheel-builder /wheels /wheels
RUN conda run -n lerobot python -m pip install --no-deps /wheels/*.whl && \
    conda run -n lerobot python -m pip check && rm -rf /wheels
COPY scripts/entrypoint.sh /usr/local/bin/lerobot-entrypoint
RUN chmod 0755 /usr/local/bin/lerobot-entrypoint && mkdir -p /data /config /cache/huggingface /cache/torch && \
    chown -R app:app /data /config /cache /home/app
ENV GELLO_CONFIG=/config/gello.yaml \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PYTHONUNBUFFERED=1
WORKDIR /workspace
USER app
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/lerobot-entrypoint"]
CMD ["lerobot-ros-doctor"]

FROM runtime AS dev
USER root
RUN conda run -n lerobot python -m pip install pytest==8.4.2 ruff==0.16.1
USER app
CMD ["bash"]
