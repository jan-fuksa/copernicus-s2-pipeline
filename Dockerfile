FROM mambaorg/micromamba:1.5.10

# ---- OS-level utilities (kept minimal) ----
# - ca-certificates: reliable TLS for git/lfs/pip/HTTPS
# - curl: optional but practical for debugging / downloads
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---- Create conda env as the default non-root micromamba user ----
# This keeps the env ownership/permissions consistent with the base image conventions.
USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba create -y -n ci -f /tmp/environment.yml \
 && micromamba clean -a -y \
 && rm -f /tmp/environment.yml

# ---- Activate env by default ----
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/conda/envs/ci/bin:$PATH

# Make micromamba activate 'ci' on container start (instead of default 'base')
ENV ENV_NAME=ci

# ---- Runtime defaults ----
WORKDIR /work

# Run as root in GitHub Actions job containers to avoid permission issues with mounted GITHUB_WORKSPACE
USER root
