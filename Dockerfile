# =============================================================================
# scPilot container image — STARTING ARTIFACT, NOT YET BUILT OR TESTED.
#
# This Dockerfile was authored on a host WITHOUT Docker available, so it has
# NOT been `docker build`-ed or run here. Treat it as a starting point for CI
# or another environment to validate/iterate on — expect to adjust pins,
# channels, or the install step on first real build.
# =============================================================================

# micromamba base: small, fast conda-compatible solver; ships a non-root `mambauser`.
FROM mambaorg/micromamba:1.5.8

# Recreate the exact scPilot conda environment from the exported spec.
# environment.yml declares `name: scpilot`; micromamba installs it into that env.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Make the env's interpreter the default for subsequent RUN/CMD layers.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install scPilot itself (editable) with the optional CNV + annotation extras
# ([extra] = infercnvpy, gtfparse, pybiomart, celltypist). --no-deps: every
# runtime dependency is already provided by environment.yml, so pip must not
# re-resolve/downgrade them from PyPI.
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
RUN pip install -e ".[extra]" --no-deps

# numba needs a writable cache dir; pin it away from a read-only $HOME.
ENV NUMBA_CACHE_DIR=/tmp/numba-cache

# Default: print the version (cheap smoke check). Override for MCP server / CLI.
CMD ["scpilot", "version"]
