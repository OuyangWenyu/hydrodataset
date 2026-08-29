#!/usr/bin/env bash
# Cloud Agent install script for hydrodataset.
#
# Idempotent repository bootstrap: installs the uv toolchain when it is
# missing and syncs every project dependency (base plus all optional extras)
# into the project-local .venv, matching the project's CI setup. After this
# runs the library, the `hydrodataset` CLI, and the pytest suite are ready.
set -euo pipefail

# Install the uv package/environment manager if it is not already available.
# The official installer places binaries in ~/.local/bin and wires that
# directory into the login-shell PATH via ~/.local/bin/env.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv --version

# Resolve and install all dependencies from the committed uv.lock into .venv.
# --all-extras mirrors the CI job (dev, docs, lint, release extras included).
uv sync --all-extras
