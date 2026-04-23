#!/bin/bash
set -euo pipefail

if [ -n "$COLAB_RELEASE_TAG" ]; then
    echo "installing in google colab"
    export UV_SYSTEM_PYTHON='false'
    uv sync --extra gpu --extra longcontext
    export ARTIFACTS_DIR=/content/nanochat/nanochat_artifacts
    mkdir -p $ARTIFACTS_DIR
    echo "export ARTIFACTS_DIR=/content/nanochat/nanochat_artifacts" >> /content/nanochat/.venv/bin/activate
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$REPO_ROOT"

    uv sync --extra gpu --extra longcontext

    echo "export HOME=/content" >> /content/nanochat/.venv/bin/activate
    echo "export OMP_NUM_THREADS=1" >> /content/nanochat/.venv/bin/activate
    echo "export NANOCHAT_BASE_DIR=/content/nanochat/nanochat_artifacts" >> /content/nanochat/.venv/bin/activate
    source .venv/bin/activate

    echo "Patching installed transformers with NanoChat model"
    python "$REPO_ROOT/dev/patch_transformers_nanochat.py"
    uv pip install -e .
    bash "$REPO_ROOT/dev/download_from_hf.sh"

    mkdir -p $NANOCHAT_BASE_DIR
else
    echo "Not a Colab release tag"
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$REPO_ROOT"

    uv sync --extra gpu --extra longcontext

    echo "export HOME=/content" >> /content/nanochat/.venv/bin/activate
    source .venv/bin/activate

    echo "Patching installed transformers with NanoChat model"
    python "$REPO_ROOT/dev/patch_transformers_nanochat.py"
    uv pip install -e .
fi

echo "Done."
