#!/bin/bash
set -euo pipefail

if [ -n "$COLAB_RELEASE_TAG" ]; then
    export UV_SYSTEM_PYTHON='false'
    echo "export HOME=/content" >> /content/nanochat/.venv/bin/activate
    export ARTIFACTS_DIR=/content/nanochat/nanochat_artifacts
    mkdir -p $ARTIFACTS_DIR
else
    echo "Not a Colab release tag"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

uv sync --extra gpu --extra longcontext

echo "export HOME=/content" >> /content/nanochat/.venv/bin/activate
source .venv/bin/activate

echo "Patching installed transformers with NanoChat model"
python "$REPO_ROOT/dev/patch_transformers_nanochat.py"
uv pip install -e .

echo "Done."
