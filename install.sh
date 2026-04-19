#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu --extra longcontext
source .venv/bin/activate

echo "Patching installed transformers with NanoChat model"
python "$REPO_ROOT/dev/patch_transformers_nanochat.py"
uv pip install -e .

echo "Done."
