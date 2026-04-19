#!/bin/bash
# Convert every nanochat checkpoint under nanochat_artifacts/ to HuggingFace
# format. Outputs land in `<artifacts>/hf_<phase>/<tag>/` where phase is one of
# baseand sft and tag is the model-tag directory (e.g. d18, d20_drope_50).
#
# Prereq: the `transformers` package must already be patched with the nanochat
# model — run `python dev/patch_transformers_nanochat.py` once first.
#
# Usage:
#   bash dev/convert_all_artifacts.sh                   # convert everything
#   ARTIFACTS_DIR=/path/to/nanochat_artifacts bash ...  # custom artifacts root

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$REPO_ROOT/nanochat_artifacts}"

if [ ! -d "$ARTIFACTS_DIR" ]; then
    echo "Artifacts directory not found: $ARTIFACTS_DIR" >&2
    exit 1
fi

cd "$REPO_ROOT"
if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Map phase label -> source subdirectory under $ARTIFACTS_DIR.
declare -A PHASE_DIRS=(
    [base]="base_checkpoints"
    [sft]="chatsft_checkpoints"
)

converted=0
skipped=0
failed=0

for phase in base sft; do
    src_root="$ARTIFACTS_DIR/${PHASE_DIRS[$phase]}"
    if [ ! -d "$src_root" ]; then
        continue
    fi
    # Iterate model-tag directories (e.g. d18, d20_drope_50). `nullglob` keeps
    # us from looping over a literal `*/` when the phase dir is empty.
    shopt -s nullglob
    for ckpt in "$src_root"/*/; do
        shopt -u nullglob
        tag="$(basename "$ckpt")"
        out_dir="$ARTIFACTS_DIR/hf_$phase/$tag"
        echo ""
        echo "=== [$phase/$tag] -> $out_dir ==="
        if [ -f "$out_dir/config.json" ]; then
            echo "Already converted (config.json exists). Skipping."
            skipped=$((skipped + 1))
            continue
        fi
        if python -m transformers.models.nanochat.convert_nanochat_checkpoints \
                --input_dir "${ckpt%/}" \
                --output_dir "$out_dir"; then
            converted=$((converted + 1))
        else
            echo "Conversion failed for $phase/$tag" >&2
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "Done. converted=$converted skipped=$skipped failed=$failed"
if [ "$failed" -gt 0 ]; then
    exit 1
fi
