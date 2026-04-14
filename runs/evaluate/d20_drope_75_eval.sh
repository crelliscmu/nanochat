#!/bin/bash
#SBATCH --gres=gpu:h200:1
#SBATCH --time=4:00:00

DEPTH=20
MODEL_TAG="d${DEPTH}_drope_75"

# Default intermediate artifacts directory is in ~/.cache/nanochat
export SCRATCH_DIR="${TMPDIR:-$HOME}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$SCRATCH_DIR/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Number of GPUs/processes per node for distributed eval. Set to match the SBATCH header above.
NPROC_PER_NODE=1

# Where to copy model files and reports after eval completes. SCRATCH_DIR is ephemeral, so we persist artifacts to FINAL_DEST_DIR.
FINAL_DEST_DIR="${FINAL_DEST_DIR:-$HOME/nano_789/nanochat_artifacts/$MODEL_TAG}"

# Python venv setup with uv. Install uv (if not already installed), create a .venv local virtual environment (if it doesn't exist), install the repo dependencies, and activate venv so that `python` uses the project's venv instead of system python.
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu --extra longcontext
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=$MODEL_TAG
fi

# -----------------------------------------------------------------------------
# Evaluations

# Evaluate the base model: CORE metric, BPB on train/val, and draw samples.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --model-tag $MODEL_TAG --device-batch-size=16 --run=$WANDB_RUN

# Long-context eval on the base (pre-SFT) checkpoint.
LONGCONTEXT_TASKS="LongBench-qasper|LongBench-hotpotqa|LongBench-gov_report|LongBench-trec|LongBench-passage_retrieval_en|LongBench-lcc|NoLiMa|LongGenBench"
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.longcontext_eval -- -i base --model-tag $MODEL_TAG -a "$LONGCONTEXT_TASKS" --max-context-len 8192 --nolima-lens 1024,2048,4096,8192 --run=$WANDB_RUN

# chat_eval on the SFT checkpoint.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- --model-tag $MODEL_TAG -i sft --run=$WANDB_RUN

# Long-context eval on the SFT checkpoint.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.longcontext_eval -- -i sft --model-tag $MODEL_TAG -a "$LONGCONTEXT_TASKS" --max-context-len 8192 --nolima-lens 1024,2048,4096,8192 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
python -m nanochat.report generate --model-tag $MODEL_TAG

# -----------------------------------------------------------------------------
# Copy model files and reports from the ephemeral SCRATCH_DIR back to
# FINAL_DEST_DIR so they survive past the job's lifetime.
echo "Copying model files and reports to $FINAL_DEST_DIR ..."
mkdir -p "$FINAL_DEST_DIR"
for sub in base_checkpoints chatsft_checkpoints chatrl_checkpoints report tokenizer/$MODEL_TAG; do
    if [ -d "$NANOCHAT_BASE_DIR/$sub" ]; then
        mkdir -p "$FINAL_DEST_DIR/$sub"
        rsync -a "$NANOCHAT_BASE_DIR/$sub/" "$FINAL_DEST_DIR/$sub/"
    fi
done
echo "Artifacts copied to $FINAL_DEST_DIR"
