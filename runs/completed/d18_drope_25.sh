#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --time=8:00:00

DEPTH=18
MAX_SEQ_LEN=4096
MODEL_TAG="d${DEPTH}_drope_25"

# Default intermediate artifacts directory is in ~/.cache/nanochat
export SCRATCH_DIR="${TMPDIR:-$HOME}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$SCRATCH_DIR/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Number of GPUs/processes per node for distributed eval. Set to match the SBATCH header above.
NPROC_PER_NODE=1

# Where to copy model files and reports after training completes. SCRATCH_DIR is ephemeral, so we persist artifacts to FINAL_DEST_DIR.
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
# Write a header section with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset --model-tag $MODEL_TAG

# -----------------------------------------------------------------------------
# Tokenizer
# Download the first ~2B characters of pretraining dataset. Each data shard is ~250M chars, so we download 2e9 / 250e6 = 8 data shards at this point. Each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk. Look at dev/repackage_data_reference.py for details on how this data was prepared.
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# Train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data.
python -m scripts.tok_train --model-tag $MODEL_TAG
# Evaluate the tokenizer (report compression ratio etc.).
python -m scripts.tok_eval --model-tag $MODEL_TAG

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=$DEPTH --model-tag $MODEL_TAG --target-param-data-ratio=9 --device-batch-size=16 --max-seq-len=$MAX_SEQ_LEN --fp8 --save-every-pct=25 --run=$WANDB_RUN --rope-removal-pct=25
# Evaluate the model: CORE metric, BPB on train/val, and draw samples.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --model-tag $MODEL_TAG --device-batch-size=16 --run=$WANDB_RUN

# Long-context eval on the base (pre-SFT) checkpoint.
LONGCONTEXT_TASKS="LongBench-qasper|LongBench-hotpotqa|LongBench-gov_report|LongBench-trec|LongBench-passage_retrieval_en|LongBench-lcc|NoLiMa|LongGenBench"
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.longcontext_eval -- -i base --model-tag $MODEL_TAG -a "$LONGCONTEXT_TASKS" --max-context-len 8192 --nolima-lens 1024,2048,4096,8192 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Synthetic identity conversations.
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# Run SFT and eval the model.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --model-tag $MODEL_TAG --device-batch-size=16 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- --model-tag $MODEL_TAG -i sft --run=$WANDB_RUN

# Long-context eval on the SFT checkpoint (same task list as the base eval above
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.longcontext_eval -- -i sft --model-tag $MODEL_TAG -a "$LONGCONTEXT_TASKS" --max-context-len 8192 --nolima-lens 1024,2048,4096,8192 --run=$WANDB_RUN

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
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
