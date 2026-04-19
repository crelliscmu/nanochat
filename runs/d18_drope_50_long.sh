#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00

DEPTH=18
MAX_SEQ_LEN=8192
SOURCE_TAG="d${DEPTH}_20tpp_drope_50"
MODEL_TAG="${SOURCE_TAG}_long"

# Default intermediate artifacts directory is in ~/.cache/nanochat
export SCRATCH_DIR="${TMPDIR:-$HOME}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$SCRATCH_DIR/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Number of GPUs/processes per node for distributed eval. Set to match the SBATCH header above.
NPROC_PER_NODE=1

mkdir -p $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG
rsync -a $HOME/nano_789/psc_artifacts/base_checkpoints/$SOURCE_TAG/ $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/
mkdir -p $NANOCHAT_BASE_DIR/tokenizer/$MODEL_TAG
rsync -a $HOME/nano_789/psc_artifacts/tokenizer/$SOURCE_TAG/ $NANOCHAT_BASE_DIR/tokenizer/$MODEL_TAG/
FINAL_DEST_DIR="${FINAL_DEST_DIR:-$HOME/nano_789/nanochat_artifacts}"
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
# Synthetic identity conversations.
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# Run SFT and eval the model. Long-context SFT: bump --max-seq-len to 8192,
# drop --device-batch-size to keep memory at parity (attention is quadratic in
# seq_len), and mix in 20K rows of allenai/tulu-v2-sft-long-mixture.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --model-tag $MODEL_TAG --device-batch-size=4 --max-seq-len=$MAX_SEQ_LEN --tulu-long-rows=100000 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- --model-tag $MODEL_TAG -i sft --run=$WANDB_RUN

python -m transformers.models.nanochat.convert_nanochat_checkpoints --input_dir "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG" --output_dir "$NANOCHAT_BASE_DIR/hf_sft/$MODEL_TAG"

python -m scripts.longcontext_eval -i sft --model-tag "$MODEL_TAG" --tensor-parallel-size=$NPROC_PER_NODE --hf-path "$NANOCHAT_BASE_DIR/hf_sft/$MODEL_TAG"

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate --model-tag $MODEL_TAG

echo "Copying model files and reports to $FINAL_DEST_DIR ..."
mkdir -p "$FINAL_DEST_DIR"
for sub in base_checkpoints/$MODEL_TAG chatsft_checkpoints/$MODEL_TAG hf_sft/$MODEL_TAG report/$MODEL_TAG tokenizer/$MODEL_TAG; do
    if [ -d "$NANOCHAT_BASE_DIR/$sub" ]; then
        mkdir -p "$FINAL_DEST_DIR/$sub"
        rsync -a "$NANOCHAT_BASE_DIR/$sub/" "$FINAL_DEST_DIR/$sub/"
    fi
done
echo "Artifacts copied to $FINAL_DEST_DIR"
