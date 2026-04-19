#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00

DEPTH=20
MAX_SEQ_LEN=8192
SOURCE_TAG="d${DEPTH}_20tpp"
MODEL_TAG="${SOURCE_TAG}_long"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/nanochat/nanochat_artifacts"
mkdir -p $NANOCHAT_BASE_DIR

# Number of GPUs/processes per node for distributed eval. Set to match the SBATCH header above.
NPROC_PER_NODE=1

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=$MODEL_TAG
fi
source .venv/bin/activate
# -----------------------------------------------------------------------------
# Write a header section with a bunch of system info and a timestamp that marks the start of the run.


# -----------------------------------------------------------------------------
# Synthetic identity conversations.
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# Run SFT and eval the model. Long-context SFT: bump --max-seq-len to 8192,
# drop --device-batch-size to keep memory at parity (attention is quadratic in
# seq_len), and mix in 100K rows of allenai/tulu-v2-sft-long-mixture.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --model-tag $MODEL_TAG --source-tag $SOURCE_TAG --device-batch-size=4 --max-seq-len=$MAX_SEQ_LEN --tulu-long-rows=150000 --run=$WANDB_RUN
# chat_eval looks up the tokenizer under MODEL_TAG; mirror it from SOURCE_TAG.
cp -rn "$NANOCHAT_BASE_DIR/tokenizer/$SOURCE_TAG" "$NANOCHAT_BASE_DIR/tokenizer/$MODEL_TAG"
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- --model-tag $MODEL_TAG -i sft --run=$WANDB_RUN

python -m transformers.models.nanochat.convert_nanochat_checkpoints --input_dir "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG" --output_dir "$NANOCHAT_BASE_DIR/hf_sft/$MODEL_TAG"

python -m scripts.longcontext_eval -i sft --model-tag "$MODEL_TAG" --tensor-parallel-size=$NPROC_PER_NODE --hf-path "$NANOCHAT_BASE_DIR/hf_sft/$MODEL_TAG"

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web
