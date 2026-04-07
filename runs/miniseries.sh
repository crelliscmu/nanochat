#!/bin/bash

# See speedrun.sh for more comments
# Usage: ./miniseries.sh [series_name]
# Example: ./miniseries.sh jan11
# Default series name is today's date (e.g., jan11)

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$TMPDIR/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Series name: from arg, env var, or default to today's date (e.g., jan11)
SERIES_NAME="${1:-${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"
# All depths in this miniseries share a single tokenizer trained under this tag.
# Per-depth model_tags will be symlinked to it inside the loop, since base_train
# loads the tokenizer from tokenizer/<model_tag>/.
TOK_TAG="${SERIES_NAME}_miniseries_tok"

# Setup (skip with SKIP_SETUP=1)
if [ -z "$SKIP_SETUP" ]; then
    # uv
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate

    # Tokenizer, download 1000 shards for pretraining
    python -m nanochat.dataset -n 1000
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768 --model-tag="${TOK_TAG}"
else
    source .venv/bin/activate
fi

# Depths to train (the "miniseries")
#DEPTHS=(12 14 16 18 20 22 24 26)
DEPTHS=(12 14 16)
# Hardware
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
# Logging
WANDB_RUN="${WANDB_RUN:-${SERIES_NAME}_miniseries}"

RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_miniseries_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "depth,model_dim,num_params,num_scaling_params,num_iterations,tokens_trained,param_data_ratio,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "${SERIES_NAME} Miniseries Training"
log "=============================================="

for d in "${DEPTHS[@]}"; do
    log "Training d=$d..."

    TAG="${SERIES_NAME}_miniseries_d${d}"
    START_TIME=$(date +%s)

    # base_train loads the tokenizer from tokenizer/<model_tag>/, so point this
    # depth's tag at the shared TOK_TAG tokenizer (relative symlink so the link
    # resolves regardless of where NANOCHAT_BASE_DIR lives).
    TOKENIZER_BASE_DIR="$NANOCHAT_BASE_DIR/tokenizer"
    mkdir -p "$TOKENIZER_BASE_DIR"
    ln -sfn "$TOK_TAG" "$TOKENIZER_BASE_DIR/$TAG"

    # Reduce --device-batch-size to avoid OOM at larger depths
    if [ $d -ge 28 ]; then
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=8"
    elif [ $d -ge 20 ]; then
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=16"
    else
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=32"
    fi

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$d \
        --run="${WANDB_RUN}_d${d}" \
        --model-tag="${TAG}" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --save-every-pct=-1 \
        $DEVICE_BATCH_SIZE_ARG \
        2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    # Extract stats from log. base_train prints param counts as one line per key
    # ("total", "transformer_matrices", "lm_head", ...) and the auto-computed
    # total batch size, so derive everything from those.
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"
    NUM_PARAMS=$(grep "^total " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    PARAMS_TRANSFORMER=$(grep "^transformer_matrices " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    PARAMS_LM=$(grep "^lm_head " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    NUM_SCALING_PARAMS=$((PARAMS_TRANSFORMER + PARAMS_LM))
    NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
    BATCH_SIZE=$(grep "Total batch size" "$LOG_FILE" | tail -1 | grep -oP 'Total batch size \K[\d,]+' | tr -d ',')
    TOKENS_TRAINED=$((NUM_ITERS * BATCH_SIZE))
    PARAM_DATA_RATIO=$(python -c "print(f'{$TOKENS_TRAINED / $NUM_SCALING_PARAMS:.2f}')")
    MODEL_DIM=$((d * 64))
    VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
    CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')

    if [ -z "$CORE_SCORE" ]; then
        CORE_SCORE="0.0"
    fi

    log "  d=$d: params=$NUM_PARAMS, scaling=$NUM_SCALING_PARAMS, ratio=$PARAM_DATA_RATIO, bpb=$VAL_BPB, CORE=$CORE_SCORE, time=${TRAIN_TIME}s"

    # Append to CSV
    echo "$d,$MODEL_DIM,$NUM_PARAMS,$NUM_SCALING_PARAMS,$NUM_ITERS,$TOKENS_TRAINED,$PARAM_DATA_RATIO,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
done

log "=============================================="
log "${SERIES_NAME} Miniseries Complete!"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
