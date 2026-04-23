#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Usage:
#   All model downloads below are commented out by default.
#   Uncomment the lines for the models you wish to download, then run:
#       bash download_from_hf.sh
#
#   - hf_base / hf_sft: plain `hf download` into $ARTIFACTS_DIR/{hf_base,hf_sft}/<name>
#   - base_checkpoints / chatsft_checkpoints: use `download_checkpoint` which
#     also extracts the bundled tokenizer/ subfolder into
#     $ARTIFACTS_DIR/tokenizer/<name>/
# ---------------------------------------------------------------------------

export ARTIFACTS_DIR=$HOME/nanochat/nanochat_artifacts
mkdir -p $ARTIFACTS_DIR

# ---------------- hf_base ----------------
# hf download crellis/d18-20tpp-hf-base            --local-dir "$ARTIFACTS_DIR/hf_base/d18_20tpp"
# hf download crellis/d18-20tpp-drope-50-hf-base   --local-dir "$ARTIFACTS_DIR/hf_base/d18_20tpp_drope_50"
# hf download crellis/d18-9tpp-hf-base             --local-dir "$ARTIFACTS_DIR/hf_base/d18_9tpp"
# hf download crellis/d18-9tpp-drope-25-hf-base    --local-dir "$ARTIFACTS_DIR/hf_base/d18_9tpp_drope_25"
# hf download crellis/d18-9tpp-drope-50-hf-base    --local-dir "$ARTIFACTS_DIR/hf_base/d18_9tpp_drope_50"
# hf download crellis/d18-9tpp-drope-75-hf-base    --local-dir "$ARTIFACTS_DIR/hf_base/d18_9tpp_drope_75"
# hf download crellis/d20-20tpp-hf-base            --local-dir "$ARTIFACTS_DIR/hf_base/d20_20tpp"
# hf download crellis/d20-20tpp-drope-50-hf-base   --local-dir "$ARTIFACTS_DIR/hf_base/d20_20tpp_drope_50"
# hf download crellis/d20-40tpp-hf-base            --local-dir "$ARTIFACTS_DIR/hf_base/d20_40tpp"
# hf download crellis/d20-40tpp-drope-50-hf-base   --local-dir "$ARTIFACTS_DIR/hf_base/d20_40tpp_drope_50"
# hf download crellis/d20-9tpp-hf-base             --local-dir "$ARTIFACTS_DIR/hf_base/d20_9tpp"
# hf download crellis/d20-9tpp-drope-25-hf-base    --local-dir "$ARTIFACTS_DIR/hf_base/d20_9tpp_drope_25"
# hf download crellis/d20-9tpp-drope-50-hf-base    --local-dir "$ARTIFACTS_DIR/hf_base/d20_9tpp_drope_50"
# hf download crellis/d20-9tpp-drope-75-hf-base    --local-dir "$ARTIFACTS_DIR/hf_base/d20_9tpp_drope_75"

# ---------------- hf_sft ----------------
# hf download crellis/d18-20tpp-hf-sft                  --local-dir "$ARTIFACTS_DIR/hf_sft/d18_20tpp"
# hf download crellis/d18-20tpp-drope-50-hf-sft         --local-dir "$ARTIFACTS_DIR/hf_sft/d18_20tpp_drope_50"
# hf download crellis/d18-20tpp-drope-50-long-hf-sft    --local-dir "$ARTIFACTS_DIR/hf_sft/d18_20tpp_drope_50_long"
# hf download crellis/d18-20tpp-long-hf-sft             --local-dir "$ARTIFACTS_DIR/hf_sft/d18_20tpp_long"
# hf download crellis/d18-9tpp-hf-sft                   --local-dir "$ARTIFACTS_DIR/hf_sft/d18_9tpp"
# hf download crellis/d18-9tpp-drope-25-hf-sft          --local-dir "$ARTIFACTS_DIR/hf_sft/d18_9tpp_drope_25"
# hf download crellis/d18-9tpp-drope-50-hf-sft          --local-dir "$ARTIFACTS_DIR/hf_sft/d18_9tpp_drope_50"
# hf download crellis/d18-9tpp-drope-75-hf-sft          --local-dir "$ARTIFACTS_DIR/hf_sft/d18_9tpp_drope_75"
# hf download crellis/d20-20tpp-hf-sft                  --local-dir "$ARTIFACTS_DIR/hf_sft/d20_20tpp"
# hf download crellis/d20-20tpp-drope-50-hf-sft         --local-dir "$ARTIFACTS_DIR/hf_sft/d20_20tpp_drope_50"
# hf download crellis/d20-20tpp-drope-50-long-hf-sft    --local-dir "$ARTIFACTS_DIR/hf_sft/d20_20tpp_drope_50_long"
# hf download crellis/d20-20tpp-long-hf-sft             --local-dir "$ARTIFACTS_DIR/hf_sft/d20_20tpp_long"
# hf download crellis/d20-40tpp-hf-sft                  --local-dir "$ARTIFACTS_DIR/hf_sft/d20_40tpp"
# hf download crellis/d20-40tpp-drope-50-hf-sft         --local-dir "$ARTIFACTS_DIR/hf_sft/d20_40tpp_drope_50"
# hf download crellis/d20-40tpp-drope-50-long-hf-sft    --local-dir "$ARTIFACTS_DIR/hf_sft/d20_40tpp_drope_50_long"
# hf download crellis/d20-40tpp-long-hf-sft             --local-dir "$ARTIFACTS_DIR/hf_sft/d20_40tpp_long"
# hf download crellis/d20-9tpp-hf-sft                   --local-dir "$ARTIFACTS_DIR/hf_sft/d20_9tpp"
# hf download crellis/d20-9tpp-drope-25-hf-sft          --local-dir "$ARTIFACTS_DIR/hf_sft/d20_9tpp_drope_25"
# hf download crellis/d20-9tpp-drope-50-hf-sft          --local-dir "$ARTIFACTS_DIR/hf_sft/d20_9tpp_drope_50"
# hf download crellis/d20-9tpp-drope-75-hf-sft          --local-dir "$ARTIFACTS_DIR/hf_sft/d20_9tpp_drope_75"

# Move the tokenizer/ subfolder produced by a checkpoint download into
# $ARTIFACTS_DIR/tokenizer/<name>/. Safe to call repeatedly.
extract_tokenizer() {
    local checkpoint_dir="$1"
    local name="$2"
    local src="$checkpoint_dir/tokenizer"
    local dst="$ARTIFACTS_DIR/tokenizer/$name"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        cp -rn "$src"/. "$dst"/
        rm -rf "$src"
    fi
}

download_checkpoint() {
    local repo="$1"
    local variant="$2"
    local name="$3"
    local dir="$ARTIFACTS_DIR/$variant/$name"
    hf download "$repo" --local-dir "$dir"
    extract_tokenizer "$dir" "$name"
}

# ---------------- base_checkpoints ----------------
# download_checkpoint crellis/d18-20tpp-base_checkpoints            base_checkpoints d18_20tpp
# download_checkpoint crellis/d18-20tpp-drope-50-base_checkpoints   base_checkpoints d18_20tpp_drope_50
# download_checkpoint crellis/d18-9tpp-base_checkpoints             base_checkpoints d18_9tpp
# download_checkpoint crellis/d18-9tpp-drope-25-base_checkpoints    base_checkpoints d18_9tpp_drope_25
# download_checkpoint crellis/d18-9tpp-drope-50-base_checkpoints    base_checkpoints d18_9tpp_drope_50
# download_checkpoint crellis/d18-9tpp-drope-75-base_checkpoints    base_checkpoints d18_9tpp_drope_75
download_checkpoint crellis/d20-20tpp-base_checkpoints            base_checkpoints d20_20tpp
download_checkpoint crellis/d20-20tpp-drope-50-base_checkpoints   base_checkpoints d20_20tpp_drope_50
download_checkpoint crellis/d20-40tpp-base_checkpoints            base_checkpoints d20_40tpp
download_checkpoint crellis/d20-40tpp-drope-50-base_checkpoints   base_checkpoints d20_40tpp_drope_50
# download_checkpoint crellis/d20-9tpp-base_checkpoints             base_checkpoints d20_9tpp
# download_checkpoint crellis/d20-9tpp-drope-25-base_checkpoints    base_checkpoints d20_9tpp_drope_25
# download_checkpoint crellis/d20-9tpp-drope-50-base_checkpoints    base_checkpoints d20_9tpp_drope_50
# download_checkpoint crellis/d20-9tpp-drope-75-base_checkpoints    base_checkpoints d20_9tpp_drope_75

# ---------------- chatsft_checkpoints ----------------
# download_checkpoint crellis/d18-20tpp-chatsft_checkpoints                 chatsft_checkpoints d18_20tpp
# download_checkpoint crellis/d18-20tpp-drope-50-chatsft_checkpoints        chatsft_checkpoints d18_20tpp_drope_50
# download_checkpoint crellis/d18-20tpp-drope-50-long-chatsft_checkpoints   chatsft_checkpoints d18_20tpp_drope_50_long
# download_checkpoint crellis/d18-20tpp-long-chatsft_checkpoints            chatsft_checkpoints d18_20tpp_long
# download_checkpoint crellis/d18-9tpp-chatsft_checkpoints                  chatsft_checkpoints d18_9tpp
# download_checkpoint crellis/d18-9tpp-drope-25-chatsft_checkpoints         chatsft_checkpoints d18_9tpp_drope_25
# download_checkpoint crellis/d18-9tpp-drope-50-chatsft_checkpoints         chatsft_checkpoints d18_9tpp_drope_50
# download_checkpoint crellis/d18-9tpp-drope-75-chatsft_checkpoints         chatsft_checkpoints d18_9tpp_drope_75
# download_checkpoint crellis/d20-20tpp-chatsft_checkpoints                 chatsft_checkpoints d20_20tpp
# download_checkpoint crellis/d20-20tpp-drope-50-chatsft_checkpoints        chatsft_checkpoints d20_20tpp_drope_50
# download_checkpoint crellis/d20-20tpp-drope-50-long-chatsft_checkpoints   chatsft_checkpoints d20_20tpp_drope_50_long
# download_checkpoint crellis/d20-20tpp-long-chatsft_checkpoints            chatsft_checkpoints d20_20tpp_long
# download_checkpoint crellis/d20-40tpp-chatsft_checkpoints                 chatsft_checkpoints d20_40tpp
# download_checkpoint crellis/d20-40tpp-drope-50-chatsft_checkpoints        chatsft_checkpoints d20_40tpp_drope_50
# download_checkpoint crellis/d20-40tpp-drope-50-long-chatsft_checkpoints   chatsft_checkpoints d20_40tpp_drope_50_long
# download_checkpoint crellis/d20-40tpp-long-chatsft_checkpoints            chatsft_checkpoints d20_40tpp_long
# download_checkpoint crellis/d20-9tpp-chatsft_checkpoints                  chatsft_checkpoints d20_9tpp
# download_checkpoint crellis/d20-9tpp-drope-25-chatsft_checkpoints         chatsft_checkpoints d20_9tpp_drope_25
# download_checkpoint crellis/d20-9tpp-drope-50-chatsft_checkpoints         chatsft_checkpoints d20_9tpp_drope_50
# download_checkpoint crellis/d20-9tpp-drope-75-chatsft_checkpoints         chatsft_checkpoints d20_9tpp_drope_75
