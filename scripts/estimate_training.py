"""
Estimate parameter count, training tokens, and training time for a given model depth
and target param-data ratio, using the same calculations as base_train.py.

Usage:
    python -m scripts.estimate_training --depth 12
    python -m scripts.estimate_training --depth 24 --target-param-data-ratio 20
    python -m scripts.estimate_training --depth 12 --num-gpus 8 --mfu 0.50
"""

import math
import argparse
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import get_peak_flops

parser = argparse.ArgumentParser(description="Estimate training parameters, tokens, and time")
parser.add_argument("--depth", type=int, required=True, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--vocab-size", type=int, default=32768, help="vocabulary size")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="data:param ratio (Chinchilla=20)")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = auto-compute optimal)")
parser.add_argument("--num-gpus", type=int, default=1, help="number of GPUs for time estimate")
parser.add_argument("--gpu", type=str, default="H100", help="GPU name for peak FLOPS lookup (e.g. H100, A100)")
parser.add_argument("--mfu", type=float, default=0.45, help="assumed model FLOPS utilization (0-1) for time estimate")
args = parser.parse_args()

# Build model on meta device (no memory allocation)
base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = GPTConfig(
    sequence_len=args.max_seq_len, vocab_size=args.vocab_size,
    n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
)
with torch.device("meta"):
    model = GPT(config)

# Parameter counts (same as base_train.py)
param_counts = model.num_scaling_params()
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()

# Scaling params for token calculation (transformer matrices + lm_head, same as base_train.py)
num_scaling_params = param_counts['transformer_matrices'] + param_counts['lm_head']
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

# Reference model for batch size calculation (same as base_train.py)
def build_model_meta(depth):
    bd = depth * args.aspect_ratio
    md = ((bd + args.head_dim - 1) // args.head_dim) * args.head_dim
    nh = md // args.head_dim
    c = GPTConfig(sequence_len=args.max_seq_len, vocab_size=args.vocab_size,
                  n_layer=depth, n_head=nh, n_kv_head=nh, n_embd=md)
    with torch.device("meta"):
        return GPT(c)

def get_scaling_params(m):
    pc = m.num_scaling_params()
    return pc['transformer_matrices'] + pc['lm_head']

d12_ref = build_model_meta(12)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2**19  # 524,288

# Auto-compute batch size (same as base_train.py)
total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))

# Training iterations and total tokens
num_iterations = target_tokens // total_batch_size
total_tokens = total_batch_size * num_iterations
total_flops = num_flops_per_token * total_tokens

# Estimated training time from FLOPs and assumed MFU
gpu_peak_flops = get_peak_flops(args.gpu)
effective_flops_per_sec = gpu_peak_flops * args.num_gpus * args.mfu
estimated_seconds = total_flops / effective_flops_per_sec
estimated_minutes = estimated_seconds / 60
estimated_hours = estimated_minutes / 60

# Print results
print(f"{'=' * 60}")
print(f"Model: depth={args.depth}, dim={model_dim}, heads={num_heads}")
print(f"{'=' * 60}")
print(f"\nParameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:>14,}")
print(f"\nScaling params (matrices + lm_head): {num_scaling_params:,}")
print(f"FLOPs per token: {num_flops_per_token:,.0f}")
print(f"\nTarget param:data ratio: {args.target_param_data_ratio}")
print(f"Target tokens:           {target_tokens:,}")
print(f"Batch size:              {total_batch_size:,}")
print(f"Num iterations:          {num_iterations:,}")
print(f"Total training tokens:   {total_tokens:,}")
print(f"Tokens:Scaling params:   {total_tokens / num_scaling_params:.2f}")
print(f"Total training FLOPs:    {total_flops:.2e}")
print(f"\nEstimated training time ({args.num_gpus}x {args.gpu} @ {args.mfu*100:.0f}% MFU):")
print(f"  GPU peak FLOPS (bf16): {gpu_peak_flops:.2e}")
print(f"  Effective FLOPS/sec:   {effective_flops_per_sec:.2e}")
if estimated_hours >= 1:
    print(f"  Time: {estimated_hours:.1f} hours ({estimated_minutes:.0f} minutes)")
else:
    print(f"  Time: {estimated_minutes:.1f} minutes")
