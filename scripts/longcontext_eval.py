"""
Long-context evaluation driver. Compatible with both base (pre-SFT) and
chat (sft / rl) checkpoints via the --source flag.

Supported tasks:
  - LongBench v1 English subsets (16 tasks, see tasks/longbench.py)
  - NoLiMa needle-in-haystack with no literal match (tasks/nolima.py)
  - LongGenBench long structured generation (tasks/longgenbench.py)

Optional dependencies (install via `pip install -e ".[longcontext]"`):
  - rouge-score   for LongBench gov_report / qmsum / multi_news / samsum
  - fuzzywuzzy    for LongBench lcc / repobench-p

Examples:

    # Single task on the SFT model (chat template, default context window)
    python -m scripts.longcontext_eval -i sft --model-tag d12 -a LongBench-qasper -x 8

    # Same task on the base model (raw text + BOS, no chat template)
    python -m scripts.longcontext_eval -i base --model-tag d12 -a LongBench-qasper -x 8

    # NoLiMa context-length sweep, multi-GPU
    torchrun --nproc_per_node=8 -m scripts.longcontext_eval -- \\
        -i sft --model-tag d12 -a NoLiMa --nolima-lens 1024,2048,4096,8192,16384

    # All long-context tasks at the model's training context length
    python -m scripts.longcontext_eval -i sft --model-tag d12
"""

import argparse
import copy
from functools import partial

import torch
import torch.distributed as dist
import wandb

from nanochat.common import (
    compute_init,
    compute_cleanup,
    get_dist_info,
    print0,
    autodetect_device_type,
    DummyWandb,
)
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.longbench import LongBench, ENGLISH_SUBSETS
from tasks.nolima import NoLiMa
from tasks.longgenbench import LongGenBench

# -----------------------------------------------------------------------------
# Prompt encoding — branches on --source so the same task data works for base
# and SFT/RL checkpoints.

def encode_prompt(tokenizer, conversation, source, max_render_tokens):
    """
    Encode a Task conversation into token ids ready for `engine.generate_batch`.

    For source="base":
        Render the user message as raw text with a leading <|bos|>. Base
        checkpoints were never trained with the chat template, so we want them
        to see the prompt as a continuation document.

    For source in ("sft", "rl"):
        Use the chat template path: render <|user_start|>...<|user_end|>
        followed by <|assistant_start|> to prime generation. We inline the
        logic from tokenizer.render_for_completion so we can pass a custom
        max_render_tokens (the default 2048 in render_conversation is way too
        small for long-context evals).
    """
    if source == "base":
        user_text = conversation["messages"][0]["content"]
        return tokenizer(user_text, prepend="<|bos|>")

    # SFT / RL path: pop the trailing assistant placeholder, render the rest,
    # and append <|assistant_start|>.
    conv_no_assistant = copy.deepcopy(conversation)
    msgs = conv_no_assistant["messages"]
    assert msgs and msgs[-1]["role"] == "assistant", \
        "Long-context tasks must end with an empty assistant turn"
    msgs.pop()
    ids, _mask = tokenizer.render_conversation(conv_no_assistant, max_tokens=max_render_tokens)
    ids.append(tokenizer.encode_special("<|assistant_start|>"))
    return ids


def middle_truncate(ids, max_len):
    """
    Keep the first half and last half of `ids`, drop the middle. This matches
    the truncation strategy used by THUDM/LongBench/LongBench/pred.py.
    """
    if len(ids) <= max_len:
        return ids
    half = max_len // 2
    return ids[:half] + ids[-half:]


# -----------------------------------------------------------------------------
# Generative loop. Mirrors run_generative_eval in scripts/chat_eval.py but
# accumulates a float score per example instead of a 0/1 pass count, and reads
# per-example max_gen_len out of the conversation dict.

def run_longcontext_eval(
    task_object,
    tokenizer,
    model,
    engine,
    source,
    max_context_len,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    score_sum = 0.0
    total = 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # Encode + middle-truncate so we always fit in max_context_len.
        ids = encode_prompt(tokenizer, conversation, source, max_render_tokens=max_context_len * 4)
        ids = middle_truncate(ids, max_context_len)

        per_example_max_new = conversation.get("max_gen_len", max_new_tokens)

        results, _ = engine.generate_batch(
            ids,
            num_samples=1,
            max_tokens=per_example_max_new,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_length = len(ids)
        completion = tokenizer.decode(results[0][prefix_length:])
        score = float(task_object.evaluate(conversation, completion))
        score_sum += score
        total += 1

        running = score_sum / total if total else 0.0
        print(
            f"\r\033[KRank {ddp_rank} | {total}/{num_problems // ddp_world_size + 1} "
            f"| running score {running:.4f}",
            end="",
            flush=True,
        )
    print()

    if ddp:
        score_tensor = torch.tensor([score_sum], dtype=torch.float64, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(score_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        score_sum = score_tensor.item()
        total = total_tensor.item()

    average = score_sum / total if total else 0.0
    print0(f"Final: {score_sum:.4f}/{total} ({average:.4f})")
    return average


# -----------------------------------------------------------------------------
# Task registry. Keys here are the strings the user passes to --task-name.

def _build_task_registry():
    registry = {}
    for subset in ENGLISH_SUBSETS:
        registry[f"LongBench-{subset}"] = partial(LongBench, subset=subset)
    registry["NoLiMa"] = partial(NoLiMa, context_length=4096)
    registry["LongGenBench"] = partial(LongGenBench, split="short")
    return registry


TASK_REGISTRY = _build_task_registry()


# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Long-context evaluation")
    parser.add_argument("-i", "--source", type=str, required=True, choices=["base", "sft", "rl"],
                        help="Which checkpoint family to load: base|sft|rl")
    parser.add_argument("-a", "--task-name", type=str, default=None,
                        help="Task name(s). Default = all. Use | to split multiple tasks.")
    parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
    parser.add_argument("-x", "--max-problems", type=int, default=None,
                        help="Max problems per task (for smoke tests)")
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=128,
                        help="Default generation budget; tasks may override per-example.")
    parser.add_argument("--max-context-len", type=int, default=None,
                        help="Override the model's prompt-token budget. Default = "
                             "meta['model_config']['sequence_len']. Capped at the "
                             "RoPE precompute size (model.rotary_seq_len).")
    parser.add_argument("--nolima-lens", type=str, default=None,
                        help="Comma-separated context lengths to sweep for NoLiMa "
                             "(e.g. 1024,2048,4096). Each length is run as a separate "
                             "NoLiMa instance.")
    parser.add_argument("--device-type", type=str, default="",
                        choices=["", "cuda", "cpu", "mps"],
                        help="Device type. empty => autodetect")
    parser.add_argument("--run", type=str, default="dummy",
                        help="wandb run name ('dummy' disables wandb logging)")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(
        args.source, device, phase="eval", model_tag=args.model_tag, step=args.step,
    )
    engine = Engine(model, tokenizer)

    # Resolve max_context_len. Same logic as base_eval.py:215.
    sequence_len = meta["model_config"]["sequence_len"]
    if args.max_context_len is not None:
        rotary_cap = getattr(model, "rotary_seq_len", None)
        if rotary_cap is not None and args.max_context_len > rotary_cap:
            parser.error(
                f"--max-context-len={args.max_context_len} exceeds RoPE cache size {rotary_cap}"
            )
        print0(f"Overriding context length: {sequence_len} -> {args.max_context_len}")
        max_context_len = args.max_context_len
    else:
        max_context_len = sequence_len

    # Decide which tasks to run.
    if args.task_name is None:
        task_names = list(TASK_REGISTRY.keys())
    else:
        task_names = args.task_name.split("|")
        for name in task_names:
            if name not in TASK_REGISTRY:
                parser.error(f"Unknown task {name!r}. Available: {sorted(TASK_REGISTRY)}")

    # wandb logging init (only on rank 0, disabled when --run=dummy)
    master_process = ddp_rank == 0
    model_slug = f"{args.model_tag or 'model'}_step{meta['step']:06d}"
    use_wandb = args.run != "dummy" and master_process
    wandb_run = wandb.init(
        project="nanochat_evals",
        name=f"{args.source}/{model_slug}_longctx",
        config=vars(args),
        tags=[args.source, "longcontext_eval"],
    ) if use_wandb else DummyWandb()

    print0(f"Source       : {args.source}")
    print0(f"Model tag    : {args.model_tag}")
    print0(f"Context len  : {max_context_len}")
    print0(f"Tasks        : {task_names}")

    results = {}
    for task_name in task_names:
        if task_name == "NoLiMa" and args.nolima_lens:
            lens = [int(x) for x in args.nolima_lens.split(",") if x.strip()]
            for ctx_len in lens:
                print0(f"\n--- {task_name} @ context_length={ctx_len} ---")
                task = NoLiMa(context_length=ctx_len)
                key = f"NoLiMa-{ctx_len}"
                results[key] = run_longcontext_eval(
                    task, tokenizer, model, engine,
                    source=args.source,
                    max_context_len=max_context_len,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    max_problems=args.max_problems,
                )
        else:
            print0(f"\n--- {task_name} ---")
            task = TASK_REGISTRY[task_name]()
            results[task_name] = run_longcontext_eval(
                task, tokenizer, model, engine,
                source=args.source,
                max_context_len=max_context_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
            )

    print0("\n=== Long-context results ===")
    for k, v in results.items():
        print0(f"{k:40s} {v:.4f}")

    # Log to wandb
    wandb_log = {}
    for task_name, score in results.items():
        wandb_log[f"score/{task_name}"] = score
    wandb_run.log(wandb_log)
    wandb_run.finish()

    # Log to nanochat report (matches base_eval.py:327 and chat_eval.py:245)
    from nanochat.report import get_report
    get_report(model_tag=args.model_tag).log(
        section=f"Long-context evaluation {args.source}",
        data=[vars(args), results],
    )

    compute_cleanup()


if __name__ == "__main__":
    main()
