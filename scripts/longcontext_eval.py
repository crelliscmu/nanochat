"""
Long-context evaluation driver, vLLM backend with per-task batching.

Compatible with both base (pre-SFT) and chat (sft / rl) checkpoints via the
--source flag (affects prompt encoding only; vLLM just loads the HF weights).

Supported tasks:
  - LongBench v1 English subsets (16 tasks, see tasks/longbench.py)
  - NoLiMa needle-in-haystack with no literal match (tasks/nolima.py)
  - LongGenBench long structured generation (tasks/longgenbench.py)

The checkpoint must already be converted to HuggingFace format (see
nanochat_pkg/convert_nanochat_checkpoints.py). By default we look under
`{NANOCHAT_BASE_DIR}/.to_vllm/{model_tag}/`; override with --hf-path.

Optional dependencies (install via `pip install -e ".[longcontext]"`):
  - rouge-score   for LongBench gov_report / qmsum / multi_news / samsum
  - fuzzywuzzy    for LongBench lcc / repobench-p

Examples:

    # Single task on the SFT model (chat template, default context window)
    python -m scripts.longcontext_eval -i sft --model-tag d20 -a LongBench-qasper -x 8

    # Same task on the base model (raw text + BOS, no chat template)
    python -m scripts.longcontext_eval -i base --model-tag d20 -a LongBench-qasper -x 8

    # NoLiMa context-length sweep with tensor-parallelism across 4 GPUs
    python -m scripts.longcontext_eval -i sft --model-tag d20 -a NoLiMa \\
        --nolima-lens 1024,2048,4096,8192,16384 --tensor-parallel-size 4

    # All long-context tasks
    python -m scripts.longcontext_eval -i sft --model-tag d20

    # any model: 
    python -m scripts.longcontext_eval -i sft --model-tag llama3-8b \
    --hf-path /path/to/Meta-Llama-3-8B-Instruct -a LongBench-qasper -x 8
"""

import argparse
import copy
import os
from functools import partial

import wandb
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from nanochat.common import get_base_dir, print0, DummyWandb
from nanochat.vllm_nanochat import register as register_nanochat_vllm

from tasks.longbench import LongBench, ENGLISH_SUBSETS
from tasks.nolima import NoLiMa
from tasks.longgenbench import LongGenBench

# -----------------------------------------------------------------------------
# Prompt encoding — branches on --source so the same task data works for base
# and SFT/RL checkpoints. Uses the HF tokenizer that ships with the converted
# checkpoint (bos_token_id + chat_template are embedded in tokenizer_config.json).

def encode_prompt(tokenizer, conversation, source):
    """Return a list of token ids ready for vLLM."""
    if source == "base":
        user_text = conversation["messages"][0]["content"]
        ids = tokenizer.encode(user_text, add_special_tokens=False)
        return [tokenizer.bos_token_id] + ids

    # SFT / RL: pop the trailing empty assistant placeholder and let the chat
    # template add <|assistant_start|> via add_generation_prompt=True.
    conv = copy.deepcopy(conversation)
    msgs = conv["messages"]
    assert msgs and msgs[-1]["role"] == "assistant", \
        "Long-context tasks must end with an empty assistant turn"
    msgs.pop()
    ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
    )
    return list(ids)


def middle_truncate(ids, max_len):
    """Keep the first and last halves, drop the middle. Matches THUDM/LongBench."""
    if len(ids) <= max_len:
        return ids
    half = max_len // 2
    return ids[:half] + ids[-half:]


# -----------------------------------------------------------------------------

def run_longcontext_eval(
    task_object,
    tokenizer,
    llm,
    source,
    max_context_len,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
):
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    prompts = []
    sampling_params_list = []
    conversations = []
    prompt_lengths = []
    for i in range(num_problems):
        conversation = task_object[i]
        ids = encode_prompt(tokenizer, conversation, source)
        ids = middle_truncate(ids, max_context_len)

        per_example_max_new = conversation.get("max_gen_len", max_new_tokens)
        sp = SamplingParams(
            max_tokens=per_example_max_new,
            temperature=temperature,
            top_k=top_k if temperature > 0 else -1,
            n=1,
        )
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        sampling_params_list.append(sp)
        conversations.append(conversation)
        prompt_lengths.append(len(ids))

    outputs = llm.generate(prompts, sampling_params_list)

    score_sum = 0.0
    total = 0
    for conv, out in zip(conversations, outputs):
        completion = out.outputs[0].text
        score = float(task_object.evaluate(conv, completion))
        score_sum += score
        total += 1

    average = score_sum / total if total else 0.0
    print0(f"Final: {score_sum:.4f}/{total} ({average:.4f})")
    return average


# -----------------------------------------------------------------------------
# Task registry.

def _build_task_registry():
    registry = {}
    for subset in ENGLISH_SUBSETS:
        registry[f"LongBench-{subset}"] = partial(LongBench, subset=subset)
    registry["NoLiMa"] = partial(NoLiMa, context_length=4096)
    registry["LongGenBench"] = partial(LongGenBench, split="short")
    return registry


TASK_REGISTRY = _build_task_registry()


# -----------------------------------------------------------------------------

def _resolve_hf_path(args):
    if args.hf_path:
        return args.hf_path
    if args.model_tag is None:
        raise SystemExit("Either --hf-path or --model-tag is required")
    return os.path.join(get_base_dir(), ".to_vllm", args.model_tag)


def main():
    parser = argparse.ArgumentParser(description="Long-context evaluation (vLLM)")
    parser.add_argument("-i", "--source", type=str, required=True, choices=["base", "sft", "rl"],
                        help="Checkpoint family — affects prompt encoding only.")
    parser.add_argument("-a", "--task-name", type=str, default=None,
                        help="Task name(s). Default = all. Use | to split multiple tasks.")
    parser.add_argument("-g", "--model-tag", type=str, default=None,
                        help="Model tag (e.g. d20). Used for logging and default --hf-path.")
    parser.add_argument("--hf-path", type=str, default=None,
                        help="Path to HF-converted checkpoint dir. Default = "
                             "{NANOCHAT_BASE_DIR}/.to_vllm/{model_tag}/")
    parser.add_argument("-x", "--max-problems", type=int, default=None,
                        help="Max problems per task (for smoke tests)")
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=128,
                        help="Default generation budget; tasks may override per-example.")
    parser.add_argument("--max-context-len", type=int, default=None,
                        help="Max prompt length in tokens. Default = max_position_embeddings.")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="vLLM max_model_len. Default = max_context_len + max_new_tokens.")
    parser.add_argument("--nolima-lens", type=str, default="1024,2048,4096,8192,16384",
                        help="Comma-separated context lengths to sweep for NoLiMa.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--enable-prefix-caching", action="store_true",
                        help="Useful for NoLiMa sweeps where the haystack repeats.")
    parser.add_argument("--run", type=str, default="not_dummy",
                        help="wandb run name ('dummy' disables wandb logging)")
    args = parser.parse_args()

    hf_path = _resolve_hf_path(args)
    if not os.path.isdir(hf_path):
        parser.error(f"HF checkpoint dir not found: {hf_path}")

    # Register the nanochat vLLM plugin (NanoChatForCausalLM -> nanochat.vllm_nanochat).
    if "tpp" in hf_path:
        register_nanochat_vllm()
    else:
        print0(f"Skipping nanochat vLLM plugin registration for {hf_path}")

    # Read HF config to determine context budget without loading the model twice.
    import json
    with open(os.path.join(hf_path, "config.json"), "r") as f:
        hf_config = json.load(f)
    sequence_len = int(hf_config["max_position_embeddings"])

    # max_model_len is the total prompt+generation budget. Default to the model's
    # native capacity; reserve `max_new_tokens` of that for generation unless the
    # user pins --max-context-len explicitly.
    max_model_len = args.max_model_len or sequence_len
    if args.max_context_len is not None:
        max_context_len = args.max_context_len
    else:
        max_context_len = max_model_len - args.max_new_tokens
    if max_context_len + args.max_new_tokens > max_model_len:
        parser.error(
            f"max_context_len ({max_context_len}) + max_new_tokens "
            f"({args.max_new_tokens}) = {max_context_len + args.max_new_tokens} "
            f"exceeds max_model_len ({max_model_len}). Lower --max-context-len or "
            f"--max-new-tokens, or raise --max-model-len (and set "
            f"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 if above the model's "
            f"max_position_embeddings={sequence_len})."
        )

    # Decide which tasks to run.
    if args.task_name is None:
        task_names = list(TASK_REGISTRY.keys())
    else:
        task_names = args.task_name.split("|")
        for name in task_names:
            if name not in TASK_REGISTRY:
                parser.error(f"Unknown task {name!r}. Available: {sorted(TASK_REGISTRY)}")

    print0(f"Source       : {args.source}")
    print0(f"Model tag    : {args.model_tag}")
    print0(f"HF path      : {hf_path}")
    print0(f"Context len  : {max_context_len}")
    print0(f"Tasks        : {task_names}")

    llm = LLM(
        model=hf_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
        trust_remote_code=False,
    )
    tokenizer = llm.get_tokenizer()

    model_slug = f"{args.model_tag or 'model'}_vllm"
    use_wandb = args.run != "dummy"
    wandb_run = wandb.init(
        entity="789_project",
        project="long_context_eval",
        name=f"{args.source}/{model_slug}",
        config=vars(args),
        tags=[args.source, "longcontext_eval"],
    ) if use_wandb else DummyWandb()

    results = {}
    for task_name in task_names:
        if task_name == "NoLiMa" and args.nolima_lens:
            lens = [int(x) for x in args.nolima_lens.split(",") if x.strip()]
            for ctx_len in lens:
                print0(f"\n--- {task_name} @ context_length={ctx_len} ---")
                task = NoLiMa(context_length=ctx_len)
                key = f"NoLiMa-{ctx_len}"
                results[key] = run_longcontext_eval(
                    task, tokenizer, llm,
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
                task, tokenizer, llm,
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

    wandb_log = {f"score/{k}": v for k, v in results.items()}
    wandb_run.log(wandb_log)
    wandb_run.finish()

    from nanochat.report import get_report
    get_report(model_tag=args.model_tag).log(
        section=f"Long-context evaluation {args.source} (vLLM)",
        data=[vars(args), results],
    )


if __name__ == "__main__":
    main()
