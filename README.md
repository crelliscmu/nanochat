# nanochat — drope & long-context miniseries

A fork of Andrej Karpathy's [`nanochat`](https://github.com/karpathy/nanochat) used to train and
evaluate a miniseries of small (~360M–480M parameter) decoder-only transformers. The series varies
three axes:

- **depth** — model size (`d18` ≈ 360M, `d20` ≈ 480M)
- **tokens-per-parameter (`tpp`)** — pretraining horizon (9 / 20 / 40)
- **RoPE-removal schedule (`drope`)** — fraction of the pretraining token budget spent with RoPE
  before it is dropped for the remainder, to study positional encoding in long-context
  generalization

A subset of SFT models is additionally fine-tuned on a long-context mixture (`_long` variants).

Pretrained checkpoints (base + SFT, in both nanochat-native and HuggingFace `transformers` format)
are published on the HF Hub under [`crellis/`](https://huggingface.co/crellis).

## Setup

Install dependencies — creates `.venv` if missing, installs `uv` if missing, syncs all deps, and
patches the installed `transformers` package with the NanoChat model:

```bash
bash install.sh
source .venv/bin/activate
```

## Download pretrained models from HuggingFace

To pull the published `base`, `sft`, and raw checkpoints into `nanochat_artifacts/`, run:

```bash
bash dev/download_from_hf.sh
```

## Training pipeline

Each model goes through:

1. **Tokenizer training** — 32,768-vocab BPE trained on ~2B characters of the pretraining dataset.
2. **Pretraining (base)** — Next-token prediction on NVIDIA ClimbMix-400B
   ([`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)).
   Horizon controlled by `target_param_data_ratio` (aka "tpp"). Sequence length 4096, batch size
   1,048,576 tokens, AdamW + Muon optimizer.
3. **Supervised fine-tuning (SFT)** — Instruction tuning on a mixture of
   [`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
   synthetic identity conversations,
   [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) `auxiliary_train`,
   [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), SimpleSpelling, and SpellingBee.
4. **Long-context SFT (`_long` variants)** — Same mixture plus 100K rows of
   [`allenai/tulu-v2-sft-long-mixture`](https://huggingface.co/datasets/allenai/tulu-v2-sft-long-mixture)
   with sequence length extended to 8,192.

## RoPE removal (drope)

Model names containing `drope_XX` follow the recipe from
[*"Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings"*](https://arxiv.org/pdf/2512.12167):
pretrain with RoPE for the first `XX%` of the token budget, remove RoPE, and spend the remaining
`(100 − XX)%` recalibrating the model without positional encodings. This aims to preserve the
optimization benefits of RoPE early in training while yielding a NoPE-style model that generalizes
better to long contexts. Models without `drope` in the name keep RoPE for the full budget
(theta = 100,000).

## Model sizes

| Depth | Layers | Hidden | Heads | Intermediate | Approx params |
|-------|--------|--------|-------|--------------|---------------|
| d18   | 18     | 1152   | 9     | 3072         | ~360M         |
| d20   | 20     | 1280   | 10    | 3456         | ~480M         |

All models use head_dim=128, vocab=32,768, RMSNorm (ε=1e-6), SwiGLU MLP, and final logit softcapping
at 15.0.

## Released checkpoints

| Model tag                     | Depth | tpp  | RoPE schedule    | Long-ctx SFT |
|-------------------------------|-------|------|------------------|--------------|
| d18_9tpp                      | 18    | 9    | always on        | no           |
| d18_9tpp_drope_25             | 18    | 9    | 25% then removed | no           |
| d18_9tpp_drope_50             | 18    | 9    | 50% then removed | no           |
| d18_9tpp_drope_75             | 18    | 9    | 75% then removed | no           |
| d18_20tpp                     | 18    | 20   | always on        | no           |
| d18_20tpp_long                | 18    | 20   | always on        | yes          |
| d18_20tpp_drope_50            | 18    | 20   | 50% then removed | no           |
| d18_20tpp_drope_50_long       | 18    | 20   | 50% then removed | yes          |
| d20_9tpp                      | 20    | 9    | always on        | no           |
| d20_9tpp_drope_25             | 20    | 9    | 25% then removed | no           |
| d20_9tpp_drope_50             | 20    | 9    | 50% then removed | no           |
| d20_9tpp_drope_75             | 20    | 9    | 75% then removed | no           |
| d20_20tpp                     | 20    | 20   | always on        | no           |
| d20_20tpp_long                | 20    | 20   | always on        | yes          |
| d20_20tpp_drope_50            | 20    | 20   | 50% then removed | no           |
| d20_20tpp_drope_50_long       | 20    | 20   | 50% then removed | yes          |
| d20_40tpp                     | 20    | 40   | always on        | no           |
| d20_40tpp_long                | 20    | 40   | always on        | yes          |
| d20_40tpp_drope_50            | 20    | 40   | 50% then removed | no           |
| d20_40tpp_drope_50_long       | 20    | 40   | 50% then removed | yes          |

Total pretraining token budgets:

| Depth | tpp | Total pretraining tokens |
|-------|-----|--------------------------|
| d18   | 9   | ≈ 2.92 B |
| d18   | 20  | ≈ 6.49 B |
| d20   | 9   | ≈ 3.95 B |
| d20   | 20  | ≈ 8.77 B |
| d20   | 40  | ≈ 17.54 B |

`drope` variants use the same total token budget as their non-drope counterpart; the budget is
split between the RoPE-on and RoPE-removed phases as described above.

## Checkpoint format: which repo should I download?

For each model tag there are **four** HuggingFace repositories:

| Repo suffix   | Stage            | Format                                                              | Use case |
|---------------|------------------|---------------------------------------------------------------------|----------|
| `...-base`    | post-pretraining | nanochat native (`model_XXXXXX.pt`, `meta_*.json`, optimizer shard) | continue training / run with the `nanochat` repo |
| `...-sft`     | post-SFT         | nanochat native (`model_XXXXXX.pt`, `meta_*.json`, optimizer shard) | continue training / run with the `nanochat` repo |
| `...-hf-base` | post-pretraining | HuggingFace `transformers`                                          | drop-in `AutoModelForCausalLM` loading |
| `...-hf-sft`  | post-SFT         | HuggingFace `transformers`                                          | drop-in `AutoModelForCausalLM` loading |

Pick `-hf-base` / `-hf-sft` for inference. Pick `-base` / `-sft` only if you plan to continue
training inside the nanochat codebase.

## Inference (HF format, SFT)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "crellis/nanochat-d20-20tpp-hf-sft"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

messages = [{"role": "user", "content": "Why is the sky blue?"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").cuda()
out = model.generate(inputs, max_new_tokens=256)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

`use_rope` in `config.json` reflects the drope setting: `true` for models that kept RoPE for the
entire pretraining budget, `false` for drope variants (rotary embeddings are not applied at
inference time).

Base (pretrained-only) checkpoints are next-token predictors and do not understand the chat
template; use `-hf-base` for completion-style prompting and `-hf-sft` for chat.

## Training compute

All runs were trained on a single H100 GPU via Slurm. Pretraining wall-clock ranges from
~4 hours (d18 @ 9tpp) to ~15 hours (d20 @ 40tpp); SFT adds ~30–90 minutes depending on variant.

## Acknowledgements

- Codebase: [`karpathy/nanochat`](https://github.com/karpathy/nanochat)
- Pretraining data: NVIDIA ClimbMix (via `karpathy/climbmix-400b-shuffle`)
- SFT data: HuggingFaceTB SmolTalk, CAIS MMLU, OpenAI GSM8K, AI2 Tulu-v2 long-mixture
- RoPE-removal recipe: [*Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings*](https://arxiv.org/pdf/2512.12167) (arXiv:2512.12167)

## License

MIT (inherits from the nanochat repository).
