# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#USAGE:
#python -m transformers.models.nanochat.convert_nanochat_checkpoints --input_dir /home/crellis/nano_789/nanochat_artifacts/d20_drope_50/chatsft_checkpoints/d20 --output_dir /home/crellis/nano_789/nanochat_artifacts/d20_drope_50/hf_sft
"""Convert NanoChat (`nanochat/gpt.py`) pretraining checkpoints to the HF NanoChat format.

The upstream model now uses SwiGLU MLPs (`mlp.w_gate` + `mlp.w_up` + `mlp.c_proj`),
rope_theta=100000, and exposes a `use_rope` toggle (DRoPE) in its loop metadata.
This converter handles all three. For backwards compatibility it also accepts the
older relu² MLP (`mlp.c_fc` + `mlp.c_proj`) used by the original karpathy/nanochat-d32
checkpoint and flips `hidden_act` accordingly.
"""

import argparse
import gc
import json
import os
import pickle
from pathlib import Path

import torch

from transformers import AutoTokenizer, NanoChatConfig, NanoChatForCausalLM


SWIGLU_MULTIPLE_OF = 128


# The nanochat tiktoken vocab ends with exactly these 9 special tokens (in order).
# Their IDs depend on the trained vocab size (e.g. 32759-32767 for vocab=32768,
# 65527-65535 for vocab=65536), but the ordering — and the tokens we pick for
# bos/eos/pad — is stable across sizes.
NANOCHAT_SPECIAL_TOKENS = (
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
)
BOS_TOKEN = "<|bos|>"
# EOS/pad depend on which training phase produced the checkpoint:
#   - base models never see chat tokens, so they use <|bos|> (the document boundary
#     used during pretraining) as EOS. Using <|assistant_end|> here makes generate()
#     effectively ignore EOS because the base model never emits that token.
#   - SFT/RL models stop at end-of-assistant, matching the chat template.
# Pad mirrors EOS in both cases to keep attention-mask math simple.
EOS_TOKEN_BY_PHASE = {
    "base": "<|bos|>",
    "sft": "<|assistant_end|>",
    "rl": "<|assistant_end|>",
}
PAD_TOKEN_BY_PHASE = dict(EOS_TOKEN_BY_PHASE)

# Exact chat template requested by the user. Kept as a raw string so Jinja control
# characters survive verbatim into the JSON.
NANOCHAT_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' -%}"
    "{%- set system_message = messages[0]['content'] -%}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}"
    "{%- set system_message = '' -%}"
    "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
    "{{- '<|bos|>' -}}"
    "{%- for message in loop_messages -%}"
    "{%- if message['role'] == 'user' -%}"
    "{%- if loop.first and system_message -%}"
    "{{- '<|user_start|>' + system_message + '\n\n' + message['content'] + '<|user_end|>' -}}"
    "{%- else -%}"
    "{{- '<|user_start|>' + message['content'] + '<|user_end|>' -}}"
    "{%- endif -%}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{- '<|assistant_start|>' + message['content'] + '<|assistant_end|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|assistant_start|>' -}}"
    "{%- endif -%}"
)


def _default_swiglu_intermediate_size(hidden_size: int, multiple_of: int = SWIGLU_MULTIPLE_OF) -> int:
    return round(8 / 3 * hidden_size / multiple_of) * multiple_of


def detect_mlp_flavor(state_dict: dict[str, torch.Tensor]) -> str:
    """Return 'swiglu' if the checkpoint has w_gate/w_up, else 'relu2'."""
    if "transformer.h.0.mlp.w_gate.weight" in state_dict and "transformer.h.0.mlp.w_up.weight" in state_dict:
        return "swiglu"
    if "transformer.h.0.mlp.c_fc.weight" in state_dict:
        return "relu2"
    raise ValueError("Could not detect MLP flavor from checkpoint (no w_gate/w_up or c_fc found)")


def infer_kv_heads(config: NanoChatConfig, state_dict: dict[str, torch.Tensor]) -> int:
    key_weight = state_dict.get("transformer.h.0.attn.c_k.weight")
    if key_weight is None:
        return config.num_key_value_heads
    rows = key_weight.shape[0]
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if rows % head_dim != 0:
        return config.num_key_value_heads
    inferred = rows // head_dim
    print(f"Inferred {inferred} key_value heads from checkpoint")
    return max(inferred, 1)


def infer_intermediate_size(state_dict: dict[str, torch.Tensor], mlp_flavor: str) -> int | None:
    """Read the MLP hidden width straight off the first layer's weights."""
    if mlp_flavor == "swiglu":
        w = state_dict.get("transformer.h.0.mlp.w_gate.weight")
    else:
        w = state_dict.get("transformer.h.0.mlp.c_fc.weight")
    if w is None:
        return None
    return w.shape[0]


def convert_layer(old_prefix: str, new_prefix: str, mlp_flavor: str) -> dict[str, str]:
    mapping = {
        f"{old_prefix}.attn.c_q.weight": f"{new_prefix}.self_attn.q_proj.weight",
        f"{old_prefix}.attn.c_k.weight": f"{new_prefix}.self_attn.k_proj.weight",
        f"{old_prefix}.attn.c_v.weight": f"{new_prefix}.self_attn.v_proj.weight",
        f"{old_prefix}.attn.c_proj.weight": f"{new_prefix}.self_attn.o_proj.weight",
    }
    if mlp_flavor == "swiglu":
        mapping.update(
            {
                f"{old_prefix}.mlp.w_gate.weight": f"{new_prefix}.mlp.gate_proj.weight",
                f"{old_prefix}.mlp.w_up.weight": f"{new_prefix}.mlp.up_proj.weight",
                f"{old_prefix}.mlp.c_proj.weight": f"{new_prefix}.mlp.down_proj.weight",
            }
        )
    else:
        # relu² (legacy) — gate+up collapse into a single fc1 and we use a 2-layer MLP.
        # NOTE: the current HF modeling is SwiGLU-only; legacy checkpoints need the
        # pre-SwiGLU modeling.py to load. Kept here for conversion bookkeeping only.
        mapping.update(
            {
                f"{old_prefix}.mlp.c_fc.weight": f"{new_prefix}.mlp.fc1.weight",
                f"{old_prefix}.mlp.c_proj.weight": f"{new_prefix}.mlp.fc2.weight",
            }
        )
    return mapping


def _resolve_tokenizer_pkl(input_path: Path) -> Path | None:
    """Locate `tokenizer.pkl` for a checkpoint.

    nanochat's `speedrun.sh` splits artifacts across sibling directories:
        <root>/base_checkpoints/<tag>/model_*.pt
        <root>/chatsft_checkpoints/<tag>/model_*.pt
        <root>/tokenizer/<model_tag>/tokenizer.pkl

    So the SFT/RL dirs have no `tokenizer.pkl` of their own — we walk up looking
    for a `tokenizer/*/tokenizer.pkl` under the nearest `<root>`. Returns the
    first match or `None`.
    """
    direct = input_path / "tokenizer.pkl"
    if direct.exists():
        return direct
    for parent in input_path.parents:
        tok_dir = parent / "tokenizer"
        if tok_dir.is_dir():
            matches = sorted(tok_dir.glob("*/tokenizer.pkl"))
            if matches:
                return matches[0]
        if (parent / "base_checkpoints").is_dir() or (parent / "chatsft_checkpoints").is_dir():
            # Stop walking up past the artifact root.
            break
    return None


def extract_special_token_ids(input_path: Path) -> dict[str, int] | None:
    """Read the {content -> id} mapping for nanochat's special tokens from tokenizer.pkl.

    Returns `None` if no pickled tiktoken tokenizer is available; callers then fall
    back to whatever IDs were already on the config. When present, this is how we
    learn that — for example — `<|bos|>` lives at id 32759 in a vocab-32768 model
    and at id 65527 in the vocab-65536 karpathy-d32 model.
    """
    tokenizer_pkl = _resolve_tokenizer_pkl(input_path)
    if tokenizer_pkl is None:
        return None
    try:
        with open(tokenizer_pkl, "rb") as f:
            tok = pickle.load(f)
    except Exception as exc:
        print(f"Warning: could not read tokenizer.pkl to learn special token IDs: {exc}")
        return None
    mapping = getattr(tok, "_special_tokens", None)
    if not mapping:
        return None
    return {content: int(tid) for content, tid in mapping.items()}


PHASE_BY_PARENT_DIR = {
    "base_checkpoints": "base",
    "chatsft_checkpoints": "sft",
    "chatrl_checkpoints": "rl",
}


def detect_phase(input_path: Path) -> str:
    """Return 'base', 'sft', or 'rl' from the checkpoint directory layout.

    nanochat lays out artifacts as `<root>/{base,chatsft,chatrl}_checkpoints/<tag>/`.
    Unrecognized layouts default to 'base' — it's the safer assumption since it
    avoids baking a chat template into a checkpoint that can't honor it.
    """
    return PHASE_BY_PARENT_DIR.get(input_path.parent.name, "base")


def _find_sibling_base_loop_state(input_path: Path) -> dict | None:
    """Look for a base checkpoint meta adjacent to an SFT/RL checkpoint.

    nanochat lays out artifacts as:
        <root>/base_checkpoints/<tag>/meta_*.json      (has loop_state)
        <root>/chatsft_checkpoints/<tag>/meta_*.json   (no loop_state)
        <root>/chatrl_checkpoints/<tag>/meta_*.json    (no loop_state)

    When converting an SFT or RL checkpoint we want to inherit `use_rope` from the
    base meta so the DRoPE toggle survives the handoff.
    """
    if detect_phase(input_path) not in {"sft", "rl"}:
        return None
    base_dir = input_path.parent.parent / "base_checkpoints" / input_path.name
    if not base_dir.is_dir():
        return None
    metas = sorted(base_dir.glob("meta_*.json"))
    if not metas:
        return None
    with open(metas[-1], "r") as f:
        blob = json.load(f)
    return blob.get("loop_state")


def load_config_from_checkpoint(input_path: Path, state_dict: dict[str, torch.Tensor]) -> NanoChatConfig:
    """Load config from meta_*.json (+ optional config.json) in the checkpoint dir.

    For checkpoints produced by `nanochat/gpt.py` the meta file contains `model_config`
    (sequence_len/vocab_size/n_layer/n_head/n_kv_head/n_embd) and `loop_state.use_rope`.
    We also cross-check tensor shapes so a slightly out-of-date meta can't silently
    misbuild the config.
    """
    meta_files = list(input_path.glob("meta_*.json"))
    mlp_flavor = detect_mlp_flavor(state_dict)

    meta_config: dict = {}
    loop_state: dict = {}
    if meta_files:
        # Pick the meta file that matches the latest checkpoint step if possible.
        meta_files.sort()
        meta_file = meta_files[-1]
        print(f"Loading config from {meta_file.name}")
        with open(meta_file, "r") as f:
            meta_blob = json.load(f)
        meta_config = meta_blob.get("model_config", meta_blob)
        loop_state = meta_blob.get("loop_state", {})

    # SFT/RL metas don't carry loop_state — the DRoPE toggle lives only on the base
    # checkpoint that produced them. Fall back to the sibling base checkpoint's meta
    # when we're converting a chat{sft,rl}_checkpoints directory.
    if "use_rope" not in loop_state:
        sibling_loop_state = _find_sibling_base_loop_state(input_path)
        if sibling_loop_state is not None:
            print(f"Inherited loop_state from sibling base checkpoint: use_rope={sibling_loop_state.get('use_rope')}")
            loop_state = sibling_loop_state

    # Defaults aligned with the SwiGLU + drope architecture in `nanochat/gpt.py`.
    hidden_size = meta_config.get("n_embd", 768)
    num_attention_heads = meta_config.get("n_head", 6)
    head_dim = meta_config.get("head_dim", hidden_size // num_attention_heads)

    config_kwargs = {
        "vocab_size": meta_config.get("vocab_size", 32768),
        "hidden_size": hidden_size,
        "num_hidden_layers": meta_config.get("n_layer", 12),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": meta_config.get("n_kv_head"),
        "head_dim": head_dim,
        "max_position_embeddings": 32768,
        "hidden_act": "silu" if mlp_flavor == "swiglu" else "relu2",
        "rope_parameters": {"rope_type": "default", "rope_theta": 100000.0},
        "use_rope": bool(loop_state.get("use_rope", True)),
    }

    inferred_ffn = infer_intermediate_size(state_dict, mlp_flavor)
    if inferred_ffn is not None:
        config_kwargs["intermediate_size"] = inferred_ffn
    elif mlp_flavor == "swiglu":
        config_kwargs["intermediate_size"] = _default_swiglu_intermediate_size(hidden_size)

    # Optional config.json can override a subset of fields (but meta + shape wins for
    # architecture). This lets users pin rope_theta / tokenizer ids etc.
    config_file = input_path / "config.json"
    if config_file.exists():
        print("Loading additional config from config.json")
        with open(config_file, "r") as f:
            extra_config = json.load(f)
        for key in [
            "attention_dropout",
            "rms_norm_eps",
            "initializer_range",
            "final_logit_softcapping",
            "attention_bias",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        ]:
            if key in extra_config:
                config_kwargs[key] = extra_config[key]
            elif key == "attention_bias" and "qkv_bias" in extra_config:
                config_kwargs[key] = extra_config["qkv_bias"]
        if "rope_parameters" in extra_config:
            config_kwargs["rope_parameters"] = extra_config["rope_parameters"]
        elif "rope_scaling" in extra_config and extra_config["rope_scaling"] is not None:
            config_kwargs["rope_parameters"] = extra_config["rope_scaling"]
        elif "rope_theta" in extra_config:
            config_kwargs["rope_parameters"] = {"rope_type": "default", "rope_theta": extra_config["rope_theta"]}

    return NanoChatConfig(**config_kwargs), mlp_flavor


def write_model(input_dir, output_dir, tokenizer_dir=None):
    """Convert NanoChat model from original checkpoint format to HuggingFace format."""
    print("Converting the model.")
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    tokenizer_search_path = Path(tokenizer_dir) if tokenizer_dir else input_path
    phase = detect_phase(input_path)
    print(f"Detected training phase: {phase}")

    # Load checkpoint first — architecture decisions depend on its tensor shapes.
    checkpoint_files = sorted(input_path.glob("model_*.pt"))
    if checkpoint_files:
        checkpoint_path = checkpoint_files[-1]
    else:
        checkpoint_path = input_path / "pytorch_model.bin"

    print(f"Fetching all parameters from the checkpoint at {checkpoint_path}...")
    old_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # torch.compile prepends every key with `_orig_mod.` when the model is wrapped.
    old_state = {k.removeprefix("_orig_mod."): v for k, v in old_state.items()}

    for key in old_state:
        if old_state[key].dtype == torch.float32:
            old_state[key] = old_state[key].to(torch.bfloat16)

    config, mlp_flavor = load_config_from_checkpoint(input_path, old_state)
    print(
        f"Loaded config hidden_size={config.hidden_size} num_layers={config.num_hidden_layers} "
        f"intermediate_size={config.intermediate_size} mlp={mlp_flavor} use_rope={config.use_rope}"
    )

    # Cross-check KV heads from the actual tensor and keep GQA consistent.
    inferred_kv = infer_kv_heads(config, old_state)
    config.num_key_value_heads = inferred_kv
    if config.num_attention_heads % config.num_key_value_heads != 0:
        print(f"Adjusting num_attention_heads from {config.num_attention_heads} to {config.num_key_value_heads}")
        config.num_attention_heads = config.num_key_value_heads

    # gpt.py pads vocab to a multiple of 64 in wte/lm_head; prefer the on-disk size.
    wte = old_state.get("transformer.wte.weight")
    if wte is not None and wte.shape[0] != config.vocab_size:
        print(f"Using padded vocab_size from checkpoint: {config.vocab_size} -> {wte.shape[0]}")
        config.vocab_size = wte.shape[0]

    print("Converting model...")
    state_dict: dict[str, torch.Tensor] = {}
    rename_map: dict[str, str] = {}

    def assign(old_key: str, new_key: str) -> None:
        tensor = old_state.get(old_key)
        if tensor is None:
            return
        state_dict[new_key] = tensor.clone()
        rename_map[old_key] = new_key

    assign("transformer.wte.weight", "model.embed_tokens.weight")
    assign("lm_head.weight", "lm_head.weight")

    for layer_idx in range(config.num_hidden_layers):
        old_prefix = f"transformer.h.{layer_idx}"
        new_prefix = f"model.layers.{layer_idx}"
        for old_key, new_key in convert_layer(old_prefix, new_prefix, mlp_flavor).items():
            assign(old_key, new_key)

    missing = [key for key in old_state.keys() if key not in rename_map]
    if missing:
        print(f"Skipped {len(missing)} legacy entries that have no equivalent in the shared implementation")
        for key in missing:
            print(f"  - {key}")

    del old_state
    gc.collect()

    config.torch_dtype = torch.bfloat16
    config.tie_word_embeddings = False

    # Pull real special-token IDs off the source tiktoken so the saved config.json
    # and generation_config.json point at <|bos|> / <|assistant_end|> instead of the
    # NanoChatConfig defaults (0 / 1 / 1).
    special_ids = extract_special_token_ids(tokenizer_search_path)
    if special_ids is not None:
        eos_token = EOS_TOKEN_BY_PHASE[phase]
        pad_token = PAD_TOKEN_BY_PHASE[phase]
        try:
            config.bos_token_id = special_ids[BOS_TOKEN]
            config.eos_token_id = special_ids[eos_token]
            config.pad_token_id = special_ids[pad_token]
            print(
                f"Special token IDs ({phase}): bos={config.bos_token_id} "
                f"eos={config.eos_token_id} ({eos_token}) "
                f"pad={config.pad_token_id} ({pad_token})"
            )
        except KeyError as exc:
            print(f"Warning: expected special token {exc} not found in tokenizer.pkl; leaving config IDs as-is")

    print("Loading the checkpoint in a NanoChat model.")
    with torch.device("meta"):
        model = NanoChatForCausalLM(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")

    if hasattr(model.config, "_name_or_path"):
        del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(output_dir)
    del state_dict, model

    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    NanoChatForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def _write_tokenizer_config(
    output_dir: Path, special_ids: dict[str, int], model_max_length: int, phase: str
) -> None:
    """Write the `tokenizer_config.json` alongside the converted tokenizer.

    The file follows the schema requested for nanochat: every special token is
    listed in `added_tokens_decoder`, bos/eos/pad tokens resolve to the real
    trained IDs. For SFT/RL the chat template is baked in so `apply_chat_template`
    works out of the box; for base models the template is omitted since the base
    model never saw the <|user_start|>/<|assistant_start|> convention during
    training and advertising chat support would produce nonsense.
    """
    added_tokens_decoder = {}
    for content in NANOCHAT_SPECIAL_TOKENS:
        tid = special_ids.get(content)
        if tid is None:
            print(f"Warning: special token {content!r} missing from tokenizer; skipping added_tokens_decoder entry")
            continue
        added_tokens_decoder[str(tid)] = {
            "content": content,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }

    eos_token = EOS_TOKEN_BY_PHASE[phase]
    pad_token = PAD_TOKEN_BY_PHASE[phase]
    tok_cfg = {
        "added_tokens_decoder": added_tokens_decoder,
        "bos_token": BOS_TOKEN,
        "eos_token": eos_token,
        "pad_token": pad_token,
        "clean_up_tokenization_spaces": False,
        "model_max_length": model_max_length,
        "model_input_names": ["input_ids", "attention_mask"],
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    if phase != "base":
        tok_cfg["chat_template"] = NANOCHAT_CHAT_TEMPLATE

    path = Path(output_dir) / "tokenizer_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
    chat_suffix = "" if phase != "base" else " (no chat_template; base model)"
    print(
        f"Wrote tokenizer_config.json ({phase}) with bos={tok_cfg['bos_token']} "
        f"eos={tok_cfg['eos_token']} pad={tok_cfg['pad_token']}{chat_suffix}"
    )


def _read_model_max_length(output_dir: Path) -> int:
    """Prefer the converted model's max_position_embeddings; fall back to 2048."""
    cfg_path = Path(output_dir) / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            return int(cfg.get("max_position_embeddings", 2048))
        except Exception:
            pass
    return 2048


def write_tokenizer(input_dir, output_dir, tokenizer_dir=None):
    """Convert and save the tokenizer + tokenizer_config.json.

    If `tokenizer_dir` is given, we look for `tokenizer.pkl` there; otherwise we
    try the checkpoint dir itself and then walk up to find a sibling
    `tokenizer/*/tokenizer.pkl` — the layout nanochat artifacts ship in.
    """
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    search_path = Path(tokenizer_dir) if tokenizer_dir else input_path
    phase = detect_phase(input_path)

    tokenizer_pkl = _resolve_tokenizer_pkl(search_path)
    if tokenizer_pkl is not None:
        print(f"Using tokenizer.pkl at {tokenizer_pkl}")
        try:
            from transformers.integrations.tiktoken import convert_tiktoken_to_fast

            with open(tokenizer_pkl, "rb") as f:
                tok_pkl = pickle.load(f)
            convert_tiktoken_to_fast(tok_pkl, output_dir)
            print("Converted tokenizer.pkl to HuggingFace format")
        except Exception as e:
            print(f"Warning: Failed to convert tokenizer.pkl: {e}")
            for filename in ("tokenizer.json", "tokenizer_config.json"):
                src = tokenizer_pkl.parent / filename
                if src.exists():
                    (out_path / filename).write_bytes(src.read_bytes())
    else:
        for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = search_path / filename
            if src.exists():
                (out_path / filename).write_bytes(src.read_bytes())

    # Emit our nanochat tokenizer_config.json using the real trained IDs — this
    # overwrites anything convert_tiktoken_to_fast may have produced because
    # downstream tooling (chat_template, pad/eos IDs) depends on our exact schema.
    special_ids = extract_special_token_ids(search_path)
    if special_ids:
        model_max_length = _read_model_max_length(out_path)
        _write_tokenizer_config(out_path, special_ids, model_max_length, phase=phase)
    else:
        print("Skipping tokenizer_config.json (no tokenizer.pkl available to resolve special token IDs)")

    print("Tokenizer saved successfully.")


def run_test(output_dir: str, prompt: str, max_new_tokens: int = 64) -> None:
    """Run a quick generation test to verify the converted model works correctly."""
    print(f"Running quick generation test with prompt: {prompt}")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = NanoChatForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(f"Generated text: {generated}")


def main():
    parser = argparse.ArgumentParser(description="Convert NanoChat checkpoints to HuggingFace format")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the original checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help=(
            "Directory containing `tokenizer.pkl` if it is not alongside the checkpoint. "
            "SFT and RL dirs in nanochat_artifacts do not ship the tokenizer — point this at "
            "e.g. `<root>/<tag>/tokenizer/<model_tag>/`. If omitted, auto-discovered by walking up."
        ),
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default=None,
        help="Optional prompt for a quick generation test",
    )
    args = parser.parse_args()

    write_model(args.input_dir, args.output_dir, tokenizer_dir=args.tokenizer_dir)
    write_tokenizer(args.input_dir, args.output_dir, tokenizer_dir=args.tokenizer_dir)

    if args.test_prompt:
        run_test(args.output_dir, args.test_prompt)


if __name__ == "__main__":
    main()
