# SPDX-License-Identifier: Apache-2.0
"""Inference-only NanoChat model compatible with HuggingFace weights.

Mirrors the architecture in ``nanochat_pkg/modeling_nanochat.py``:
    - Pre-norm transformer with SwiGLU MLP
    - GQA via ``num_key_value_heads``
    - Per-head QK RMSNorm (weightless, over ``head_dim``)
    - Optional RoPE (``use_rope=False`` for the DRoPE ablation)
    - RoPE -> QK norm ordering (opposite of OLMo2)
    - ``final_logit_softcapping`` applied in the LogitsProcessor
    - Untied embeddings, no biases
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm import ModelRegistry
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors


def _rope_parameters_from_config(config) -> dict:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        return dict(rope_parameters)
    params = {
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "rope_type": "default",
    }
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        params.update(rope_scaling)
    return params


class NanoChatAttention(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = getattr(
            config, "num_key_value_heads", self.total_num_heads
        )
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=bias,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head RMSNorm over head_dim. NanoChatRMSNorm is weightless.
        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, has_weight=False
        )
        self.k_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, has_weight=False
        )

        # DRoPE: when use_rope=False we skip rotary entirely.
        self.use_rope = getattr(config, "use_rope", True)
        if self.use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=_rope_parameters_from_config(config),
            )
        else:
            self.rotary_emb = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn",
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_shape = q.shape
        k_shape = k.shape
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(q_shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(k_shape)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # NanoChat ordering: RoPE first, then QK norm.
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        q, k = self._apply_qk_norm(q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class NanoChatMLP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class NanoChatDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.self_attn = NanoChatAttention(
            vllm_config=vllm_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = NanoChatMLP(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile
class NanoChatModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: NanoChatDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states"], config.hidden_size
            )
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader  # type: ignore[attr-defined]
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Weightless RMSNorms have no parameters to load.
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class NanoChatForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = NanoChatModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        soft_cap = getattr(config, "final_logit_softcapping", None)
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=soft_cap
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head.weight"] if self.config.tie_word_embeddings else None
            ),
        )
        return loader.load_weights(weights)


def register() -> None:
    """vLLM plugin entrypoint — registers NanoChatForCausalLM."""
    ModelRegistry.register_model(
        "NanoChatForCausalLM",
        "nanochat.vllm_nanochat:NanoChatForCausalLM",
    )
