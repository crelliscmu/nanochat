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

from ...configuration_utils import PretrainedConfig


def _default_swiglu_intermediate_size(hidden_size: int, multiple_of: int = 128) -> int:
    """SwiGLU hidden_dim = round(8/3 * hidden_size / multiple_of) * multiple_of."""
    return round(8 / 3 * hidden_size / multiple_of) * multiple_of


class NanoChatConfig(PretrainedConfig):
    r"""
    Configuration class for a NanoChat model matching the architecture in `nanochat/gpt.py`:
    rotary position embeddings (with an optional `use_rope` toggle for the DRoPE study),
    QK RMSNorm, untied token/unembedding weights, SwiGLU MLPs, RMSNorm without learnable
    parameters, no bias in linear layers, and optional Grouped Query Attention.

    Args:
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the NanoChat model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*):
            Dimension of the SwiGLU MLP. If `None`, computed as
            `round(8/3 * hidden_size / 128) * 128` to match `nanochat/gpt.py`.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*):
            Number of KV heads for GQA. Defaults to `num_attention_heads`.
        head_dim (`int`, *optional*):
            Explicit per-head dimension. Defaults to `hidden_size // num_attention_heads`.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Non-linearity for the SwiGLU gate branch.
        attention_dropout (`float`, *optional*, defaults to 0.0):
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
        initializer_range (`float`, *optional*, defaults to 0.02):
        rope_parameters (`dict`, *optional*):
            Defaults to `{"rope_type": "default", "rope_theta": 100000.0}`, matching `nanochat/gpt.py`.
        use_cache (`bool`, *optional*, defaults to `True`):
        use_rope (`bool`, *optional*, defaults to `True`):
            Whether to apply rotary position embeddings. Setting to `False` disables RoPE
            entirely at runtime — used to reproduce the DRoPE (disabled-RoPE) experiments.
        final_logit_softcapping (`float`, *optional*, defaults to 15.0):
        attention_bias (`bool`, *optional*, defaults to `False`):
        bos_token_id (`int`, *optional*, defaults to 0):
        eos_token_id (`int`, *optional*, defaults to 1):
        pad_token_id (`int`, *optional*, defaults to 1):
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
    """

    model_type = "nanochat"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_rep",
        "layers.*.self_attn.k_proj": "colwise_rep",
        "layers.*.self_attn.v_proj": "colwise_rep",
        "layers.*.self_attn.o_proj": "rowwise_rep",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 768,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 2048,
        hidden_act: str = "silu",
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        rope_parameters: dict | None = None,
        use_cache: bool = True,
        use_rope: bool = True,
        final_logit_softcapping: float | None = 15.0,
        attention_bias: bool = False,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else _default_swiglu_intermediate_size(hidden_size)
        )
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.use_rope = use_rope
        self.final_logit_softcapping = final_logit_softcapping
        self.attention_bias = attention_bias

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Validate the correctness of rotary position embedding parameters.
        # Must be done after super().__init__() to avoid being overridden by kwargs.
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": 100000.0}
        self.rope_parameters = rope_parameters


__all__ = ["NanoChatConfig"]
