"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct

from hyla.models.attention import MultiHeadDotProductAttention
from hyla.models.feedforward import MlpBlock, SwiGluBlock
from hyla.models.position import RelativePositionBiases


@struct.dataclass
class TransformerConfig:
    num_heads: int = 4
    num_layers: int = 2
    emb_dim: int = 64
    qk_dim: int = 64
    v_dim: int = 64
    mlp_dim: int = 256
    output_dim: int = -1
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    sequence_mixer: str = "softmax_attention"
    state_mixer: str = "gelu_mlp"
    use_causal_mask: bool = True
    use_absolute_position: bool = False
    use_relative_position: bool = True
    relative_position_max_dist: int = 16
    share_params_over_layers: bool = False
    norm: str = "layer"
    normalize_qk: bool = False
    block_type: str = "default"
    vocab_size: Optional[int] = None
    moe_num: Optional[int] = None
    moe_top_k: Optional[int] = None


class AddPositionEncoding(nn.Module):
    config: TransformerConfig

    @staticmethod
    def sinusoidal_init(emb_dim, max_len=2048, min_scale=1.0, max_scale=10000.0):
        """1D Sinusoidal Position Embedding

        Args:
            max_len: maximum possible length for the input.
            min_scale: float: minimum frequency-scale in sine grating.
            max_scale: float: maximum frequency-scale in sine grating.

        Returns:
            output: init function returning `(1, max_len, emb_dim)`
        """
        pe = np.zeros((max_len, emb_dim), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (emb_dim // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, emb_dim // 2) * scale_factor)
        pe[:, : emb_dim // 2] = np.sin(position * div_term)
        pe[:, emb_dim // 2: 2 * (emb_dim // 2)] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, emb_dim]

        return jnp.array(pe)

    @nn.compact
    def __call__(self, inputs):
        assert inputs.ndim == 3
        batch_size, length, emb_dim = inputs.shape
        pos_embedding = self.sinusoidal_init(emb_dim)
        pe = pos_embedding[:, :length, :]

        return inputs + pe


class TransformerBlock(nn.Module):
    """Transformer encoder-decoder layer.

    Args:
        config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    def setup(self):
        config = self.config

        self.dropout = nn.Dropout(rate=config.dropout_rate)

        if config.norm == "layer":
            self.norm1 = nn.LayerNorm()
            self.norm2 = nn.LayerNorm()
        elif config.norm == "rms":
            self.norm1 = nn.RMSNorm()
            self.norm2 = nn.RMSNorm()
        else:
            raise ValueError(f"Unknown norm: {config.norm}")

        if config.use_relative_position:
            self.relative_embedding = RelativePositionBiases(
                num_buckets=config.relative_position_max_dist,
                max_distance=config.relative_position_max_dist,
                num_heads=config.num_heads,
            )

        # Define sequence mixer (attention)
        mha = partial(
            MultiHeadDotProductAttention,
            num_heads=config.num_heads,
            qk_features=config.qk_dim,
            v_features=config.v_dim,
            dropout_rate=config.attention_dropout_rate,
            normalize_qk=config.normalize_qk,
        )
        if config.sequence_mixer == "softmax_attention":
            self.sequence_mixer = mha(attention_norm="softmax", target_network="default")
        elif config.sequence_mixer == "linear_attention":
            self.sequence_mixer = mha(attention_norm="none", target_network="default")
        elif config.sequence_mixer == "linear_attention_rms_head":
            self.sequence_mixer = mha(attention_norm="rms_head", target_network="default")
        elif config.sequence_mixer == "linear_hypatt":
            self.sequence_mixer = mha(attention_norm="rms_head", target_network="mlp_relu")
        elif config.sequence_mixer == "linear_hypatt_mlp_linear":
            self.sequence_mixer = mha(attention_norm="rms_head", target_network="mlp_linear")
        elif config.sequence_mixer == "linear_hypatt_mlp_linear_no_rms_head":
            self.sequence_mixer = mha(attention_norm="none", target_network="mlp_linear")
        elif config.sequence_mixer == "linear_hypatt_no_rms_head":
            self.sequence_mixer = mha(attention_norm="none", target_network="mlp_relu")
        elif config.sequence_mixer == "softmax_hypatt":
            self.sequence_mixer = mha(attention_norm="softmax", target_network="mlp_relu")
        elif config.sequence_mixer == "swiglu_hypatt":
            self.sequence_mixer = mha(attention_norm="rms_head", target_network="swiglu")
        else:
            raise ValueError(f"Unknown sequence_mixer: {config.sequence_mixer}")

        # Define state mixer (MLP)
        if config.state_mixer == "gelu_mlp":
            self.state_mixer = MlpBlock(hidden_dim=config.mlp_dim, dropout_rate=config.dropout_rate)
        elif config.state_mixer == "swiglu":
            self.state_mixer = SwiGluBlock(hidden_dim=config.mlp_dim)
        else:
            raise ValueError(f"Unknown state_mixer: {config.state_mixer}")

    def __call__(self, inputs, deterministic, mask=None):
        assert inputs.ndim == 3

        if self.config.use_relative_position:
            attn_bias = self.relative_embedding(inputs.shape[1], inputs.shape[1])
            self.sow("intermediates", "attn_bias", attn_bias)
        else:
            attn_bias = None

        if self.config.block_type == "default":
            x = self.norm1(inputs)
            x = self.sequence_mixer(x, deterministic=deterministic, mask=mask, attention_bias=attn_bias)
            x = self.dropout(x, deterministic=deterministic)
            x = x + inputs
            y = self.norm2(x)
            y = self.state_mixer(y, deterministic=deterministic)
            out = x + y
        elif self.config.block_type == "reversed":
            x = self.norm1(inputs)
            x = self.state_mixer(x, deterministic=deterministic)
            x = self.dropout(x, deterministic=deterministic)
            x = x + inputs
            y = self.norm2(x)
            y = self.sequence_mixer(y, deterministic=deterministic, mask=mask, attention_bias=attn_bias)
            out = x + y
        elif self.config.block_type == "no_ffn":
            x = self.norm1(inputs)
            x = self.state_mixer(x, deterministic=deterministic)
            x = self.dropout(x, deterministic=deterministic)
            out = x + inputs
        else:
            raise ValueError("Undefined block type {}".format(self.config.block_type))

        return out


class CausalTransformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, deterministic, mask=None):
        config = self.config

        if config.vocab_size:
            assert inputs.ndim == 2  # (batch, len)
            if config.use_causal_mask:
                mask = nn.combine_masks(mask, nn.make_causal_mask(inputs))

            embedding_table = nn.Embed(
                num_embeddings=config.vocab_size,
                features=config.emb_dim,
                name="embedding"
            )
            y = embedding_table(inputs)
        else:
            assert inputs.ndim == 3  # (batch, len, input_dim)
            if config.use_causal_mask:
                mask = nn.combine_masks(mask, nn.make_causal_mask(inputs[:, :, 0]))

            y = nn.Dense(config.emb_dim, name="embedding")(inputs)

        if config.use_absolute_position:
            y = AddPositionEncoding(config=config, name="posembed_input")(y)

        y = nn.Dropout(rate=config.dropout_rate)(y, deterministic=deterministic)

        if config.share_params_over_layers:
            block = TransformerBlock(config=config, name="transformer_block")
            for layer in range(config.num_layers):
                y = block(y, deterministic, mask=mask)
        else:
            for layer in range(config.num_layers):
                y = TransformerBlock(config=config, name=f"transformer_block_{layer}")(
                    y, deterministic, mask=mask)

        y = nn.LayerNorm(name="decoder_norm")(y)

        if config.vocab_size and config.output_dim < 1:
            # Use transpose of embedding matrix for logit transform when output_dim not specified
            logits = embedding_table.attend(y)
            # Normalize pre-softmax logits
            logits = logits / jnp.sqrt(y.shape[-1])
        else:
            logits = nn.Dense(config.output_dim, name="logitdense")(y)

        return logits
