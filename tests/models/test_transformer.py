"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest

import chex
import flax
import jax
import jax.numpy as jnp

from hyla.models.transformer import CausalTransformer, TransformerConfig


class TransformerTest(unittest.TestCase):
    def test_output_shape(self):
        input_dim = 7
        output_dim = 5
        batch_dim = 13
        sequence_len = 19
        rng = jax.random.PRNGKey(0)
        input = jnp.ones((batch_dim, sequence_len, input_dim))

        for sequence_mixer in ["linear_attention", "softmax_attention", "linear_hypatt"]:
            for state_mixer in ["gelu_mlp", "swiglu", "moe_swiglu", "moe_gelu_mlp", "hypmoe_mlp", "hypmoe_swiglu"]:
        # for sequence_mixer in ["linear_hypatt"]:
        #     for state_mixer in ["moe_gelu_mlp"]:
                config = TransformerConfig(
                    output_dim=output_dim,
                    sequence_mixer=sequence_mixer,
                    state_mixer=state_mixer,
                    moe_num=3,
                    moe_top_k=3,
                )
                transformer = CausalTransformer(config)
                init_rngs = {'params': rng, 'dropout': rng}
                variables = transformer.init(init_rngs, input, True)
                output = transformer.apply(variables, input, deterministic=False, rngs={'dropout': rng})
                chex.assert_shape(output, (batch_dim, sequence_len, output_dim))

    def test_extract_intermediates(self):
        input_dim = 7
        output_dim = 5
        batch_dim = 13
        sequence_len = 19
        num_layers = 3

        rng = jax.random.PRNGKey(0)
        input = jnp.ones((batch_dim, sequence_len, input_dim))
        config = TransformerConfig(output_dim=output_dim, num_layers=num_layers, norm="rms")
        transformer = CausalTransformer(config)
        init_rngs = {'params': rng, 'dropout': rng}
        variables = transformer.init(init_rngs, input, True)
        output, state = transformer.apply(
            variables, input, deterministic=True, rngs={'dropout': rng}, mutable='intermediates')

        a = [v[0] for k, v in flax.traverse_util.flatten_dict(state).items() if "attn_weights" in k]
        chex.assert_shape(a[0], (batch_dim, config.num_heads, sequence_len, sequence_len))
        assert len(a) == num_layers

        b = [v[0] for k, v in flax.traverse_util.flatten_dict(state).items() if "attn_bias" in k]
        chex.assert_shape(b[0], (1, config.num_heads, sequence_len, sequence_len))
        assert len(b) == num_layers

    def test_lanugage_input(self):
        vocab_size = 10
        batch_dim = 13
        sequence_len = 19

        rng = jax.random.PRNGKey(0)
        input = jnp.ones((batch_dim, sequence_len), dtype=int)
        config = TransformerConfig(vocab_size=vocab_size)
        transformer = CausalTransformer(config)
        init_rngs = {'params': rng, 'dropout': rng}
        variables = transformer.init(init_rngs, input, True)
        output = transformer.apply(variables, input, deterministic=False, rngs={'dropout': rng})
        chex.assert_shape(output, (batch_dim, sequence_len, vocab_size))


if __name__ == '__main__':
    unittest.main()
