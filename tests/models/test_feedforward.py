"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial
import unittest

import chex
import jax

from hyla.models.feedforward import HyperMixtureOfExpertsMLPBlock, HyperMixtureOfExpertsSwiGluBlock, MixtureOfExpertsBlock, SwiGluBlock


class FeedforwardTest(unittest.TestCase):
    def test_hyper_mixture(self):
        input_dim = 7
        batch_dim = 13
        sequence_len = 19

        rng = jax.random.PRNGKey(0)
        input = jax.random.normal(rng, (batch_dim, sequence_len, input_dim))

        # moe = HyperMixtureOfExpertsMLPBlock(num_experts=3, hidden_dim=128, dropout_rate=0.0)
        moe = HyperMixtureOfExpertsSwiGluBlock(num_experts=3, hidden_dim=128)
        init_rngs = {'params': rng, 'dropout': rng}
        variables = moe.init(init_rngs, input, True)
        output = moe.apply(variables, input, deterministic=False, rngs={'dropout': rng})
        chex.assert_shape(output, (batch_dim, sequence_len, input_dim))

    def test_mixture_of_swiglu(self):
        input_dim = 7
        batch_dim = 13
        sequence_len = 19

        rng = jax.random.PRNGKey(0)
        input = jax.random.normal(rng, (batch_dim, sequence_len, input_dim))

        state_mixer = partial(SwiGluBlock, hidden_dim=23)
        moe = MixtureOfExpertsBlock(num_experts=3, top_k=3, state_mixer=state_mixer)
        init_rngs = {'params': rng, 'dropout': rng}
        variables = moe.init(init_rngs, input, True)
        output = moe.apply(variables, input, deterministic=False, rngs={'dropout': rng})
        chex.assert_shape(output, (batch_dim, sequence_len, input_dim))


if __name__ == '__main__':
    unittest.main()
