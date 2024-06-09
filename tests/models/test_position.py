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
import jax
import jax.numpy as jnp

from hyla.models.position import RelativePositionBiases


class RelativePositionBiasesTest(unittest.TestCase):
    rng = jax.random.PRNGKey(0)

    def test_output_shape(self):
        pos_bias = RelativePositionBiases(
            num_buckets := 8, max_distance := 8, num_heads := 4
        )
        params = pos_bias.init(self.rng, qlen := 16, klen := 16)
        pos_bias_output = pos_bias.apply(params, qlen, klen)
        chex.assert_shape(pos_bias_output, (1, num_heads, qlen, klen))
        assert len(jnp.unique(pos_bias_output)) == ((num_buckets) * num_heads)
