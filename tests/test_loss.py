"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest

import jax
import jax.numpy as jnp
from hyla.data.base import Dataset

from hyla.loss import AutoregressiveCrossEntropy
from hyla.models.transformer import CausalTransformer, TransformerConfig


class LossTest(unittest.TestCase):
    def test_autoregressive_softmax(self):
        rng = jax.random.key(0)
        vocab_size = 13
        batch_size = 5
        seq_len = 3
        # batch = Dataset(x=jnp.ones((batch_size, seq_len), dtype=int))
        batch = Dataset(x=jnp.arange(batch_size * seq_len, dtype=int).reshape(batch_size, seq_len))

        transformer = CausalTransformer(TransformerConfig(vocab_size=vocab_size))
        init_rngs = {'params': rng, 'dropout': rng}
        params = transformer.init(init_rngs, batch.x, True)

        loss_fn = AutoregressiveCrossEntropy(apply_fn=transformer.apply)
        loss, metrics = loss_fn(params, rng, batch, True)


if __name__ == '__main__':
    unittest.main()
