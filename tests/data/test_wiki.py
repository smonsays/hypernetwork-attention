"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest

import jax.numpy as jnp
import tensorflow_datasets as tfds

from hyla.data.wiki import create_wikitext_datasets


class WikiTestCase(unittest.TestCase):
    variant = "wikitext-2-raw-v1"
    batch_size = 64
    seq_len = 512
    vocab_size = 2000

    def test_create_wikitext_datasets(self):
        (ds_train, dss_eval, _), data_info = create_wikitext_datasets(
            variant=self.variant,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            epochs=1,
        )
        tokenizer = data_info["tokenizer"]
        assert self.vocab_size == tokenizer.vocab_size()

        # test_string = "This is a test of the tokenizer, olé."
        # print(tokenizer.detokenize(tokenizer.tokenize(test_string)))

        # unique_tokens = []
        # for d in ds_train:
        #     unique_tokens.append(jnp.unique(d.x))

        # unique_tokens = jnp.unique(jnp.concatenate(unique_tokens))
        # print(len(unique_tokens))

        # seq_lens = []
        # for d in ds_train:
        #     seq_lens.append(jnp.sum(d.x > 2, axis=1))

        for ds in [ds_train, *dss_eval.values()]:
            num_sequences = 0
            for d in ds:
                assert jnp.all(jnp.sum(d.x > 2, axis=1) > 0), "make sure there are no empty sequences (0, 1, 2 are delimiter tokens)"
                assert d.x.shape == (self.batch_size, self.seq_len)
                num_sequences += d.x.shape[0]
            print("num_sequences: ", num_sequences)

        # tfds.benchmark(ds_train)
