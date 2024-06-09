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
import flax.linen as nn
import jax
import jax.numpy as jnp

from hyla.models.hypernetwork import Hypernetwork, VarianceScaledDense, VarianceScaledKernel, VarianceScaledMLP


class HypernetworkTest(unittest.TestCase):
    def test_output_shape(self):
        latent_dim = 3
        input_dim = 7
        output_dim = 5
        batch_dim = 13
        meta_batch_dim = 17

        rng = jax.random.key(0)
        input = jnp.ones((batch_dim, input_dim))
        latent = jnp.ones((meta_batch_dim, latent_dim))

        hnet = Hypernetwork(
            input_shape=input.shape,
            target_network=VarianceScaledMLP(output_dim, 3, 32),
            rng_collection_names=("dropout", ),
            weight_generator=nn.Dense
        )
        init_rngs = {'params': rng, 'dropout': rng, 'target': rng}
        variables = jax.jit(hnet.init)(init_rngs, latent, input)
        output = jax.jit(hnet.apply)(variables, latent, input, rngs={'dropout': rng, 'target': rng})
        chex.assert_shape(output, (meta_batch_dim, batch_dim, output_dim))

    def test_variance_scaled_kernel(self):
        rng = jax.random.PRNGKey(0)
        input_dim = 5
        output_dim = 7
        batch_dim = 11
        input = jnp.ones((batch_dim, input_dim))

        kernel_fn = VarianceScaledKernel(input_dim, output_dim, distribution="truncated_normal")
        variables = kernel_fn.init({"params": rng})
        kernel = kernel_fn.apply(variables)
        chex.assert_shape(kernel, (input_dim, output_dim))

        dense_fn = nn.Dense(output_dim)

        @jax.vmap
        def init_both(rng):
            rng1, rng2 = jax.random.split(rng)

            kernel_fn = VarianceScaledKernel(input_dim, output_dim, distribution="truncated_normal")
            vars = kernel_fn.init({"params": rng1})
            kernel1 = kernel_fn.apply(vars)

            vars = dense_fn.init({"params": rng2}, input)
            kernel2 = vars["params"]["kernel"]

            return kernel1, kernel2

        rngs = jax.random.split(rng, 20480)
        k1, k2 = init_both(rngs)

        chex.assert_trees_all_close(jnp.std(k1), jnp.std(k2), atol=1e-3)

    def test_variance_scaled_dense(self):
        rng = jax.random.PRNGKey(0)
        input_dim = 7
        output_dim = 5
        batch_dim = 3
        input = jnp.ones((batch_dim, input_dim))

        dense_fn = VarianceScaledDense(output_dim)
        variables = dense_fn.init({"params": rng}, input)
        output = dense_fn.apply(variables, input)
        chex.assert_shape(output, (batch_dim, output_dim))


if __name__ == '__main__':
    unittest.main()
