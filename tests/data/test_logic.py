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
import plotly.express as px

from hyla.data.logic import FuzzyLogicGenerator, create_fuzzy_logic_datasets


class LogicTestCase(unittest.TestCase):
    rng = jax.random.PRNGKey(0)
    batch_size = 512
    epochs = 2

    def test_plot_fuzzy_logic_fn(self):
        latent = jnp.array([
            [0, 0],
            [0, 1],
            [1, 1],
        ])
        num_terms, num_variables = latent.shape
        inputs = jax.random.uniform(self.rng, shape=(10000, num_variables))
        y = FuzzyLogicGenerator.apply_fuzzy_logic(latent, inputs)
        # y = jnp.digitize(y, bins=jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        fig = px.scatter(x=inputs[:, 0], y=inputs[:, 1], color=y)
        fig.show()
        # fig.write_image("fuzzy.svg")

    def test_FuzzyLogicGenerator_visual(self):
        num_tasks, num_samples = 6, 1280
        flogic = FuzzyLogicGenerator(2, 2, frac_test=0.5, frac_ood_conj=0.0, seed=0)
        (x, y), z = flogic.sample(self.rng, num_tasks, num_samples, "train", "uniform")

        for x_, y_ in zip(x, y):
            fig = px.scatter(x=x_[:, 0], y=x_[:, 1], color=y_[:, 0])
            fig.show()

    def test_FuzzyLogicGenerator(self):
        num_tasks, num_samples = 64, 128
        num_terms, num_variables = 3, 5
        flogic = FuzzyLogicGenerator(
            num_variables, num_terms, frac_test=0.2, frac_ood_conj=0.1, seed=0)

        for latent_dist in ["train", "test", "ood", "ind+ood"]:
            (x, y), z = flogic.sample(self.rng, num_tasks, num_samples, latent_dist, "uniform")
            chex.assert_shape(x, (num_tasks, num_samples, num_variables))
            chex.assert_shape(y, (num_tasks, num_samples, 1))
            chex.assert_shape(z, (num_tasks, num_terms, num_variables))

        for input_dist in ["uniform", "fixed_context", "fixed_query"]:
            (x, y), z = flogic.sample(self.rng, num_tasks, num_samples, "train", input_dist)
            chex.assert_shape(x, (num_tasks, num_samples, num_variables))
            chex.assert_shape(y, (num_tasks, num_samples, 1))
            chex.assert_shape(z, (num_tasks, num_terms, num_variables))

            if input_dist == "fixed_context":
                chex.assert_trees_all_equal(*[xi[:-1] for xi in x])

            if input_dist == "fixed_query":
                chex.assert_trees_all_equal(*[xi[-1] for xi in x])

    def test_create_logic_dataset(self):
        num_terms, num_variables = 2, 3

        with jax.disable_jit(False):
            (ds_train, ds_eval, ds_callback), _ = create_fuzzy_logic_datasets(
                batch_size := 128,
                seq_len := 384,
                num_train=128000,
                num_test=128,
                num_ood=128,
                num_valid=128,
                num_variables=num_variables,
                num_terms=num_terms,
                frac_test=0.1,
                frac_ood_conj=0.25,
                seed=0,
            )
            for ds in [ds_train, *ds_eval.values(), *ds_callback.values()]:
                batch = next(iter(ds))
                chex.assert_shape(batch.x, (batch_size, seq_len, num_variables + 1))
                chex.assert_shape(batch.y, (batch_size, 1))
                chex.assert_shape(batch.info["latents"], (batch_size, num_terms, num_variables))

        # import tensorflow_datasets as tfds
        # tfds.benchmark(ds_train)
