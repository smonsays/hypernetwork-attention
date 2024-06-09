"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math
import unittest

import chex
import jax
import jax.numpy as jnp

from hyla.data.raven import SymbolicRavenGenerator, create_raven_datasets


class RavenTestCase(unittest.TestCase):
    rng = jax.random.PRNGKey(0)

    def test_sample_instance(self):
        with jax.disable_jit(False):
            generator = SymbolicRavenGenerator(
                num_features := 5,
                feature_maxval := 8,
                grid_size := 3,
                permute_features=False,
                frac_ood=0.25,
                seed=0,
            )

            latents = jnp.arange(generator.num_rules)
            rngs = jax.random.split(self.rng, len(latents))
            instances = generator.sample_instance(rngs, latents)

            assert instances.shape == (generator.num_rules, grid_size, grid_size)
            assert jnp.all(instances < feature_maxval)
            assert jnp.all(instances >= 0)

    def test_sample(self):
        with jax.disable_jit(False):
            batch_size = 7
            generator = SymbolicRavenGenerator(
                num_features := 5,
                feature_maxval := 64,
                grid_size := 3,
                permute_features=True,
                frac_ood=0.25,
                seed=0,
            )
            instances, info = generator.sample(self.rng, batch_size, latent_dist="train")
            latents, perms = info["latents"], info["perms"]
            assert instances.shape == (batch_size, grid_size, grid_size, num_features)
            assert latents.shape == (batch_size, num_features)
            assert perms.shape == (batch_size, grid_size, num_features)

    def test_is_ambiguous(self):
        with jax.disable_jit(False):
            batch_size = 2048
            steps = 2
            seed = 2023

            for split in ["train", "test", "ood"]:
                rng = jax.random.PRNGKey(seed)
                generator = SymbolicRavenGenerator(
                    num_features := 4,
                    feature_maxval := 8,
                    grid_size := 3,
                    permute_features=True,
                    frac_ood=0.25,
                    seed=seed,
                )

                def scan_ambiguous(_, r):
                    instances, _ = generator.sample(r, batch_size, latent_dist=split)
                    num_ambiguous = generator.check_is_ambiguous(instances)
                    return None, num_ambiguous

                _, num_ambiguous = jax.lax.scan(scan_ambiguous, None, jax.random.split(rng, steps))

                ambig_mean = jnp.mean(num_ambiguous > 0)
                ambig_sem = jnp.std(num_ambiguous > 0) / math.sqrt(num_ambiguous.size)
                print("\n Fraction of ambiguous instances in {} split: {}±{}".format(
                    split, ambig_mean, ambig_sem))

    def test_create_raven_dataset(self):
        num_features = 8
        feature_maxval = 64
        grid_size = 3
        permute_features = True

        with jax.disable_jit(False):
            (ds_train, ds_eval, ds_callback), _ = create_raven_datasets(
                batch_size := 128,
                seq_len := ((9 - 1) * num_features) + num_features,
                num_train=128000,
                num_test=128,
                num_ood=128,
                num_features=num_features,
                feature_maxval=feature_maxval,
                grid_size=grid_size,
                permute_features=permute_features,
                frac_ood=0.25,
                seed=0,
            )
            for ds in [ds_train, *ds_eval.values(), *ds_callback.values()]:
                batch = next(iter(ds))
                chex.assert_shape(batch.x, (batch_size, seq_len, feature_maxval))
                chex.assert_shape(batch.y, (batch_size, num_features))
                chex.assert_shape(batch.info["latents"], (batch_size, num_features))
                assert jnp.all(batch.x <= 1.0)
                assert jnp.all(batch.x >= 0.0)
                assert jnp.all(batch.y <= feature_maxval)
                assert jnp.all(batch.y >= 0)

        # import tensorflow_datasets as tfds
        # tfds.benchmark(ds_train)
