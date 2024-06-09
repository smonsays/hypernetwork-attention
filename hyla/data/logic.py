"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging
from functools import partial
from itertools import combinations, product
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from .base import Dataset, DatasetGenerator, SyntheticDataloader


class FuzzyLogicGenerator(DatasetGenerator):
    def __init__(
        self,
        num_variables: int,
        num_terms: int,
        frac_test: float,
        frac_ood_conj: float,
        seed: int,
    ):
        super().__init__(input_shape=(num_variables, ), output_dim=())
        self.num_variables = num_variables
        self.num_terms = num_terms
        self.frac_test = frac_test
        self.frac_ood_conj = frac_ood_conj
        self.fixed_rng = jax.random.PRNGKey(seed)

        # Generate all possible conjunctions and split into in-dist and out-dist
        self.conjunctions_all = jnp.array(list(product([0, 1], repeat=num_variables)))
        self.num_conj = len(self.conjunctions_all)
        self.num_ood = int(self.num_conj * self.frac_ood_conj)

        # Generate in- and out-dist conjunctions
        self.terms_idx_in_dist = jnp.array(
            list(combinations(range(self.num_conj - self.num_ood), num_terms)))
        self.terms_idx_out_dist = jnp.array(
            list(combinations(range(self.num_conj - self.num_ood, self.num_conj), num_terms)))

        # Split in-dist into train and test
        self.num_test = int(len(self.terms_idx_in_dist) * self.frac_test)
        self.terms_idx_in_dist = jax.random.permutation(self.fixed_rng, self.terms_idx_in_dist)
        self.terms_idx_train = self.terms_idx_in_dist[self.num_test:]
        self.terms_idx_test = self.terms_idx_in_dist[:self.num_test]

        logging.info("FuzzyLogicGenerator initialized with:")
        logging.info("{} conjunctions ({} ood-conjunctions)".format(self.num_conj, self.num_ood))
        logging.info("{} in-dist terms ({} train, {} test)".format(
            len(self.terms_idx_in_dist), len(self.terms_idx_train), len(self.terms_idx_test)))
        logging.info("{} out-dist terms".format(len(self.terms_idx_out_dist)))

        assert len(self.terms_idx_train) > 0, "Train set is empty"
        assert len(self.terms_idx_test) > 0, "Test set is empty"
        if self.frac_ood_conj > 0:
            assert self.num_ood >= self.num_terms, "Not enough out-dist conjunctions"
            assert len(self.terms_idx_out_dist) > 0, "OOD set is empty"

    @staticmethod
    @partial(jnp.vectorize, signature="(t,d),(d)->()")
    def apply_fuzzy_logic_multiplication(latent, input):
        out = jnp.where(latent, input, 1.0 - input)  # not
        out = jnp.prod(out, axis=1)  # and
        out = jax.lax.reduce(  # or
            out, 0.0, lambda x, y: 1.0 - ((1.0 - x) * (1.0 - y)), dimensions=(0,))
        return out

    @staticmethod
    @partial(jnp.vectorize, signature="(t,d),(d)->()")
    def apply_fuzzy_logic(latent, input):
        # Zadeh
        out = jnp.where(latent, input, 1.0 - input)  # not
        out = jnp.min(out, axis=1)  # and
        out = jnp.max(out)

        return out

    def sample_latents(self, rng: chex.PRNGKey, num_tasks: int, latent_dist: str):
        """ Sample `num_tasks` random terms and convert into latents """
        if latent_dist == "train":
            terms_idx_all = self.terms_idx_train
        elif latent_dist == "test":
            terms_idx_all = self.terms_idx_test
        elif latent_dist == "ood":
            terms_idx_all = self.terms_idx_out_dist
        elif latent_dist == "ind":
            terms_idx_all = self.terms_idx_in_dist
        elif latent_dist == "ind+ood":
            terms_idx_all = jnp.concatenate(
                (self.terms_idx_in_dist, self.terms_idx_out_dist), axis=0)
        else:
            raise ValueError(f"Invalid latent_dist: {latent_dist}")

        terms_idx = jax.random.choice(rng, terms_idx_all, shape=(num_tasks,), replace=True)
        latents = self.conjunctions_all[terms_idx]

        return latents

    def sample_inputs(self, rng: chex.PRNGKey, num_tasks: int, num_samples, input_dist: str):
        if input_dist == "uniform":
            inputs = jax.random.uniform(rng, shape=(num_tasks, num_samples, self.num_variables))
        elif input_dist == "fixed_context":
            context = jnp.repeat(
                jax.random.uniform(self.fixed_rng, shape=(1, num_samples - 1, self.num_variables)),
                repeats=num_tasks,
                axis=0,
            )
            query = jax.random.uniform(rng, shape=(num_tasks, 1, self.num_variables))
            inputs = jnp.concatenate((context, query), axis=1)
        elif input_dist == "fixed_query":
            context = jax.random.uniform(
                rng, shape=(num_tasks, num_samples - 1, self.num_variables))
            query = jnp.repeat(
                jax.random.uniform(self.fixed_rng, shape=(1, 1, self.num_variables)),
                repeats=num_tasks,
                axis=0,
            )
            inputs = jnp.concatenate((context, query), axis=1)
        else:
            raise ValueError(f"Invalid input dist: {input_dist}")

        return inputs

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "num_tasks",
            "num_samples",
            "latent_dist",
            "input_dist",
        ),
    )
    def sample(
        self,
        rng: chex.PRNGKey,
        num_tasks: int,
        num_samples: int,
        latent_dist: str,
        input_dist: str,
    ):
        rng_tasks, rng_samples = jax.random.split(rng)
        inputs = self.sample_inputs(rng_samples, num_tasks, num_samples, input_dist)
        latents = self.sample_latents(rng_tasks, num_tasks, latent_dist)
        targets = self.apply_fuzzy_logic(jnp.expand_dims(latents, axis=1), inputs)
        targets = jnp.expand_dims(targets, axis=-1)

        return (inputs, targets), latents


def create_fuzzy_logic_datasets(
    batch_size: int,
    seq_len: int,
    num_train: int,
    num_test: int,
    num_ood: int,
    num_valid: int,
    *,
    num_variables: int,
    num_terms: int,
    frac_test: float,
    frac_ood_conj: float,
    seed: int,
) -> Tuple[SyntheticDataloader, SyntheticDataloader, SyntheticDataloader]:
    generator = FuzzyLogicGenerator(num_variables, num_terms, frac_test, frac_ood_conj, seed)

    @partial(jax.jit, static_argnames=("batch_size", "seq_len", "latent_dist", "input_dist"))
    def sample_fn(rng, batch_size, seq_len, latent_dist, input_dist="uniform"):
        (inputs, targets), latents = generator.sample(
            rng, batch_size, seq_len, latent_dist, input_dist)

        x = jnp.concatenate((inputs, targets), axis=-1).at[:, -1, -1].set(0.0)
        y = targets[:, -1]
        base_mse = jnp.mean((targets - jnp.mean(targets, axis=1, keepdims=True)) ** 2, axis=1)

        return Dataset(x, y, info={"base_mse": base_mse, "latents": latents})

    ds_train = SyntheticDataloader(
        num_train, batch_size, seq_len, partial(sample_fn, latent_dist="train"), seed)
    ds_test = SyntheticDataloader(
        num_test, batch_size, seq_len, partial(sample_fn, latent_dist="test"), seed)
    ds_ood = SyntheticDataloader(
        num_ood, batch_size, seq_len, partial(sample_fn, latent_dist="ood"), seed)
    ds_valid = SyntheticDataloader(
        num_valid, batch_size, seq_len, partial(sample_fn, latent_dist="train"), seed)

    s_fn = partial(sample_fn, latent_dist="train", input_dist="fixed_context")
    ds_valid_fixed_context = SyntheticDataloader(num_valid, batch_size, seq_len, s_fn, seed)

    s_fn = partial(sample_fn, latent_dist="train", input_dist="fixed_query")
    ds_valid_fixed_query = SyntheticDataloader(num_valid, batch_size, seq_len, s_fn, seed)

    s_fn = partial(sample_fn, latent_dist="test", input_dist="fixed_query")
    ds_test_fixed_query = SyntheticDataloader(num_valid, batch_size, seq_len, s_fn, seed)

    ds_eval = {
        "test": ds_test,
        "ood": ds_ood,
    }

    ds_callback = {
        "valid": ds_valid,
        "valid_fixed_context": ds_valid_fixed_context,
        "valid_fixed_query": ds_valid_fixed_query,
        "test_fixed_query": ds_test_fixed_query,
    }

    return (ds_train, ds_eval, ds_callback), {}
