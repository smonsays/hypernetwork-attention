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
from itertools import combinations_with_replacement, permutations, product
from typing import Tuple

import chex
import numpy as np
from einops import rearrange
import jax
import jax.numpy as jnp

from .base import Dataset, DatasetGenerator, SyntheticDataloader


class SymbolicRavenGenerator(DatasetGenerator):
    def __init__(
        self,
        num_features: int,
        feature_maxval: int,
        grid_size: int,
        permute_features: bool,
        frac_ood: float,
        seed: int,
    ) -> None:
        super().__init__(input_shape=(num_features,), output_dim=())
        self.num_features = num_features
        self.feature_maxval = feature_maxval
        self.grid_size = grid_size
        self.permute_features = permute_features
        self.frac_ood = frac_ood
        self.fixed_rng = jax.random.key(seed)

        # Generate in- and out-dist latents
        self.latents_all = jnp.array(
            list(combinations_with_replacement(range(self.num_rules), num_features)))
        self.num_latents = len(self.latents_all)
        self.num_ood = int(self.num_latents * self.frac_ood)

        latents_idx_all = jax.random.permutation(self.fixed_rng, jnp.arange(self.num_latents))
        self.latents_idx_in_dist = latents_idx_all[self.num_ood:]
        self.latents_idx_out_dist = latents_idx_all[:self.num_ood]

        logging.info("SymbolicRavenGenerator initialized with:")
        logging.info("{} in-dist latents".format(len(self.latents_idx_in_dist)))
        logging.info("{} out-dist latents".format(len(self.latents_idx_out_dist)))

        if len(jnp.unique(self.latents_all[self.latents_idx_out_dist])) < self.num_rules:
            logging.warning("Not all rules contained in ood set.")

        assert len(self.latents_idx_in_dist) > 0, "In-dist set is empty"
        if self.frac_ood > 0:
            assert len(self.latents_idx_out_dist) > 0, "OOD set is empty"

    @property
    def num_rules(self):
        return len(self.rules)

    @property
    def rules(self):
        """
        Each rule is a function that returns an array of shape (grid_size, grid_size).
        """
        num_examples, seq_len = self.grid_size, self.grid_size
        maxval = self.feature_maxval

        def constant(rng):
            const = jax.random.randint(rng, shape=(num_examples, 1), minval=0, maxval=maxval)
            return jnp.broadcast_to(const, shape=(num_examples, seq_len))

        def progression(rng, inc: int):
            start = jax.random.randint(rng, shape=(num_examples, 1), minval=0, maxval=maxval)
            return (start + inc * jnp.arange(0, seq_len)[jnp.newaxis]) % maxval

        def arithmetic(rng, subtract: bool):
            xs = jax.random.randint(rng, shape=(num_examples, seq_len-1), minval=0, maxval=maxval)
            if subtract:
                xs = xs.at[:, 1:].set(xs[:, 1:] * (-1))

            res = jnp.sum(xs, keepdims=True, axis=1) % maxval
            return jnp.concatenate((jnp.abs(xs), res), axis=-1)

        def distribute_three(rng):
            rng_choice, rng_perm = jax.random.split(rng)
            symbols = jax.random.choice(rng_choice, maxval, shape=(seq_len, ), replace=False)
            symbols = jnp.broadcast_to(symbols, shape=(num_examples, seq_len))
            return jax.random.permutation(rng_perm, symbols, axis=1, independent=True)

        def distractor(rng):
            # NOTE: If this rule is desired, it requires adding a mask to the loss
            #       as this feature is not predictable.
            raise NotImplementedError
            return jax.random.choice(rng, maxval, shape=(num_examples, seq_len), replace=True)

        return (
            # TODO: Additional rules to add could be max, min
            constant,
            partial(progression, inc=1),
            partial(progression, inc=2),
            partial(progression, inc=-1),
            partial(progression, inc=-2),
            partial(arithmetic, subtract=False),
            partial(arithmetic, subtract=True),
            distribute_three,
        )

    @partial(jnp.vectorize, excluded=(0,), signature="(2),()->(n,n)")
    def sample_instance(self, rng: chex.PRNGKey, latent: chex.Array):
        """
        Given a rule specified by latent, sample a valid corresponding instance.
        """
        return jax.lax.switch(latent, self.rules, rng)

    def sample_latents(self, rng: chex.PRNGKey, num_tasks: int, latent_dist: str):
        if latent_dist == "train":
            latents_idx_all = self.latents_idx_in_dist
        elif latent_dist == "test" or latent_dist == "test_unpermuted":
            latents_idx_all = self.latents_idx_in_dist
        elif latent_dist == "ood" or latent_dist == "ood_unpermuted":
            latents_idx_all = self.latents_idx_out_dist
        elif latent_dist == "ind+ood":
            latents_idx_all = jnp.concatenate(
                (self.latents_idx_in_dist, self.latents_idx_out_dist), axis=0)
        else:
            raise ValueError(f"Invalid latent_dist: {latent_dist}")

        latents_idx = jax.random.choice(rng, latents_idx_all, shape=(num_tasks,), replace=True)

        return self.latents_all[latents_idx]

    @partial(jax.jit, static_argnames=("self", "num_tasks", "latent_dist"))
    def sample(self, rng: chex.PRNGKey, num_tasks: int, latent_dist: str):
        rng_tasks, rng_samples, rng_perm = jax.random.split(rng, 3)
        latents = self.sample_latents(rng_tasks, num_tasks, latent_dist)

        # Sample instances for each task and for each feature dimension
        rngs_samples = jax.random.split(rng_samples, latents.size).reshape(*latents.shape, 2)
        instances = self.sample_instance(rngs_samples, latents)  # (tasks, features, grid, grid)
        instances = jnp.moveaxis(instances, 1, 3)  # (tasks, grid, grid, features)

        if self.permute_features and "unpermuted" not in latent_dist:
            # Permute the features along the sequence such that in addition to finding the correct
            # rule, the solution also requires identifying over which features across entities
            # the rule applies
            @partial(jnp.vectorize, signature=("(2),(n,m)->(n,m),(m)"))
            def apply_consistent_permutation(rng, x):
                # x has shape (seq_len, num_features)
                # apply a consistent permutation of the features across seq_len
                perm_idx = jax.random.permutation(rng, jnp.arange(self.num_features))

                return x[:, perm_idx], perm_idx

            # The permutation needs to be consistent across num_examples.
            # We swap axes so we can vmap over seq_len and swap back afterwards.
            instances = jnp.swapaxes(instances, 1, 2)
            rngs_perm = jax.random.split(
                rng_perm, num_tasks * self.grid_size).reshape(num_tasks, self.grid_size, 2)
            instances, perms = apply_consistent_permutation(rngs_perm, instances)
            instances = jnp.swapaxes(instances, 1, 2)
        else:
            perms = None

        return instances, {"latents": latents, "perms": perms}

    def check_is_ambiguous(self, instances):
        """
        Takes a batch of sraven problem instances and checks whether they have ambiguous answers,
        i.e. whether there are multiple sets of rules that fit the query but return different
        answers.
        """
        def is_constant(x):
            split_index = (self.grid_size - 1) * self.grid_size
            first_rows, last_row = x[:split_index], x[split_index:]

            # Check first two rows
            first_rows = rearrange(first_rows, "(n m) -> n m", n=self.grid_size-1, m=self.grid_size)
            first_rows_const = jnp.all(jax.vmap(lambda x: jnp.all(x == x[0]))(first_rows))
            last_row_const = jnp.all(last_row == last_row[0])

            return jnp.logical_and(first_rows_const, last_row_const), last_row[0]

        def is_progression(x):
            diff_mod = (jnp.diff(x) % self.feature_maxval)
            diff_mod_masked = diff_mod[(np.arange(self.grid_size**2 - 2) + 1) % 3 != 0]

            step_size = diff_mod_masked[0]
            same_step_size = jnp.all(diff_mod_masked == step_size)
            step_size_in_range = (
                (step_size == 1) |
                (step_size == 2) |
                (step_size == -1 % self.feature_maxval) |
                (step_size == -2 % self.feature_maxval)
            )
            next_step = (x[-1] + step_size) % self.feature_maxval
            return jnp.logical_and(same_step_size, step_size_in_range), next_step

        def is_sum(x):
            @jax.vmap
            def is_sum_(y):
                return (jnp.sum(y[:2]) % self.feature_maxval) == y[2]

            y = x[:(self.grid_size - 1) * self.grid_size]
            y = rearrange(y, "(n m) -> n m", n=self.grid_size-1, m=self.grid_size)
            return jnp.all(is_sum_(y)), (x[-2] + x[-1]) % self.feature_maxval

        def is_difference(x):
            @jax.vmap
            def is_difference_(y):
                y = y.at[1:-1].set(y[1:-1] * (-1))
                return (jnp.sum(y[:2]) % self.feature_maxval) == y[2]

            y = x[:(self.grid_size - 1) * self.grid_size]
            y = rearrange(y, "(n m) -> n m", n=self.grid_size - 1, m=self.grid_size)
            return jnp.all(is_difference_(y)), (x[-2] - x[-1]) % self.feature_maxval

        def is_distribute_three(x):
            split_index = (self.grid_size - 1) * self.grid_size
            first_rows, last_row = x[:split_index], x[split_index:]

            # Check first two rows
            first_rows = rearrange(first_rows, "(n m) -> n m", n=self.grid_size-1, m=self.grid_size)
            first_rows_sorted = jnp.sort(first_rows, axis=1)
            first_rows_true = jnp.all(first_rows_sorted[0] == first_rows_sorted)

            # Check last row
            possible_vals_sorted = first_rows_sorted[0]
            last_row_matches = last_row[:, jnp.newaxis] == possible_vals_sorted[jnp.newaxis, :]
            last_row_true = jnp.all(jnp.any(last_row_matches, axis=1))
            answer = possible_vals_sorted[jnp.argmin(jnp.sum(last_row_matches, axis=0))]

            return jnp.logical_and(first_rows_true, last_row_true), answer

        @partial(jnp.vectorize, signature="(n)->(r),(r)")
        def check_all_rules(x):
            rule_applies, rule_answer = zip(*[
                    is_constant(x),
                    is_progression(x),
                    is_sum(x),
                    is_difference(x),
                    is_distribute_three(x),
            ])
            return jnp.array(rule_applies), jnp.array(rule_answer)

        def check_all_rules_given_perm(count, perm):
            instances_perm = jnp.vectorize(
                lambda x, p: x[p], signature="(m),(m)->(m)")(instances, perm)
            query = rearrange(instances_perm, "b n s m -> b m (n s)")[:, :, :-1]
            answer = rearrange(instances_perm, "b n s m -> b m (n s)")[:, :, -1]

            rule_applies, rule_answer = check_all_rules(query)
            answer_differs = answer[:, :, jnp.newaxis] != rule_answer
            any_rule_fits_a_feature = jnp.logical_and(rule_applies, answer_differs).any(axis=2)
            all_features_fit_by_rule = jnp.all(any_rule_fits_a_feature, axis=1)

            count += all_features_fit_by_rule

            return count, None

        # scanning over all possible permutations
        # NOTE: This quickly becomes intractable for large `num_features`
        perms = jnp.array(list(
            product(permutations(range(self.num_features)), repeat=self.grid_size)))
        count, _ = jax.lax.scan(check_all_rules_given_perm, jnp.zeros(len(instances)), perms)
        return count


def create_raven_datasets(
    batch_size: int,
    seq_len: int,
    num_train: int,
    num_test: int,
    num_ood: int,
    *,
    num_features: int,
    feature_maxval: int,
    grid_size: int,
    permute_features: bool,
    frac_ood: float,
    seed: int,
) -> Tuple[SyntheticDataloader, SyntheticDataloader, SyntheticDataloader]:

    assert seq_len == ((grid_size**2 - 1) * num_features) + num_features

    generator = SymbolicRavenGenerator(
        num_features, feature_maxval, grid_size, permute_features, frac_ood, seed)

    @partial(jax.jit, static_argnames=("batch_size", "seq_len", "latent_dist"))
    def sample_fn(rng, batch_size, seq_len, latent_dist):
        instances, info = generator.sample(rng, batch_size, latent_dist)

        # instances has shape (batch_size, num_samples, seq_len, features)
        sequence = instances.reshape(batch_size, -1, num_features)
        x, y = sequence[:, :-1, :], sequence[:, -1, :]

        # One-hot encode input features
        x = jax.nn.one_hot(x, feature_maxval)
        x = x.reshape(batch_size, (grid_size**2 - 1) * num_features, feature_maxval)

        # Append empty tokens to the end of the input sequence
        x = jnp.concatenate((x, jnp.zeros((batch_size, num_features, feature_maxval))), axis=1)

        return Dataset(x, y, info)

    ds_train = SyntheticDataloader(
        num_train, batch_size, seq_len, partial(sample_fn, latent_dist="train"), seed)
    ds_test = SyntheticDataloader(
        num_test, batch_size, seq_len, partial(sample_fn, latent_dist="test"), seed)
    ds_ood = SyntheticDataloader(
        num_ood, batch_size, seq_len, partial(sample_fn, latent_dist="ood"), seed)

    ds_ood_unpermuted = SyntheticDataloader(
        num_ood, batch_size, seq_len, partial(sample_fn, latent_dist="ood_unpermuted"), seed)

    ds_test_unpermuted = SyntheticDataloader(
        num_test, batch_size, seq_len, partial(sample_fn, latent_dist="test_unpermuted"), seed)

    ds_eval = {
        "test": ds_test,
        "ood": ds_ood,
    }

    ds_callback = {
        "ood_unpermuted": ds_ood_unpermuted,
        "test_unpermuted": ds_test_unpermuted,
    }

    return (ds_train, ds_eval, ds_callback), {}
