"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Dict

import chex
import jax.numpy as jnp
import optax
from flax import struct
from jax import lax

from hyla.data.base import Dataset
from hyla.experiment import ExperimentLoss


@struct.dataclass
class SymbolicRavenLoss(ExperimentLoss):
    num_features: int
    num_feature_values: int

    def __call__(self, params: Dict, rng: chex.PRNGKey, batch: Dataset, deterministic: bool):

        kwargs = dict(deterministic=deterministic, rngs={"dropout": rng})
        logits = self.apply_fn(params, batch.x, **kwargs)[:, -self.num_features:]

        # treat as multiple classification problem
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.y)
        loss = jnp.mean(loss)

        metrics = {
            "loss": loss,
            "acc": jnp.mean(jnp.equal(jnp.argmax(logits, axis=-1), batch.y).all(axis=1)),
            "acc_per_feature": jnp.mean(jnp.equal(jnp.argmax(logits, axis=-1), batch.y)),
        }

        return loss, metrics


class FuzzyLogicLoss(ExperimentLoss):
    def __call__(self, params: Dict, rng: chex.PRNGKey, batch: Dataset, deterministic: bool):

        kwargs = dict(deterministic=deterministic, rngs={"dropout": rng})
        logits = self.apply_fn(params, batch.x, **kwargs)[:, -1]

        loss = optax.squared_error(predictions=logits, targets=batch.y)
        loss_aggr = jnp.mean(loss)
        r2 = 1 - (loss / batch.info["base_mse"])

        metrics = {
            "loss": loss_aggr,
            "r2": jnp.mean(r2),
        }

        return loss_aggr, metrics


class AutoregressiveCrossEntropy(ExperimentLoss):
    def __call__(self, params: Dict, rng: chex.PRNGKey, batch: Dataset, deterministic: bool):

        def shift_right(x, axis=1):
            """Shift the input to the right by padding and slicing on axis."""
            pad_widths = [(0, 0)] * len(x.shape)
            pad_widths[axis] = (1, 0)
            padded = jnp.pad(x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
            return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)

        inputs = shift_right(batch.x)
        targets = batch.x
        weights = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)
        denominator = jnp.sum(weights)

        logits = self.apply_fn(
            params, inputs, deterministic=deterministic, rngs={"dropout": rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
        loss = jnp.sum(loss * weights) / denominator

        perplexity = jnp.mean(jnp.exp(jnp.sum(loss * weights, axis=1) / jnp.sum(weights, axis=1)))
        acc = jnp.sum(jnp.equal(jnp.argmax(logits, axis=-1), targets) * weights) / denominator

        metrics = {
            "loss": loss,
            "acc": acc,
            "perplexity": perplexity,
        }

        return loss, metrics
