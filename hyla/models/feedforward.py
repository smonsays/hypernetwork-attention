"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn


class MlpBlock(nn.Module):
    hidden_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, inputs, deterministic):
        x = nn.Dense(self.hidden_dim)(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(inputs.shape[-1])(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        return x


class SwiGluBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, inputs, deterministic):
        gate = nn.Dense(self.hidden_dim, use_bias=False)(inputs)
        gate = nn.silu(gate)
        x = nn.Dense(self.hidden_dim, use_bias=False)(inputs)
        x = nn.Dense(inputs.shape[-1], use_bias=False)(x * gate)

        return x
