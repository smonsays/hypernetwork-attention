"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc
import enum
import os
import pickle
import time
from functools import partial
from typing import Callable, Dict, Tuple, Type

import chex
import flax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import struct

from hyla.data.base import Dataset, Dataloader
from hyla.logging import Logger, StandardLogger

Metrics = Dict[str, chex.Array]


class CallbackEvent(enum.Enum):
    START = enum.auto()
    STEP = enum.auto()
    END = enum.auto()


class Callback(abc.ABC):
    """
    Callbacks are expected to take care of jit-compiling themselves if possible.
    """
    def __init__(self, log_level: int, onevent: CallbackEvent) -> None:
        self.log_level = log_level
        self.onevent = onevent

    @abc.abstractmethod
    def __call__(self, ctx, exp_state) -> Metrics:
        pass


@struct.dataclass
class ExperimentLoss(abc.ABC):
    apply_fn: Callable

    @abc.abstractmethod
    def __call__(
        self, params: Dict, rng: chex.PRNGKey, batch: Dataset, deterministic: bool
    ) -> Tuple[float, Metrics]:
        pass


@struct.dataclass
class ExperimentState:
    optim: optax.OptState
    params: Dict
    rng: chex.PRNGKey
    step: int


class Experiment:
    def __init__(
        self,
        config: Dict,
        model: flax.linen.Module,
        loss: Type[ExperimentLoss],
        optimizer: optax.GradientTransformation,
        train_loader: Dataloader,
        eval_loaders: Dict[str, Dataloader],
        callback_loaders: Dict[str, Dataloader],
        logger: Tuple[Logger] = [StandardLogger()],
        callbacks: Tuple[Callback] = tuple(),
        log_every: int = 1,
        eval_every: int = 1,
        log_level: int = 0,
    ):
        assert eval_every % log_every == 0, "`eval_every` needs to be a multiple of `log_every`"

        self.config = config
        self.model = model
        self.loss_fn = loss(apply_fn=self.model.apply)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loaders = eval_loaders
        self.callback_loaders = callback_loaders
        self.logger = logger
        self.callbacks = callbacks
        self.log_every = log_every
        self.eval_every = eval_every
        self.log_level = log_level

    def trigger_callback(self, exp_state: ExperimentState, onevent: CallbackEvent):
        metrics = dict()
        for c in self.callbacks:
            if c.onevent == onevent and c.log_level <= self.log_level:
                metrics.update(c(exp_state=exp_state, ctx=self))

        return metrics

    def log(self, step: int, log_dict: Dict, prefix: str = ""):
        for l in self.logger:
            l.log(step, {prefix + "_" + k: log_dict[k] for k in log_dict})

    def reset(self, rng: chex.PRNGKey) -> ExperimentState:
        rng_exp, rng_params, rng_dropout = jax.random.split(rng, 3)
        sample_batch = next(iter(self.train_loader))
        init_rngs = {'params': rng_params, 'dropout': rng_dropout}
        params = self.model.init(init_rngs, sample_batch.x, deterministic=True)
        optim = self.optimizer.init(params)

        return ExperimentState(optim=optim, params=params, rng=rng_exp, step=0)

    @staticmethod
    def load(directory: str) -> Tuple[ExperimentState, Dict]:
        config = pickle.load(open(os.path.join(directory, "config.pkl"), "rb"))
        state = pickle.load(open(os.path.join(directory, "state.pkl"), "rb"))
        return config, state

    def save(self, exp_state: ExperimentState):
        pickle.dump(self.config, open(os.path.join(self.config.log_dir, "config.pkl"), "wb"))
        pickle.dump(exp_state, open(os.path.join(self.config.log_dir, "state.pkl"), "wb"))

    def run(self, exp_state: ExperimentState):

        # Trigger callbacks on CallbackEvent.START
        self.log(exp_state.step, self.trigger_callback(exp_state, CallbackEvent.START), "callback")

        prev_time = time.time()
        for step, batch in enumerate(iter(self.train_loader)):
            exp_state, metrics = self.train_step(exp_state, batch)

            if step % self.log_every == 0:
                steps_per_sec = self.log_every / (time.time() - prev_time)
                prev_time = time.time()
                metrics = {**metrics, "steps_per_sec": steps_per_sec}
                self.log(exp_state.step, metrics, prefix="train")

            if step % self.eval_every == 0:
                for name, eval_loader in self.eval_loaders.items():
                    self.log(exp_state.step, self.eval(exp_state, eval_loader), prefix=name)

                # Trigger callbacks on CallbackEvent.STEP
                self.log(exp_state.step, self.trigger_callback(
                    exp_state, CallbackEvent.STEP), "callback")

        for name, eval_loader in self.eval_loaders.items():
            self.log(exp_state.step, self.eval(exp_state, eval_loader), prefix=name)

        # Trigger callbacks on CallbackEvent.END
        self.log(exp_state.step, self.trigger_callback(exp_state, CallbackEvent.END), "callback")

    def eval(self, exp_state: ExperimentState, eval_loader: Dataloader):
        metrics_list, rng = [], exp_state.rng

        for batch in iter(eval_loader):
            rng, rng_test = jax.random.split(rng)
            metrics_list.append(self.eval_step(exp_state, rng_test, batch))

        metrics = jtu.tree_map(lambda *args: jnp.stack((args)), *metrics_list)
        metrics = jtu.tree_map(lambda x: jnp.mean(x, axis=0), metrics)

        return metrics

    @partial(jax.jit, static_argnames="self")
    def eval_step(self, exp_state: ExperimentState, rng: chex.PRNGKey, batch: Dataset) -> Dict:
        return self.loss_fn(exp_state.params, rng, batch, deterministic=True)[1]

    @partial(jax.jit, static_argnames="self")
    def train_step(self, exp_state: ExperimentState, batch: Dataset) -> Tuple[ExperimentState, Dict]:
        rng_grad, rng_new = jax.random.split(exp_state.rng)
        (loss, metrics), grads = jax.value_and_grad(
            self.loss_fn, has_aux=True)(exp_state.params, rng_grad, batch, deterministic=False)

        params_update, optim = self.optimizer.update(grads, exp_state.optim, exp_state.params)
        params = optax.apply_updates(exp_state.params, params_update)

        exp_state = ExperimentState(
            params=params,
            optim=optim,
            rng=rng_new,
            step=exp_state.step + 1,
        )

        metrics.update({
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "param_norm": optax.global_norm(params),
        })

        return exp_state, metrics
