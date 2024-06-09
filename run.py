"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging
import os
import random
import socket
import time
import uuid
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import tensorflow as tf
from absl import app, flags
from ml_collections import config_flags

from callbacks import LogicLatents, RavenLatents
from hyla.data.logic import create_fuzzy_logic_datasets
from hyla.data.raven import create_raven_datasets
from hyla.data.wiki import create_wikitext_datasets
from hyla.experiment import CallbackEvent, Experiment
from hyla.logging import StandardLogger, WandbLogger
from hyla.loss import AutoregressiveCrossEntropy, FuzzyLogicLoss, SymbolicRavenLoss
from hyla.models.transformer import CausalTransformer, TransformerConfig

FLAGS = flags.FLAGS

# Import ml_collections flags, expose jax flags and add additional abseil flags
config_flags.DEFINE_config_file(
    name="config",
    # default="configs/logic.py:logic_4var_2term;transformer",
    # default="configs/wiki.py:wikitext-2-raw-v1;linear_hypatt",
    default="configs/raven.py:raven_3x3_4features;linear_hypatt",
    help_string="Training configuration.",
)
jax.config.parse_flags_with_absl()
# flags.DEFINE_bool("checkpoint", False, "Whether to save the checkpoint of trained models.")
flags.DEFINE_integer("eval_every", default=10000, help="Evaluation frequency.")
flags.DEFINE_integer("log_level", default=1, help="Logging level.")
flags.DEFINE_integer("log_every", default=1000, help="Logging frequency.")
flags.DEFINE_list("logger", default=["standard", "wandb"], help="List of `Logger`s to use")
flags.DEFINE_bool("synchronize", default=True, help="Synchronize logs  to remote.")
flags.DEFINE_string("workdir", default="logs", help="Working directory for saving logs.")


def setup(config, logger=None, callbacks=[]):
    # Instantiate data
    logging.info("Loading dataset...")
    if "logic" in config.name:
        (train_loader, eval_loaders, callback_loaders), info_data = create_fuzzy_logic_datasets(
            batch_size=config.batch_size,
            seed=config.seed,
            **config.data,
        )
        loss_fn = FuzzyLogicLoss
        config.model.relative_position_max_dist = config.data.seq_len
    elif "raven" in config.name:
        (train_loader, eval_loaders, callback_loaders), info_data = create_raven_datasets(
            batch_size=config.batch_size,
            seed=config.seed,
            **config.data,
        )
        loss_fn = partial(
            SymbolicRavenLoss,
            num_features=config.data.num_features,
            num_feature_values=config.data.feature_maxval
        )
        config.model.relative_position_max_dist = config.data.seq_len
        config.model.output_dim = config.data.feature_maxval
    elif "wiki" in config.name:
        (train_loader, eval_loaders, callback_loaders), info_data = create_wikitext_datasets(
            batch_size=config.batch_size, **config.data)
        loss_fn = AutoregressiveCrossEntropy
        config.model.vocab_size = int(info_data["tokenizer"].vocab_size())
        config.model.relative_position_max_dist = config.data.seq_len
    else:
        return ValueError(f"Unknown dataset {config.name}")

    logging.info("...done")

    # Instantiate model
    if config.get("width_multiplier", -1) > 0:
        logging.info("Scaling model width by factor of {}".format(config.width_multiplier))
        config.model.num_heads *= config.width_multiplier
        config.model.emb_dim *= config.width_multiplier
        config.model.qk_dim *= config.width_multiplier
        config.model.v_dim *= config.width_multiplier
        config.model.mlp_dim *= config.width_multiplier

    sequence_model = CausalTransformer(TransformerConfig(**config.model))

    # Instantiate optimizer
    optimizer_ops = []

    if config.grad_clip_norm is not None:
        optimizer_ops.append(optax.clip_by_global_norm(config.grad_clip_norm))

    # Instantiate learning rate scheduler
    if "num_train" in info_data:
        train_steps = info_data["num_train"] // config.batch_size
    else:
        train_steps = len(train_loader)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=train_steps - config.warmup_steps,
        end_value=config.lr * 0.1,
    )

    # import matplotlib.pyplot as plt
    # plt.plot([schedule(i) for i in range(len(train_loader))])
    # plt.show()

    optimizer_ops.append(
        getattr(optax, config.optimizer)(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
            mask=(lambda p: jax.tree_util.tree_map(  # mask weight decay for biases and layernorms
                lambda x: x.ndim != 1, p)) if config.mask_weight_decay else None,
        )
    )
    optimizer = optax.chain(*optimizer_ops)

    # Instantiate experiment runner
    return Experiment(
        config=config,
        model=sequence_model,
        loss=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loaders=eval_loaders,
        callback_loaders=callback_loaders,
        logger=logger,
        callbacks=callbacks,
        log_every=FLAGS.log_every,
        eval_every=FLAGS.eval_every,
        log_level=FLAGS.log_level,
    )


def main(argv):
    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory
    # and make it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')
    logging.info("Running on {}".format(jax.default_backend()))

    # Get config from flags
    config = flags.FLAGS.config
    assert config.name, "No config found."

    if config.seed is None:
        config.seed = random.randint(0, 99999)

    # Setup logging
    if not config.log_dir:
        unique_id = str(uuid.uuid4())[-4:]
        hostname = socket.gethostname()
        datetime = time.strftime("%Y%m%d_%H%M%S_")
        id = datetime + hostname + "_" + unique_id + "_{}".format(config.name)
        config.log_dir = os.path.join(os.getcwd(), FLAGS.workdir, id)

    logging.info("Logging to {}".format(config.log_dir))

    logger = []
    for l in FLAGS.logger:
        if l == "standard":
            # get machine name
            logger.append(StandardLogger(config.log_dir))
        elif l == "wandb":
            logger.append(WandbLogger(config, config.log_dir, FLAGS.synchronize))
        else:
            raise ValueError()

    # Create callbacks
    callbacks = []
    if "logic" in config.name:
        callbacks.extend((
            LogicLatents(log_level=2, onevent=CallbackEvent.START, save_to_disk=True),
            LogicLatents(log_level=2, onevent=CallbackEvent.STEP, save_to_disk=False),
            LogicLatents(log_level=2, onevent=CallbackEvent.END, save_to_disk=True),
        ))
    elif "raven" in config.name:
        callbacks.extend((
            RavenLatents(log_level=2, onevent=CallbackEvent.START, data_config=config.data),
            RavenLatents(log_level=2, onevent=CallbackEvent.END, data_config=config.data),
        ))

    # Setup experiment
    exp = setup(config, logger, callbacks)

    # Reset state of experiment runner
    exp_state = exp.reset(jax.random.PRNGKey(config.seed))

    # Log number of parameters
    logging.info("Running on {}".format(jax.default_backend()))
    logging.info(jtu.tree_map(jnp.shape, exp_state.params["params"]))
    exp.log(0, {
        "num_params": jax.flatten_util.ravel_pytree(exp_state.params["params"])[0].shape[0]
    })

    # Run runner
    logging.info("Start training with parametrization:\n{}".format(config))
    exp.run(exp_state)

    # Save experiment state
    logging.info("Saving experiment state to disk ...")
    exp.save(exp_state)
    logging.info("...done")


if __name__ == "__main__":
    # with jax.disable_jit():
    app.run(main)
