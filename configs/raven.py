"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import ml_collections as mlc


def data_config(name: str = ""):
    config = mlc.ConfigDict()

    if name == "raven_3x3_4features":
        config.num_features = 4
        config.feature_maxval = 8
        config.num_train = 20_000_000
        config.num_test = 51200
        config.num_ood = 51200
        config.grid_size = 3
        config.permute_features = True
        config.frac_ood = 0.25
    else:
        raise NotImplementedError

    config.seq_len = ((config.grid_size**2 - 1) * config.num_features) + config.num_features

    return config


def model_config(name: str, output_dim: int):
    config = mlc.ConfigDict()

    config.output_dim = output_dim
    config.emb_dim = 128
    config.num_heads = 16
    config.num_layers = 4
    config.qk_dim = 64
    config.v_dim = 64
    config.mlp_dim = 256
    config.dropout_rate = 0.0
    config.use_causal_mask = False
    config.use_absolute_position = False
    config.use_relative_position = True
    config.relative_position_max_dist = 64
    config.share_params_over_layers = False
    config.norm = "layer"
    config.block_type = "default"
    config.normalize_qk = False
    config.state_mixer = "gelu_mlp"

    if name == "transformer":
        config.sequence_mixer = "softmax_attention"
    elif name == "linear_transformer":
        config.sequence_mixer = "linear_attention"
    else:
        config.sequence_mixer = name

    return config


def get_config(name: str = ""):

    config = mlc.ConfigDict()
    config.name = name
    config.log_dir = ""
    data, model = name.split(";")
    config.data = data_config(data)
    config.model = model_config(model, output_dim=config.data.feature_maxval)
    config.model.relative_position_max_dist = config.data.seq_len

    config.batch_size = 128
    config.grad_clip_norm = None
    config.lr = 1e-3
    config.warmup_steps = 1000
    config.weight_decay = 1e-1
    config.mask_weight_decay = True
    config.optimizer = "adamw"
    config.seed = 2024
    config.width_multiplier = -1

    return config
