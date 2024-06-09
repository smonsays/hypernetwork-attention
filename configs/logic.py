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

    if name == "logic_5var_3term":
        config.num_train = 100_000 * 128  # divisible by batch_size
        config.num_test = 16000
        config.num_ood = 16000
        config.num_valid = 16000
        config.num_variables = 5
        config.num_terms = 3  # 2 ** (config.num_variables - 2)
        config.frac_test = 0.5
        config.frac_ood_conj = 0.25
        config.seq_len = 16  # samples of datapoints
    elif name == "logic_4var_2term":
        config.num_train = 50_000 * 128  # divisible by batch_size
        config.num_valid = 16000
        config.num_test = 16000
        config.num_ood = 16000
        config.num_variables = 4
        config.num_terms = 2  # 2 ** (config.num_variables - 2)
        config.frac_test = 0.5
        config.frac_ood_conj = 0.25
        config.seq_len = 16  # samples of datapoints
    elif name == "logic_3var_2term":
        config.num_train = 50_000 * 128  # divisible by batch_size
        config.num_valid = 16000
        config.num_test = 16000
        config.num_ood = 16000
        config.num_variables = 3
        config.num_terms = 2  # 2 ** (config.num_variables - 2)
        config.frac_test = 0.5
        config.frac_ood_conj = 0.25
        config.seq_len = 16  # samples of datapoints
    else:
        raise NotImplementedError

    return config


def model_config(name: str):
    config = mlc.ConfigDict()

    config.output_dim = 1
    config.emb_dim = 128
    config.num_heads = 8
    config.num_layers = 2
    config.qk_dim = 16
    config.v_dim = 16
    config.mlp_dim = 256
    config.dropout_rate = 0.0
    config.use_causal_mask = False
    config.use_absolute_position = False
    config.use_relative_position = True
    config.relative_position_max_dist = 128
    config.share_params_over_layers = False
    config.norm = "layer"
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
    config.model = model_config(model)
    config.data = data_config(data)
    assert config.model.relative_position_max_dist >= config.data.seq_len

    config.batch_size = 128
    config.grad_clip_norm = None
    config.lr = 1e-3
    config.warmup_steps = 100
    config.weight_decay = 1e-1
    config.mask_weight_decay = True
    config.optimizer = "adamw"
    config.seed = 2024

    return config
