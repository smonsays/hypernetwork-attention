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
    config.variant = name

    if name == "wikitext-103-raw-v1":
        config.seq_len = 128
        config.vocab_size = 6144  # 4829 unique characters
        config.epochs = 5
    elif name == "wikitext-2-raw-v1":
        config.seq_len = 128
        config.vocab_size = 2048
        config.epochs = 1
    else:
        raise NotImplementedError

    return config


def model_config(name: str):
    config = mlc.ConfigDict()

    config.vocab_size = -1
    config.emb_dim = 128
    config.mlp_dim = 256
    config.qk_dim = 128
    config.v_dim = 128
    config.num_heads = 16
    config.num_layers = 8
    config.dropout_rate = 0.0
    config.use_causal_mask = True
    config.use_absolute_position = False
    config.use_relative_position = True
    config.relative_position_max_dist = -1
    config.share_params_over_layers = False
    config.norm = "layer"
    config.block_type = "default"
    config.moe_num = -1
    config.moe_top_k = -1

    if name == "transformer" or name == "softmax_transformer":
        config.sequence_mixer = "softmax_attention"
        config.state_mixer = "swiglu"
    elif name == "linear_transformer":
        config.sequence_mixer = "linear_attention"
        config.state_mixer = "swiglu"
    elif "hypatt" in name:
        config.sequence_mixer = name
        config.state_mixer = "swiglu"
    else:
        raise NotImplementedError

    return config


def get_config(name: str = ""):

    config = mlc.ConfigDict()
    config.name = name
    config.log_dir = ""
    data, model = name.split(";")
    config.data = data_config(data)
    config.model = model_config(model)
    config.model.relative_position_max_dist = config.data.seq_len

    config.batch_size = 128
    config.grad_clip_norm = None
    config.lr = 1e-3
    config.warmup_steps = 1000
    config.weight_decay = 0.1
    config.mask_weight_decay = True
    config.optimizer = "adamw"
    config.seed = 2024

    return config
