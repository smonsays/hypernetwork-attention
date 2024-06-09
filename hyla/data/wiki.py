"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
from datasets import load_dataset
from sentencepiece import SentencePieceTrainer

from hyla.data.base import Dataloader, Dataset

AUTOTUNE = tf.data.AUTOTUNE


def dump_ds_to_textfile(dataset: tf.data.Dataset, maxchars: int = int(1e7)) -> Tuple[str, int]:
    """
    Write part of a TFDS sentence dataset to a text file.
    """
    char_count = 0
    ds_iter = dataset.as_numpy_iterator()
    with tempfile.NamedTemporaryFile(delete=False, prefix='/tmp/ds_chars') as outfp:
        while char_count < maxchars:
            example = next(ds_iter)
            line = example["text"] + b'\n'
            char_count += len(line)
            outfp.write(line)

    return outfp.name


def load_or_train_tokenizer(
        ds: tf.data.Dataset,
        vocab_size: int,
        vocab_path: str,
        model_type: str = "bpe",
        use_cached_tokenizer: bool = True,
):
    vocab_path = os.path.expanduser(vocab_path)

    if not os.path.exists(vocab_path) or not use_cached_tokenizer:
        # NOTE: Code adapted from flax example lm1b
        # Train a unigram tokenizer using the sentencepiece library
        fname = dump_ds_to_textfile(ds)
        with tempfile.NamedTemporaryFile(delete=False, prefix='/tmp/sp_tmp') as model_fp:
            pass  # just need the tmp-filename

        argstr = ' '.join([
            f'--input={fname}',
            f'--vocab_size={vocab_size}',
            f'--character_coverage={1.0}',
            f'--model_prefix={model_fp.name}',
            f'--model_type={model_type}',
            f'--unk_id={0}',
            f'--bos_id={1}',
            f'--eos_id={2}',
        ])
        SentencePieceTrainer.Train(argstr)
        tf.io.gfile.copy(model_fp.name + '.model', vocab_path, overwrite=True)

    with tf.io.gfile.GFile(vocab_path, 'rb') as model_fp:
        sp_model = model_fp.read()

    tokenizer = tf_text.SentencepieceTokenizer(model=sp_model, add_eos=True)

    return tokenizer


def preprocess_dataset(
    ds: tf.data.Dataset,
    tokenizer,
    *,
    batch_size: int,
    shuffle: bool,
    epochs: Optional[int] = None,
    max_length: int,
    shuffle_buffer_size: int = 1024,
    drop_remainder: bool = True,
    prefetch_size: int = AUTOTUNE,
) -> tf.data.Dataset:
    # Tokenize
    ds = ds.map(lambda d: tokenizer.tokenize(d["text"]), num_parallel_calls=AUTOTUNE)

    # Split sequences that are longer than max_length
    def split_into_chunks(x):
        x = tf.data.Dataset.from_tensor_slices(x)
        return x.batch(max_length, drop_remainder=False)

    ds = ds.flat_map(split_into_chunks)

    # Filter empty sequences, i.e. at least two tokens per sequence
    ds = ds.filter(lambda d: tf.greater(tf.shape(d)[0], 2))

    cardinality = ds.reduce(np.int64(0), lambda x, _: x + 1).numpy()

    # Shuffle, repeat, batch and prefetch
    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size)

    if epochs:
        ds = ds.repeat(epochs)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=[max_length],
        padding_values=0,
        drop_remainder=drop_remainder,
    )

    if prefetch_size:
        ds = ds.prefetch(prefetch_size)

    # Convert into numpy iterator over Dataset tuples
    ds = ds.map(lambda x: Dataset(x=x))
    ds = tfds.as_numpy(ds)

    return ds, cardinality


def create_wikitext_datasets(
    variant: str,
    batch_size: int,
    seq_len: int,
    epochs: int,
    *,
    vocab_size: int,
    offline: bool = False,
) -> Tuple[Dataloader, Dataloader, Dataloader]:

    # Load data
    assert variant in ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]
    data_dir = "~/data/tensorflow/"
    name = "huggingface:wikitext/" + variant

    if offline:
        # Directly load with huggingface to allow offline loading
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        ds_all = load_dataset('wikitext', 'wikitext-103-raw-v1')
        ds_train = ds_all["train"].to_tf_dataset()
        ds_test = ds_all["test"].to_tf_dataset()
        ds_valid = ds_all["validation"].to_tf_dataset()
    else:
        ds_train = tfds.load(name, split='train', data_dir=data_dir)
        ds_test = tfds.load(name, split='test', data_dir=data_dir)
        ds_valid = tfds.load(name, split='validation', data_dir=data_dir)

    # Load or train the sentencepiece tokenizer
    vocab_path = os.path.join(data_dir, name.split(":")[1], "0.0.0/vocab.model")
    tokenizer = load_or_train_tokenizer(ds_train, vocab_size, vocab_path)

    ds_train, num_data_train = preprocess_dataset(
        ds_train,
        tokenizer,
        max_length=seq_len,
        batch_size=batch_size,
        shuffle=True,
        epochs=epochs,
    )

    ds_test, num_data_test = preprocess_dataset(
        ds_test,
        tokenizer,
        max_length=seq_len,
        batch_size=batch_size,
        shuffle=False,
    )

    ds_valid, num_data_valid = preprocess_dataset(
        ds_valid,
        tokenizer,
        max_length=seq_len,
        batch_size=batch_size,
        shuffle=False,
    )

    dss = (ds_train, {"test": ds_test, "valid": ds_valid}, {})
    info = {"tokenizer": tokenizer, "num_train": num_data_train * epochs}

    return dss, info
