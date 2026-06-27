"""
Prepare training data for volunteer-computed nanoGPT.

Downloads the tiny shakespeare dataset, tokenizes at the character level,
and writes train.bin / val.bin / meta.pkl for the coordinator to serve.

Usage:
    python -m nanogpt_volunteer.prepare_data
    python -m nanogpt_volunteer.prepare_data --data_dir data/my_dataset --input_file my_corpus.txt
"""

import os
import pickle
import argparse

import numpy as np

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def download_shakespeare(path):
    import urllib.request
    print(f"Downloading tiny shakespeare to {path}...")
    urllib.request.urlretrieve(DATA_URL, path)


def prepare(data_dir, input_file=None):
    os.makedirs(data_dir, exist_ok=True)

    if input_file and os.path.exists(input_file):
        txt_path = input_file
    else:
        txt_path = os.path.join(data_dir, 'input.txt')
        if not os.path.exists(txt_path):
            download_shakespeare(txt_path)

    with open(txt_path, 'r', encoding='utf-8') as f:
        data = f.read()
    print(f"Dataset length: {len(data):,} characters")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Unique characters: {vocab_size}")
    print(f"Characters: {''.join(chars)}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens:   {len(val_ids):,}")

    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))

    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"Data prepared in {data_dir}/")


def main():
    p = argparse.ArgumentParser(description='Prepare training data')
    p.add_argument('--data_dir', default='data/shakespeare_char')
    p.add_argument('--input_file', default=None, help='Path to a custom text file')
    args = p.parse_args()
    prepare(args.data_dir, args.input_file)


if __name__ == '__main__':
    main()
