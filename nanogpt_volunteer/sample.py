"""
Sample from a trained nanoGPT model checkpoint.

Usage:
    python -m nanogpt_volunteer.sample --out_dir out-volunteer
    python -m nanogpt_volunteer.sample --out_dir out-volunteer --start "To be or not"
"""

import os
import pickle
import argparse
from contextlib import nullcontext

import torch

from .model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser(description='Sample from a trained nanoGPT model')
    p.add_argument('--out_dir', default='out-volunteer')
    p.add_argument('--start', default='\n', help='Starting text or FILE:path.txt')
    p.add_argument('--num_samples', type=int, default=5)
    p.add_argument('--max_new_tokens', type=int, default=500)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=200)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'float16'])
    p.add_argument('--data_dir', default='data/shakespeare_char',
                   help='Data directory containing meta.pkl for encoding')
    args = p.parse_args()

    torch.manual_seed(1337)
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    conf = GPTConfig(**checkpoint['model_args'])
    model = GPT(conf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)

    meta_path = os.path.join(args.data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    start_text = args.start
    if start_text.startswith('FILE:'):
        with open(start_text[5:], 'r', encoding='utf-8') as f:
            start_text = f.read()

    start_ids = encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    with torch.no_grad():
        with ctx:
            for k in range(args.num_samples):
                y = model.generate(x, args.max_new_tokens,
                                   temperature=args.temperature, top_k=args.top_k)
                print(decode(y[0].tolist()))
                print('---------------')


if __name__ == '__main__':
    main()
