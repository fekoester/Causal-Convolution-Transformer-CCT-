#!/usr/bin/env python3
"""
Generate text from a checkpoint.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch

from cct import CCTConfig, CausalConvTransformerLM
from cct.checkpoint import load_checkpoint, load_sidecar_config
from cct.tokenizer import load_meta, get_tokenizer_from_meta


@torch.no_grad()
def generate(model, idx, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.seq_length :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-8, temperature)

        # top-k filter in logits
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)

        # top-p (nucleus) on probs
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0.0
            probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Used to pick the right tokenizer (char vs GPT-2 BPE).")
    parser.add_argument("--prompt", type=str, default="Hello world")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None,
                        help="Optional: override context length; default uses training value from sidecar.")
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config from sidecar
    cfg_dict = load_sidecar_config(args.ckpt)
    if args.seq_length is not None:
        cfg_dict["seq_length"] = args.seq_length

    # tokenizer per dataset meta
    meta = load_meta(args.data_dir)
    tok = get_tokenizer_from_meta(meta)
    cfg_dict["vocab_size"] = int(meta["vocab_size"])

    config = CCTConfig(**cfg_dict)
    model = CausalConvTransformerLM(config).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()

    x = torch.tensor([tok.encode(args.prompt)], dtype=torch.long).to(device)
    y = generate(model, x, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
    print(tok.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
