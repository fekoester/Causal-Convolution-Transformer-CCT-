#!/usr/bin/env python3
"""
Compute validation loss and perplexity for a checkpoint.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
from torch.utils.data import DataLoader

from cct import LMDataset, CCTConfig, CausalConvTransformerLM
from cct.checkpoint import load_checkpoint, load_sidecar_config
from cct.tokenizer import load_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--seq-length", type=int, default=None,
                        help="Optional: override context length; default uses training value from sidecar.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model config from sidecar (authoritative)
    cfg_dict = load_sidecar_config(args.ckpt)
    if args.seq_length is not None:
        cfg_dict["seq_length"] = args.seq_length  # allow manual override if desired

    # ensure vocab_size matches dataset meta
    meta = load_meta(args.data_dir)
    cfg_dict["vocab_size"] = int(meta["vocab_size"])

    config = CCTConfig(**cfg_dict)
    model = CausalConvTransformerLM(config).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()

    val_ds = LMDataset(os.path.join(args.data_dir, "val.bin"), config.seq_length)
    val_loader = DataLoader(val_ds, batch_size=8, drop_last=False)

    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)

    avg = total_loss / max(1, n)
    ppl = float(torch.exp(torch.tensor(avg)))
    print(f"Validation loss {avg:.4f}, perplexity {ppl:.2f}")


if __name__ == "__main__":
    main()
