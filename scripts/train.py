#!/usr/bin/env python3
"""
Train a GPT-2 style LM with causal convolution residuals.
- CPU/GPU auto
- Datasets: tiny (char) or openwebtext (GPT-2 BPE)
- Fresh training or resume (absolute target or +N more epochs)
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # allow "from cct import ..." without install

import argparse
import random, numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from cct import (
    CCTConfig,
    CausalConvTransformerLM,
    LMDataset,
    build_optimizer,
    WarmHoldCosineLR,
    save_checkpoint,
    try_write_sidecar_config,
)
from cct.download_data import prepare_dataset
from cct.checkpoint import resolve_resume_arg, load_sidecar_config, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    # dataset / io
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["openwebtext", "tiny"], default="openwebtext",
                        help="Choose 'openwebtext' (GPT-2 BPE) or 'tiny' (char-level).")
    parser.add_argument("--subset", type=str, default=None,
                        help="Optional HF slice for openwebtext, e.g. 'train[:1%]' for quick tests.")
    parser.add_argument("--tag", type=str, required=True)

    # model (used when starting fresh; ignored on resume except seq_length override via sidecar if changed)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--dyn-window", type=int, default=2)
    parser.add_argument("--dyn-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)

    # optimization
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=1)  # safer default
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.01)
    parser.add_argument("--hold-frac", type=float, default=0.01)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # reproducibility / resume
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path/tag/dir to checkpoint to resume from (e.g. 'tiny_v3', 'checkpoints/tiny_v3', or an explicit .pt).")
    parser.add_argument("--epochs-add", type=int, default=None,
                        help="If set, run this many additional epochs beyond the checkpoint’s epoch. Overrides --epochs.")
    parser.add_argument("--resume-exact", action="store_true",
                        help="Treat --resume as an exact file path; do not auto-pick latest in a dir/tag.")

    args = parser.parse_args()

    # seeding
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    # ensure data exists; fetch meta for vocab_size/tokenizer kind
    meta = prepare_dataset(args.data_dir, dataset=args.dataset, subset=args.subset)
    vocab_size = int(meta["vocab_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda_amp = (device == "cuda")
    accum = max(1, args.accum_steps)

    # resolve resume path (optional)
    ckpt_path = None
    if args.resume:
        if args.resume_exact:
            if not os.path.isfile(args.resume):
                raise FileNotFoundError(f"--resume-exact given but not a file: {args.resume}")
            ckpt_path = os.path.abspath(args.resume)
        else:
            ckpt_path = resolve_resume_arg(args.resume)

    # Build config: from sidecar if resuming, else from CLI + meta
    if ckpt_path:
        cfg_dict = load_sidecar_config(ckpt_path)  # authoritative training config
        cfg_dict["vocab_size"] = vocab_size        # ensure current dataset vocab matches
    else:
        cfg_dict = dict(
            vocab_size=vocab_size,
            seq_length=args.seq_length,
            n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
            dyn_window=args.dyn_window, dyn_layers=args.dyn_layers, dropout=args.dropout,
        )

    # model/optim/scaler
    config = CCTConfig(**cfg_dict)
    model = CausalConvTransformerLM(config).to(device)
    optimizer = build_optimizer(model, args.lr)
    scaler = GradScaler(enabled=use_cuda_amp)

    # data loader (need this to compute steps/epoch)
    train_ds = LMDataset(os.path.join(args.data_dir, "train.bin"), config.seq_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # sidecar for this run/tag (create if missing)
    ckpt_dir = os.path.join("checkpoints", args.tag)
    try_write_sidecar_config(ckpt_dir, config.to_dict())

    # resume state (load AFTER optimizer/scaler exist; scheduler will be built for the remaining run)
    start_epoch, step = 0, 0
    if ckpt_path:
        state = load_checkpoint(ckpt_path, model, optimizer, scaler)
        start_epoch = int(state.get("epoch", -1)) + 1
        step = int(state.get("step", 0))

    # determine target epochs
    if args.epochs_add is not None:
        target_epochs = start_epoch + max(0, args.epochs_add)
    else:
        target_epochs = args.epochs

    print(f"[train] device={device} amp={'cuda' if use_cuda_amp else 'off'} accum={accum}")
    if ckpt_path:
        print(f"[resume] from {ckpt_path} -> start_epoch={start_epoch}, step={step}, target_epochs={target_epochs}")
    else:
        print(f"[fresh] start_epoch=0, target_epochs={target_epochs}")

    # nothing to do? exit cleanly
    if start_epoch >= target_epochs:
        print(f"[train] start_epoch ({start_epoch}) >= target ({target_epochs}); nothing to do.")
        return

    # scheduler for the REMAINING run
    steps_remaining = max(1, (target_epochs - start_epoch) * len(train_loader) // accum)
    scheduler = WarmHoldCosineLR(optimizer, steps_remaining, args.warmup_frac, args.hold_frac, args.min_lr_ratio) if args.use_scheduler else None

    # train
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, target_epochs):
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            if use_cuda_amp:
                with autocast(device_type="cuda"):
                    _, loss = model(x, y)
                    loss = loss / accum
                scaler.scale(loss).backward()
            else:
                _, loss = model(x, y)
                (loss / accum).backward()

            do_step = ((it + 1) % accum == 0) or ((it + 1) == len(train_loader))
            if do_step:
                if args.grad_clip and args.grad_clip > 0:
                    if use_cuda_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                if use_cuda_amp:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler: scheduler.step()
                step += 1

                if step % 100 == 0:
                    print(f"epoch {epoch} iter {it} step {step}: loss {loss.item():.4f}")

        state_blob = {"epoch": epoch, "step": step,
                      "scheduler": scheduler.state_dict() if scheduler else None}
        save_checkpoint(os.path.join(ckpt_dir, f"epoch{epoch+1}.pt"),
                        model, optimizer, scaler, scheduler, state_blob)

if __name__ == "__main__":
    main()
