# cct/checkpoint.py
import os
import json
import torch
from typing import Optional, Dict, Any
import re


def save_checkpoint(path, model, optimizer, scaler, scheduler, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scaler": scaler.state_dict() if scaler else None,
        # save a tiny scheduler state if it has state_dict; else None
        "scheduler": getattr(scheduler, "state_dict", lambda: None)() if scheduler else None,
        "state": state,
    }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # explicitly allow full load
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("state", {})

def try_write_sidecar_config(ckpt_dir: str, config: dict):
    """Write model_config.json once for reproducibility."""
    os.makedirs(ckpt_dir, exist_ok=True)
    sidecar = os.path.join(ckpt_dir, "model_config.json")
    if not os.path.exists(sidecar):
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


def ckpt_dir_from_path(ckpt_path: str) -> str:
    """Return the directory containing the checkpoint file."""
    if os.path.isdir(ckpt_path):
        return ckpt_path
    return os.path.dirname(os.path.abspath(ckpt_path))


def load_sidecar_config(ckpt_path_or_dir: str) -> Dict[str, Any]:
    """Load model_config.json next to the checkpoint."""
    d = ckpt_dir_from_path(ckpt_path_or_dir)
    sidecar = os.path.join(d, "model_config.json")
    if not os.path.exists(sidecar):
        raise FileNotFoundError(f"model_config.json not found in {d}")
    with open(sidecar, "r", encoding="utf-8") as f:
        return json.load(f)

def latest_checkpoint_in_dir(d: str) -> str | None:
    if not os.path.isdir(d):
        return None
    files = [f for f in os.listdir(d) if f.endswith(".pt")]
    if not files:
        return None
    def epnum(name):
        m = re.search(r"epoch(\d+)\.pt$", name)
        return int(m.group(1)) if m else -1
    files.sort(key=lambda f: (epnum(f), os.path.getmtime(os.path.join(d, f))), reverse=True)
    return os.path.join(d, files[0])

def resolve_resume_arg(resume: str) -> str:
    # explicit file
    if os.path.isfile(resume):
        return resume
    # explicit directory
    if os.path.isdir(resume):
        path = latest_checkpoint_in_dir(resume)
        if path: return path
        raise FileNotFoundError(f"No .pt files in {resume}")
    # treat as tag under checkpoints/
    cand = os.path.join("checkpoints", resume)
    if os.path.isdir(cand):
        path = latest_checkpoint_in_dir(cand)
        if path: return path
        raise FileNotFoundError(f"No .pt files in {cand}")
    raise FileNotFoundError(f"Could not resolve resume target: {resume}")