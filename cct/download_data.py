# cct/download_data.py
import os
import json
import numpy as np
from typing import Literal, Dict, Any

from tqdm import tqdm
from datasets import load_dataset

from .tokenizer import get_gpt2_tokenizer


def _save_meta(data_dir: str, meta: Dict[str, Any]) -> None:
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_meta_if_exists(data_dir: str) -> Dict[str, Any] | None:
    path = os.path.join(data_dir, "meta.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _already_prepared(data_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(data_dir, "train.bin"))
        and os.path.exists(os.path.join(data_dir, "val.bin"))
        and os.path.exists(os.path.join(data_dir, "meta.json"))
    )


def _prepare_tiny(data_dir: str) -> None:
    """
    Tiny Shakespeare (char-level).
    Saves:
      - train.bin / val.bin (uint16 ids)
      - meta.json with {'dataset':'tiny','itos':[...],'vocab_size':N}
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if _already_prepared(data_dir):
        print(f"[cct] Found existing tiny Shakespeare dataset in {data_dir}")
        return

    print("[cct] Downloading tiny Shakespeare (char-level)…")
    import requests

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url, timeout=60).text

    print("[cct] Building char vocabulary…")
    chars = sorted(list(set(text)))
    itos = chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    vocab_size = len(itos)
    print(f"[cct] tiny vocab_size = {vocab_size}")

    print("[cct] Encoding to ids…")
    ids = np.array([stoi[c] for c in text], dtype=np.uint32)  # temporary
    # 90/10 split
    split = int(0.9 * len(ids))
    train_ids = ids[:split]
    val_ids = ids[split:]

    # safe to uint16 since vocab_size <= 65535 by a wide margin here
    train_ids.astype(np.uint16).tofile(train_path)
    val_ids.astype(np.uint16).tofile(val_path)

    _save_meta(data_dir, {"dataset": "tiny", "itos": itos, "vocab_size": vocab_size})

    print(f"[cct] Saved {train_path} ({len(train_ids)} tokens)")
    print(f"[cct] Saved {val_path} ({len(val_ids)} tokens)")
    print("[cct] tiny Shakespeare ready.")


def _prepare_openwebtext(data_dir: str, subset: str | None = None) -> None:
    """
    OpenWebText via HuggingFace (GPT-2 BPE).
    subset: optional HF split slice like "train[:1%]" for quick tests.
    Saves:
      - train.bin / val.bin (uint16 ids)
      - meta.json with {'dataset':'openwebtext','vocab_size':50257}
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if _already_prepared(data_dir):
        print(f"[cct] Found existing OpenWebText dataset in {data_dir}")
        return

    split = subset if subset else "train"
    print(f"[cct] Downloading OpenWebText (split='{split}') via HuggingFace…")
    dataset = load_dataset("openwebtext", split=split)

    print("[cct] Tokenizing with GPT-2 tokenizer (this can take a while)…")
    tok = get_gpt2_tokenizer()

    # Accumulate ids across docs, then 90/10 split by token count
    all_ids = []
    for ex in tqdm(dataset, desc="Tokenizing OWT"):
        ids = tok.encode(ex["text"])
        all_ids.extend(ids)

    total = len(all_ids)
    print(f"[cct] Total tokens: {total}")
    if total == 0:
        raise RuntimeError("No tokens produced. Check dataset download/tokenizer.")

    split_ix = int(0.9 * total)
    train_ids = np.array(all_ids[:split_ix], dtype=np.uint32)
    val_ids = np.array(all_ids[split_ix:], dtype=np.uint32)

    # GPT-2 vocab fits in uint16 (0..50256)
    train_ids.astype(np.uint16).tofile(train_path)
    val_ids.astype(np.uint16).tofile(val_path)

    _save_meta(data_dir, {"dataset": "openwebtext", "vocab_size": 50257})

    print(f"[cct] Saved {train_path} ({len(train_ids)} tokens)")
    print(f"[cct] Saved {val_path} ({len(val_ids)} tokens)")
    print("[cct] OpenWebText ready.")


def prepare_dataset(
    data_dir: str,
    dataset: Literal["openwebtext", "tiny"] = "openwebtext",
    subset: str | None = None,
) -> Dict[str, Any]:
    """
    Ensure tokenized dataset exists in data_dir.
    Returns the meta dict.
    - dataset="openwebtext" (default): GPT-2 BPE tokenizer; optional subset like "train[:1%]".
    - dataset="tiny": tiny Shakespeare char-level tokenizer.

    Files created: train.bin, val.bin, meta.json
    """
    meta = _load_meta_if_exists(data_dir)
    if not _already_prepared(data_dir):
        if dataset == "tiny":
            _prepare_tiny(data_dir)
        else:
            _prepare_openwebtext(data_dir, subset=subset)
        meta = _load_meta_if_exists(data_dir)
    if meta is None:
        raise RuntimeError("prepare_dataset failed to write meta.json")
    return meta
