# cct/tokenizer.py
from __future__ import annotations
import json
import os
from typing import List, Dict, Any, TYPE_CHECKING

# Only for type hints; does not import at runtime
if TYPE_CHECKING:
    from transformers import GPT2TokenizerFast

_tokenizer = None


def get_gpt2_tokenizer() -> "GPT2TokenizerFast":
    """
    Lazily create a GPT-2 tokenizer so modules that only need meta helpers
    (e.g., validate) don't import `transformers`.
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import GPT2TokenizerFast  # lazy import
        _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # ensure pad token exists (GPT-2 lacks one by default)
        if _tokenizer.pad_token is None:
            _tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return _tokenizer


class CharTokenizer:
    """
    Simple char-level tokenizer (for tiny Shakespeare).
    Expects meta with 'itos' (list of characters).
    """
    def __init__(self, itos: List[str]):
        self.itos = itos
        self.stoi = {ch: i for i, ch in enumerate(itos)}

    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] if 0 <= i < len(self.itos) else "?" for i in ids)


def load_meta(data_dir: str) -> Dict[str, Any]:
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"meta.json not found in {data_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tokenizer_from_meta(meta: Dict[str, Any]):
    dataset = meta.get("dataset")
    if dataset == "tiny":
        return CharTokenizer(meta["itos"])
    # default: GPT-2
    return get_gpt2_tokenizer()
