import math
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CCTConfig:
    vocab_size: int = 50_257
    seq_length: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    dyn_window: int = 2
    dyn_layers: int = 1
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q, self.k, self.v = (nn.Linear(n_embd, n_embd) for _ in range(3))
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop.p)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class TransformerBlockWithCausalConvs(nn.Module):
    """Pre-norm transformer block with causal conv residual branch."""

    def __init__(self, n_embd, n_head, dropout, L, K=1):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(n_embd), MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )
        self.conv_path = nn.ModuleList([nn.Conv1d(n_embd, n_embd, kernel_size=L + 1, bias=False) for _ in range(K)])
        with torch.no_grad():
            for conv in self.conv_path:
                conv.weight[:, :, 0].zero_()
        self.inter_norm, self.act, self.res_scale = nn.LayerNorm(n_embd), nn.GELU(), (1.0 / K if K > 0 else 0.0)
        self._enabled = K > 0

    def forward(self, x):
        if self._enabled:
            H = x.transpose(1, 2)
            acc = None
            for i, conv in enumerate(self.conv_path):
                pad = conv.kernel_size[0] - 1
                y = conv(F.pad(H, (pad, 0))).transpose(1, 2)
                y = self.inter_norm(y)
                y = self.act(y) if i < len(self.conv_path) - 1 else y
                acc = y if acc is None else acc + y
            x = x + self.res_scale * acc
        return x + self.attn(self.ln1(x)) + self.mlp(self.ln2(x))


class CausalConvTransformerLM(nn.Module):
    def __init__(self, config: CCTConfig):
        super().__init__()
        self.config = config
        self.seq_length = config.seq_length - 1
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_length, config.n_embd))
        self.blocks = nn.Sequential(
            *[
                TransformerBlockWithCausalConvs(
                    config.n_embd, config.n_head, config.dropout, config.dyn_window, config.dyn_layers
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f, self.lm_head = nn.LayerNorm(config.n_embd), nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # GPT-2 style init: small weights help stabilize early loss
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.seq_length
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
