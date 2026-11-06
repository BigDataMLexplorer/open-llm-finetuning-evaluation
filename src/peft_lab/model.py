from __future__ import annotations

import copy
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(self.dropout(x))) * self.scale


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        def shape(z):
            return z.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = shape(self.q_proj(x)), shape(self.k_proj(x)), shape(self.v_proj(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_mask = mask[:, None, None, :].bool()
        scores = scores.masked_fill(~key_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x, mask))
        return self.ln2(x + self.ff(x))


class TinyIntentTransformer(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, d_model: int = 64, n_heads: int = 4, layers: int = 2, d_ff: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(layers)])
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids.ne(0)
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.head(pooled)


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def inject_lora(model: TinyIntentTransformer, r: int = 4, alpha: float = 8.0) -> TinyIntentTransformer:
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        block.attn.q_proj = LoRALinear(block.attn.q_proj, r=r, alpha=alpha)
        block.attn.v_proj = LoRALinear(block.attn.v_proj, r=r, alpha=alpha)
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
