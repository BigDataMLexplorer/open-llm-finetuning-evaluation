from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import Example, encode


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def as_loader(examples: list[Example], vocab: dict[str, int], batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    x = torch.tensor([encode(e.text, vocab) for e in examples], dtype=torch.long)
    y = torch.tensor([e.label for e in examples], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def train(model: nn.Module, examples: list[Example], vocab: dict[str, int], epochs: int, lr: float, weight_decay: float = 0.0) -> list[float]:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = as_loader(examples, vocab, shuffle=True)
    losses=[]
    model.train()
    for _ in range(epochs):
        total=0.0; n=0
        for x,y in loader:
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0)
            opt.step()
            total += float(loss.detach()) * len(y); n += len(y)
        losses.append(total/n)
    return losses
