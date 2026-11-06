from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .data import Example
from .train import as_loader


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    edges=np.linspace(0,1,bins+1); ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(conf>lo)&(conf<=hi)
        if mask.any():
            ece += mask.mean() * abs((pred[mask]==labels[mask]).mean() - conf[mask].mean())
    return float(ece)


def evaluate(model, examples: list[Example], vocab: dict[str,int]) -> dict:
    model.eval(); ys=[]; probs=[]
    with torch.no_grad():
        for x,y in as_loader(examples,vocab,shuffle=False):
            p=torch.softmax(model(x),dim=-1)
            probs.append(p.cpu().numpy()); ys.append(y.cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(probs); pred=p.argmax(axis=1)
    cm=confusion_matrix(y,pred)
    pairs=[]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i!=j and cm[i,j]>0:
                pairs.append((int(cm[i,j]),i,j))
    pairs=sorted(pairs,reverse=True)[:5]
    return {
        "accuracy": float(accuracy_score(y,pred)),
        "macro_f1": float(f1_score(y,pred,average="macro",zero_division=0)),
        "ece": expected_calibration_error(p,y),
        "top_confusions": [{"count":n,"true":i,"pred":j} for n,i,j in pairs],
    }
