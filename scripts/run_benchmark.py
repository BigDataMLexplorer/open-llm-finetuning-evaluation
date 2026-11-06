from __future__ import annotations

import json
from pathlib import Path
import statistics

import torch

from peft_lab.data import build_dataset, build_vocab, INTENTS
from peft_lab.model import TinyIntentTransformer, clone_model, inject_lora, count_params
from peft_lab.train import seed_everything, train
from peft_lab.eval import evaluate

ROOT=Path(__file__).resolve().parents[1]


def run_seed(seed:int) -> dict:
    seed_everything(seed)
    splits=build_dataset(seed=17)
    vocab=build_vocab(splits)
    base=TinyIntentTransformer(len(vocab),len(INTENTS))
    train(base,splits['formal_train'],vocab,epochs=10,lr=2e-3,weight_decay=1e-4)
    base_formal=evaluate(base,splits['formal_eval'],vocab)
    base_col=evaluate(base,splits['colloquial_eval'],vocab)

    full=clone_model(base)
    for p in full.parameters(): p.requires_grad=True
    train(full,splits['colloquial_train'],vocab,epochs=7,lr=1.5e-3,weight_decay=1e-4)
    full_formal=evaluate(full,splits['formal_eval'],vocab)
    full_col=evaluate(full,splits['colloquial_eval'],vocab)

    lora=inject_lora(clone_model(base),r=4,alpha=8)
    train(lora,splits['colloquial_train'],vocab,epochs=12,lr=8e-3)
    lora_formal=evaluate(lora,splits['formal_eval'],vocab)
    lora_col=evaluate(lora,splits['colloquial_eval'],vocab)

    replay=inject_lora(clone_model(base),r=4,alpha=8)
    replay_examples=list(splits['colloquial_train']) + list(splits['formal_train'][::5])
    train(replay,replay_examples,vocab,epochs=12,lr=8e-3)
    replay_formal=evaluate(replay,splits['formal_eval'],vocab)
    replay_col=evaluate(replay,splits['colloquial_eval'],vocab)

    total,trainable=count_params(lora)
    return {
      'seed':seed,
      'params':{'total':total,'lora_trainable':trainable,'trainable_pct':trainable/total*100},
      'base':{'formal':base_formal,'colloquial':base_col},
      'full_ft':{'formal':full_formal,'colloquial':full_col},
      'lora':{'formal':lora_formal,'colloquial':lora_col},
      'lora_replay':{'formal':replay_formal,'colloquial':replay_col},
    }


def avg(rows, path):
    vals=[]
    for r in rows:
        cur=r
        for key in path: cur=cur[key]
        vals.append(cur)
    return {'mean':statistics.mean(vals),'stdev':statistics.stdev(vals) if len(vals)>1 else 0.0}


def main():
    torch.set_num_threads(4)
    rows=[run_seed(s) for s in [11,29,47]]
    summary={
      'seeds':[11,29,47],
      'base_formal_accuracy':avg(rows,['base','formal','accuracy']),
      'base_colloquial_accuracy':avg(rows,['base','colloquial','accuracy']),
      'full_ft_formal_accuracy':avg(rows,['full_ft','formal','accuracy']),
      'full_ft_colloquial_accuracy':avg(rows,['full_ft','colloquial','accuracy']),
      'lora_formal_accuracy':avg(rows,['lora','formal','accuracy']),
      'lora_colloquial_accuracy':avg(rows,['lora','colloquial','accuracy']),
      'replay_formal_accuracy':avg(rows,['lora_replay','formal','accuracy']),
      'replay_colloquial_accuracy':avg(rows,['lora_replay','colloquial','accuracy']),
      'lora_trainable_pct':rows[0]['params']['trainable_pct'],
    }
    summary['full_ft_forgetting_pp']=(summary['base_formal_accuracy']['mean']-summary['full_ft_formal_accuracy']['mean'])*100
    summary['lora_forgetting_pp']=(summary['base_formal_accuracy']['mean']-summary['lora_formal_accuracy']['mean'])*100
    summary['replay_forgetting_pp']=(summary['base_formal_accuracy']['mean']-summary['replay_formal_accuracy']['mean'])*100
    out=ROOT/'artifacts'/'benchmark.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({'summary':summary,'runs':rows},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
