import torch
from peft_lab.model import TinyIntentTransformer, inject_lora, count_params

def test_lora_reduces_trainable_parameters():
    m=TinyIntentTransformer(100,8);total,_=count_params(m);m=inject_lora(m,r=4);total2,trainable=count_params(m)
    assert total2>total and 0 < trainable/total2 < .10
    y=m(torch.randint(0,100,(2,12)));assert y.shape==(2,8)
