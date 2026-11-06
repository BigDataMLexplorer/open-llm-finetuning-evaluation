from peft_lab.data import build_dataset, build_vocab

def test_dataset_is_deterministic_and_balanced():
    a=build_dataset(seed=17,train_per_intent=10,eval_per_intent=5);b=build_dataset(seed=17,train_per_intent=10,eval_per_intent=5)
    assert [x.text for x in a['formal_train']]==[x.text for x in b['formal_train']]
    assert len(a['formal_train'])==80 and len(a['colloquial_eval'])==40
    assert len(build_vocab(a))>50
