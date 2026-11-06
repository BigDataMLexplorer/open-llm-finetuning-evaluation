"""GPU recipe for real open-model SFT. Not executed in CI.

Uses public PolyAI/BANKING77 and modern Hugging Face TRL + PEFT APIs.
Run on a CUDA GPU after installing the optional `gpu` dependencies.
"""
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen3-0.6B"
DATASET = "PolyAI/banking77"

raw = load_dataset(DATASET)
label_names = raw["train"].features["label"].names

def format_row(row):
    return {
        "prompt": [{"role":"user","content":f"Classify the banking support intent. Return only the intent label.\nQuery: {row['text']}"}],
        "completion": [{"role":"assistant","content":label_names[row['label']]}],
    }

train_ds = raw["train"].map(format_row, remove_columns=raw["train"].column_names)
eval_ds = raw["test"].map(format_row, remove_columns=raw["test"].column_names)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

config = SFTConfig(
    output_dir="outputs/qwen3-banking77-lora",
    learning_rate=1e-4,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    max_length=256,
    completion_only_loss=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=MODEL,
    args=config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model()
