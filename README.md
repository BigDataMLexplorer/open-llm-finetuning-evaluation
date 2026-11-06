# Open-Model Fine-Tuning & Evaluation Lab

Independent portfolio project for reproducible parameter-efficient fine-tuning, retention measurement and evaluation engineering.

The repository intentionally separates two tracks:

1. **Measured CPU/CI benchmark** — a tiny transformer is trained on synthetic banking-intent phrasing, then adapted to a lexical/style shift with full fine-tuning, LoRA, and LoRA + replay.
2. **GPU-ready open-model recipe** — Hugging Face TRL + PEFT LoRA for Qwen3-0.6B on public BANKING77. No Qwen metric is claimed until that recipe is genuinely run on GPU.

## Measured 3-seed retention benchmark

| Strategy | Original-style accuracy | Adapted-style accuracy | Forgetting |
| --- | ---: | ---: | ---: |
| Base before adaptation | **100.0%** | **50.6%** | — |
| Full fine-tune | **93.6%** | **100.0%** | **6.4 pp** |
| LoRA | **96.5%** | **100.0%** | **3.5 pp** |
| LoRA + 20% replay | **98.8%** | **100.0%** | **1.3 pp** |

LoRA trains **2.53% of model parameters** in the measured benchmark.

Results are means across deterministic seeds 11, 29 and 47. This is a synthetic adaptation/retention benchmark, not a claim about Qwen or BANKING77 model accuracy.

## What it demonstrates

- PyTorch training and evaluation pipeline
- low-rank adapters injected into attention query/value projections
- parameter-efficiency accounting
- sequential adaptation / catastrophic-forgetting measurement
- replay-based retention strategy
- accuracy, macro-F1, calibration error and confusion analysis
- deterministic multi-seed experiment reporting
- CI-generated benchmark artifacts
- GPU-ready TRL / PEFT recipe for Qwen3

## Run the measured benchmark

```bash
pip install -e '.[dev]'
pytest -q
PYTHONPATH=src python scripts/run_benchmark.py
PYTHONPATH=src python scripts/plot_benchmark.py
```

## Run the Qwen recipe on GPU

```bash
pip install -e '.[gpu]'
python recipes/qwen3_lora_sft.py
```

The GPU recipe uses public `PolyAI/banking77` data. Metrics from that recipe are intentionally not pre-filled.
