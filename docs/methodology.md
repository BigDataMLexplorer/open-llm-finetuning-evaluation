# Methodology

The CI benchmark is deliberately small and synthetic so it can be rerun without downloading models or using a GPU. It tests the engineering properties of sequential adaptation: learning a lexical/style shift, parameter efficiency, retention, multi-seed stability and evaluation reporting.

The repository separately provides a GPU-ready Hugging Face TRL/PEFT recipe for Qwen3-0.6B on the public BANKING77 dataset. Metrics from that recipe are **not** claimed unless a real GPU run is executed and its artifacts are added.

This separation keeps the portfolio reproducible without inventing open-LLM results.
