# Data attribution

The measured CI benchmark in this repository uses **independently generated synthetic text only**.

The optional GPU recipe in `recipes/qwen3_lora_sft.py` downloads the public **BANKING77** dataset from Hugging Face (`PolyAI/banking77`). BANKING77 contains 13,083 online banking queries across 77 intents and is distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

Original dataset citation:

> Iñigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson, Ivan Vulić. "Efficient Intent Detection with Dual Sentence Encoders." Proceedings of the 2nd Workshop on NLP for Conversational AI, 2020.

Dataset source: `https://huggingface.co/datasets/PolyAI/banking77`

No BANKING77 examples are committed to this repository by default.
