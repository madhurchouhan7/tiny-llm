---
language:
- en
license: mit
tags:
- gpt2
- tiny-llm
- educational
- transformers-from-scratch
- pytorch
---

# Tiny-LLM-From-Scratch (~50M)

A ~50M parameter GPT-2-style decoder-only Transformer built **entirely from scratch** in PyTorch
and pre-trained on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

This model was created for educational purposes to demonstrate how a GPT-style LLM works
at the tensor/code level.  Every component—tokenizer, data pipeline, attention, MLP,
training loop, and generation—is implemented from scratch without using the `transformers` library.

## Architecture Details

| Parameter | Value |
|---|---|
| **Architecture** | Decoder-only Transformer (Pre-LayerNorm, GPT-2 style) |
| **Total parameters** | 50,677,760 (50.68 M) |
| **Context length** | 256 tokens |
| **Vocabulary** | 16,384 Byte-Level BPE tokens |
| **Embedding dimension (C)** | 640 |
| **Attention heads (H)** | 10 (head_dim = 64) |
| **Transformer layers (N)** | 6 |
| **FFN hidden dim** | 2,560 (4 x C) |
| **Activation** | GELU |
| **Weight tying** | No |

## Training Details

| Setting | Value |
|---|---|
| **Dataset** | FineWeb-Edu sample-10BT |
| **Training tokens** | ~20 M |
| **Tokenizer** | Custom Byte-Level BPE (trained on 80k docs) |
| **Optimiser** | AdamW (lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)) |
| **LR schedule** | Cosine with 500-step linear warm-up |
| **Effective batch size** | 128 sequences (32 × 4 grad accum) |
| **Steps** | 5,000 |
| **AMP** | FP16 (PyTorch autocast + GradScaler) |
| **Hardware** | Google Colab T4 (16 GB) |

## Sampling Capabilities

Supports autoregressive text generation with:
- **Temperature** scaling
- **Top-K** filtering
- **Nucleus (Top-P)** sampling

## Usage (PyTorch)

```python
from huggingface_hub import hf_hub_download
import torch, json

# Download files
config_path = hf_hub_download(repo_id="tiny-llm-50m", filename="config.json")
weights_path = hf_hub_download(repo_id="tiny-llm-50m", filename="pytorch_model.bin")

# Rebuild model (copy GPTConfig and GPT classes from the training notebook)
with open(config_path) as f:
    cfg_dict = json.load(f)
config = GPTConfig(**cfg_dict)
model  = GPT(config)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()
```

## Source Code

Full project source: [Tiny-LLM-From-Scratch](https://github.com/madhurchouhan7/tiny-llm)
