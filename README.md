# Tiny-LLM-From-Scratch

A **~50M parameter GPT-2-style decoder-only Transformer** built from scratch in PyTorch and pretrained on a small subset of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

> **Goal:** Understand how a GPT-style language model works at the tensor/code level — from raw text all the way to generated output — rather than simply obtaining a working model.

The GPT-style model architecture, data pipeline, training loop, checkpointing, and generation/sampling logic are implemented directly in PyTorch without using the `transformers` library. The BPE tokenizer is trained using Hugging Face's `tokenizers` library.

---

## Architecture at a Glance

```text
Raw text
   ↓  Byte-Level BPE tokenizer (vocab = 16,384)
Token IDs (B, T)
   ↓
Token embedding + positional embedding
   ↓
(B, T, 640)
   ↓
× 6 Transformer Blocks
   ├── LayerNorm → Causal Multi-Head Attention → Residual
   └── LayerNorm → Feed-Forward MLP → Residual
   ↓
(B, T, 640)
   ↓
Final LayerNorm
   ↓
Linear LM Head
   ↓
Logits (B, T, 16,384)
   ↓
Cross-entropy loss / Sampling
   ↓
Next token
```

### Model Configuration

| Parameter | Value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Parameters | **50,677,760 (~50.68M)** |
| Vocabulary | 16,384 Byte-Level BPE tokens |
| Context length | 256 tokens |
| Embedding dimension | 640 |
| Attention heads | 10 |
| Head dimension | 64 |
| Transformer layers | 6 |
| FFN hidden dimension | 2,560 (4 × embedding dimension) |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Weight tying | No |
| Dropout | 0.1 |

---

## Training

The model was trained on a controlled subset of FineWeb-Edu to keep the experiment feasible on a single Google Colab T4.

| Setting | Value |
|---|---|
| Dataset | FineWeb-Edu sample-10BT |
| Training tokens | ~20M |
| Tokenizer | Custom Byte-Level BPE |
| Tokenizer training data | 80k documents |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.1 |
| Betas | (0.9, 0.95) |
| LR schedule | Cosine decay with 500-step linear warm-up |
| Effective batch size | 128 sequences |
| Gradient accumulation | 4 |
| Training steps | 5,000 |
| Precision | FP16 AMP |
| Hardware | Google Colab T4 (16 GB) |

> **Note:** This is an educational proof-of-concept, not a production-scale pretrained language model. The ~20M-token training budget was intentionally kept small for a single-GPU experiment.

---

## Sampling

The generation implementation supports:

- Greedy decoding
- Temperature sampling
- Top-K sampling
- Top-P / nucleus sampling

Example:

```bash
python scripts/generate.py \
    --checkpoint checkpoints/50m/best.pt \
    --prompt "The future of artificial intelligence" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9
```

---

## Project Structure

```text
tiny-llm/
├── configs/
│   ├── debug.yaml          # Tiny model for fast local testing
│   ├── tiny.yaml           # Small overfitting / debugging run
│   └── 50m.yaml            # ~50M-parameter model
│
├── src/
│   ├── tokenizer/          # BPE tokenizer wrapper
│   ├── data/               # Preprocessing, dataset, dataloader
│   ├── model/              # Embeddings, attention, MLP, block, GPT
│   ├── training/           # Trainer, optimizer, checkpointing
│   └── generation.py       # Text generation and sampling
│
├── scripts/
│   ├── train_tokenizer.py
│   ├── analyze_dataset.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
│
├── tests/                  # Pytest unit tests
├── experiments/            # Logs and generated samples
├── checkpoints/            # Saved model/optimizer states
├── artifacts/              # Tokenizer artifacts
├── docs/                   # Architecture and tensor-shape docs
└── train_and_publish_colab.ipynb
```

---

## Quickstart

### 🚀 Google Colab

Open [`train_and_publish_colab.ipynb`](./train_and_publish_colab.ipynb) in Google Colab with a T4 GPU.

The notebook provides the training/export workflow for the Tiny LLM, including FineWeb-Edu preprocessing, model training, tokenizer/model export, and Hugging Face publishing.

### Local Execution

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

#### 2. Train the BPE tokenizer

```bash
python scripts/train_tokenizer.py
```

#### 3. Analyze and preprocess FineWeb-Edu

```bash
python scripts/analyze_dataset.py
python scripts/prepare_data.py --config configs/tiny.yaml
```

#### 4. Train the model

```bash
python scripts/train.py --config configs/50m.yaml
```

#### 5. Generate text

```bash
python scripts/generate.py \
    --checkpoint checkpoints/50m/best.pt \
    --prompt "The future of artificial intelligence" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9
```

---

## Hugging Face Model

The trained model and tokenizer are published on Hugging Face:

**[cannizaroo/tiny-llm-50m](https://huggingface.co/cannizaroo/tiny-llm-50m)**

The repository contains the exported PyTorch weights, configuration, tokenizer files, and model card.

Because this is a custom PyTorch implementation, the model is currently intended to be loaded using the project's own `GPTConfig` and `GPT` classes rather than `AutoModelForCausalLM`.

Example:

```python
from huggingface_hub import hf_hub_download
import json
import torch

repo_id = "cannizaroo/tiny-llm-50m"

config_path = hf_hub_download(
    repo_id=repo_id,
    filename="config.json",
)

weights_path = hf_hub_download(
    repo_id=repo_id,
    filename="pytorch_model.bin",
)

with open(config_path) as f:
    cfg_dict = json.load(f)

config = GPTConfig(**cfg_dict)
model = GPT(config)

state_dict = torch.load(
    weights_path,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()
```

The `GPTConfig` and `GPT` implementations are available in this repository.

---

## Development Phases

| Phase | What gets built |
|---|---|
| 1 | Repository scaffold, configs, BPE tokenizer |
| 2 | Dataset analysis and preprocessing pipeline |
| 3 | Token + positional embeddings |
| 4 | Single-head causal self-attention |
| 5 | Multi-head attention |
| 6 | MLP, LayerNorm, residual connections, Transformer block |
| 7 | Full GPT model + language-model loss |
| 8 | Tiny-batch overfit test |
| 9 | Small FineWeb-Edu run |
| 10 | ~50M model training run |
| 11 | Generation + evaluation |
| 12 | Profiling + documentation |

---

## Design Principles

### Built for understanding

The implementation is intentionally modular so that each major Transformer component can be inspected independently.

### Tensor shapes matter

The project documents tensor transformations throughout the model:

```text
Input IDs
(B, T)

Embeddings
(B, T, C)

Q / K / V
(B, H, T, D)

Attention scores
(B, H, T, T)

Attention output
(B, H, T, D)

Merged heads
(B, T, C)

Final logits
(B, T, V)
```

### No nanoGPT copy-paste

[nanoGPT](https://github.com/karpathy/nanoGPT) was used only as an implementation reference when necessary. The project was independently structured and implemented to understand the architecture rather than reproduce an existing codebase.

---

## Limitations

This model is intentionally small and was trained on only ~20M tokens.

Therefore:

- It is not intended to compete with production language models.
- Generated text quality is limited by the small model size and training budget.
- The model should be treated as an educational experiment and implementation reference.

---

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original Transformer paper
2. [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. [PyTorch Documentation](https://pytorch.org/docs/)
4. [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers)
5. [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
6. [nanoGPT](https://github.com/karpathy/nanoGPT) — implementation reference

---

## License

MIT
