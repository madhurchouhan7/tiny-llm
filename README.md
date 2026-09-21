# Tiny-LLM-From-Scratch

A GPT-2-style decoder-only Transformer language model built **from scratch** in PyTorch.

> **Goal**: Understand how a large language model works at the tensor/code level —
> from raw text all the way to generated output — not simply obtain a working model.

---

## Architecture at a Glance

```
Raw text
   ↓  BPE tokenizer (vocab = 16 384)
Token IDs  (B, T)
   ↓  Token embedding  +  Positional embedding
(B, T, 512)
   ↓  × 6 Transformer Blocks
      ├── LayerNorm → Causal Multi-Head Attention → Residual
      └── LayerNorm → Feed-Forward (MLP) → Residual
(B, T, 512)
   ↓  Final LayerNorm
   ↓  Linear LM Head
Logits  (B, T, 16384)
   ↓  Cross-entropy loss  /  Sampling
Next token
```

**Target size**: ~36–38 M parameters  
**Configuration**: `embedding_dim=512`, `num_heads=8`, `num_layers=6`, `context_length=256`

---

## Project Structure

```
tiny-llm-from-scratch/
├── configs/            # YAML hyperparameter files
│   ├── debug.yaml      # Tiny model for fast local testing
│   ├── tiny.yaml       # Overfitting / 1-5 M token run
│   └── 50m.yaml        # Full ~36M-param model
├── src/
│   ├── tokenizer/      # BPE tokenizer wrapper
│   ├── data/           # Preprocessing, dataset, dataloader
│   ├── model/          # Embeddings, attention, MLP, block, GPT
│   ├── training/       # Trainer, optimizer, checkpointing
│   └── generation.py   # Text generation & sampling
├── scripts/            # Runnable entry points
├── tests/              # Pytest unit tests
├── experiments/        # Logs & generated samples
├── checkpoints/        # Saved model/optimizer states
├── artifacts/          # Tokenizer vocabulary files
└── docs/               # Architecture & tensor-shape docs
```

---

## Quickstart

### 🚀 Google Colab (One-Click Pre-training & Hugging Face Publish)
Open and run [`train_and_publish_colab.ipynb`](file:///G:/Machine%20Learning/tiny-llm/train_and_publish_colab.ipynb) in Google Colab (with free T4 GPU). It contains the entire self-contained pipeline from streaming FineWeb-Edu to pushing the final model and tokenizer to your Hugging Face account.

---

### Local Execution

#### 1. Install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Train the BPE tokenizer
```bash
python scripts/train_tokenizer.py
```

#### 3. Analyze & preprocess FineWeb-Edu
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

## Development Phases

| Phase | What gets built |
|-------|----------------|
| 1 | Repo scaffold, configs, BPE tokenizer |
| 2 | Dataset analysis, preprocessing pipeline |
| 3 | Token + positional embeddings |
| 4–6 | Attention, MLP, Transformer block |
| 7 | Full GPT model + loss |
| 8 | Tiny-batch overfit test |
| 9 | Small FineWeb-Edu run (1–5 M tokens) |
| 10 | Full 100–200 M token training run |
| 11 | Generation + evaluation scripts |
| 12 | Profiling + documentation |

---

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original Transformer paper
2. [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. [PyTorch documentation](https://pytorch.org/docs/)
4. [HuggingFace tokenizers](https://huggingface.co/docs/tokenizers)
5. [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
6. [nanoGPT](https://github.com/karpathy/nanoGPT) — used as an implementation *reference*, not copied
