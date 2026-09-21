# Training Guide — Tiny-LLM-From-Scratch

## 1. Quick Start Workflow

### Step 1: Train BPE Tokenizer
Train a custom 16,384-vocabulary tokenizer from a sample of FineWeb-Edu:
```bash
python scripts/train_tokenizer.py --vocab-size 16384 --num-docs 50000
```

### Step 2: Preprocess Dataset Shards
Stream and tokenize FineWeb-Edu into binary memmap chunks (`train.bin`, `val.bin`):
```bash
# For small verification run:
python scripts/prepare_data.py --config configs/tiny.yaml

# For full 50M model run:
python scripts/prepare_data.py --config configs/50m.yaml
```

### Step 3: Run Model Training
Start training with automatic checkpointing and validation:
```bash
python scripts/train.py --config configs/50m.yaml
```

### Step 4: Text Generation & Sampling
Sample text from the trained checkpoint:
```bash
python scripts/generate.py \
    --checkpoint checkpoints/50m/best.pt \
    --prompt "Artificial intelligence enables" \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9
```

### Step 5: Validation Evaluation
Evaluate cross-entropy loss and perplexity on unseen test data:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/50m/best.pt \
    --val-bin data/processed/50m/val.bin
```

---

## 2. Hardware and Performance Recommendations

- **Local CPU**: Use `configs/debug.yaml` for testing code logic and forward/backward passes.
- **CUDA GPUs (RTX 3080/4090/A100)**: Enable `use_amp: true` to leverage Automatic Mixed Precision (FP16) and gradient accumulation for highest throughput.
