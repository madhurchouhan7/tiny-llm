"""
Script to generate the complete train_and_publish_colab.ipynb notebook.
"""

import json

notebook = {
    "cells": [],
    "metadata": {
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 2,
}

def add_md(source: str):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    })

def add_code(source: str):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    })

# Section 1: Overview
add_md(
"""# Tiny-LLM-From-Scratch (35.8M) Pre-Training & HuggingFace Publish Notebook
This notebook trains a **~35.8M parameter GPT-2 style decoder-only Transformer** from scratch on **FineWeb-Edu**, logs training metrics & perplexity, tests text generation with custom sampling, and publishes the final model and BPE tokenizer directly to the **Hugging Face Hub**.

### Target Architecture (Option A)
- **Parameters**: 35,823,616 (~35.8M)
- **Vocabulary**: 16,384 Byte-Level BPE tokens
- **Context Length ($T$)**: 256 tokens
- **Embedding Dim ($C$)**: 512
- **Heads ($H$)**: 8 ($D = 64$ per head)
- **Layers ($N$)**: 6 Transformer blocks
- **MLP**: Pre-LayerNorm Residuals + GELU activation ($C \\to 4C \\to C$)
- **Hardware**: GPU (T4 / V100 / A100 in Google Colab) with Automatic Mixed Precision (AMP FP16)
"""
)

# Section 2: Dependencies
add_md("## 1. Environment Setup & Dependencies")
add_code(
"""# Install required dependencies
!pip install -q tokenizers pyyaml tqdm datasets huggingface_hub

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
"""
)

# Section 3: Tokenizer
add_md(
"""## 2. BPE Tokenizer Implementation & Training on FineWeb-Edu
We implement and train a 16,384-vocabulary Byte-Level BPE tokenizer directly on streamed educational documents from FineWeb-Edu.
"""
)
add_code(
"""import os
import json
from pathlib import Path
from typing import List, Iterator, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

ENDOFTEXT_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"

class BPETokenizer:
    def __init__(self, vocab_size: int = 16_384):
        self.vocab_size = vocab_size
        self._tokenizer: Optional[Tokenizer] = None
        self.endoftext_id: Optional[int] = None
        self.pad_id: Optional[int] = None

    def train(self, texts: Iterator[str], show_progress: bool = True) -> None:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder(add_prefix_space=False)

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[ENDOFTEXT_TOKEN, PAD_TOKEN],
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
            show_progress=show_progress,
            min_frequency=2,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        self._tokenizer = tokenizer
        self._cache_special_token_ids()

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, directory: str) -> None:
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer.model.save(str(save_dir))
        self._tokenizer.save(str(save_dir / "full_tokenizer.json"))
        config = {
            "vocab_size": self.vocab_size,
            "endoftext_id": self.endoftext_id,
            "pad_id": self.pad_id,
            "endoftext_token": ENDOFTEXT_TOKEN,
            "pad_token": PAD_TOKEN,
        }
        with open(save_dir / "tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        load_dir = Path(directory)
        with open(load_dir / "tokenizer_config.json", "r") as f:
            config = json.load(f)
        instance = cls(vocab_size=config["vocab_size"])
        instance._tokenizer = Tokenizer.from_file(str(load_dir / "full_tokenizer.json"))
        instance.endoftext_id = config["endoftext_id"]
        instance.pad_id = config["pad_id"]
        return instance

    def _cache_special_token_ids(self) -> None:
        self.endoftext_id = self._tokenizer.token_to_id(ENDOFTEXT_TOKEN)
        self.pad_id = self._tokenizer.token_to_id(PAD_TOKEN)

    def get_vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

print("BPETokenizer defined.")
"""
)

add_md("### Train Tokenizer from FineWeb-Edu Stream")
add_code(
"""from datasets import load_dataset

def stream_fineweb_edu(num_docs=40_000):
    print(f"Streaming {num_docs:,} documents from FineWeb-Edu (sample-10BT)...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    count = 0
    for example in dataset:
        text = example.get("text", "")
        if text and len(text.strip()) > 100:
            yield text
            count += 1
            if count >= num_docs:
                break

tokenizer = BPETokenizer(vocab_size=16384)
tokenizer.train(texts=stream_fineweb_edu(num_docs=30_000))
tokenizer.save("artifacts/tokenizer")
print(f"Tokenizer trained and saved! Vocab size: {tokenizer.get_vocab_size():,}")
"""
)

# Section 4: Dataset Preprocessing
add_md(
"""## 3. Streaming Preprocessing & Binary Dataset Shards
Converts raw streamed text into memory-mapped uint16 binary shards: `train.bin` (98%) and `val.bin` (2%).
"""
)
add_code(
"""import numpy as np

def preprocess_fineweb(tokenizer, target_train_tokens=5_000_000, output_dir="data/processed"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_bin = open(out_path / "train.bin", "wb")
    val_bin = open(out_path / "val.bin", "wb")

    train_buf, val_buf = [], []
    train_tokens, val_tokens = 0, 0
    doc_idx = 0

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    print(f"Preprocessing target: {target_train_tokens:,} training tokens...")

    for example in dataset:
        text = example.get("text", "")
        if not text or len(text.strip()) < 100:
            continue

        token_ids = tokenizer.encode(text) + [tokenizer.endoftext_id]
        
        # 98% Train / 2% Validation split
        if doc_idx % 50 == 0:
            val_buf.extend(token_ids)
            val_tokens += len(token_ids)
        else:
            train_buf.extend(token_ids)
            train_tokens += len(token_ids)

        doc_idx += 1

        if len(train_buf) >= 500_000:
            np.array(train_buf, dtype=np.uint16).tofile(train_bin)
            train_buf.clear()
        if len(val_buf) >= 100_000:
            np.array(val_buf, dtype=np.uint16).tofile(val_bin)
            val_buf.clear()

        if doc_idx % 1000 == 0:
            print(f"Processed {doc_idx:,} docs | Train tokens: {train_tokens:,} / {target_train_tokens:,}")

        if train_tokens >= target_train_tokens:
            break

    if train_buf:
        np.array(train_buf, dtype=np.uint16).tofile(train_bin)
    if val_buf:
        np.array(val_buf, dtype=np.uint16).tofile(val_bin)

    train_bin.close()
    val_bin.close()
    print(f"Preprocessing complete! Train tokens: {train_tokens:,} | Val tokens: {val_tokens:,}")

# Change target_train_tokens to 20_000_000+ for longer training runs on Colab GPU
preprocess_fineweb(tokenizer, target_train_tokens=5_000_000)
"""
)

# Section 5: Model Architecture
add_md(
"""## 4. GPT-2 Style Model Architecture (~35.8M Parameters)
1. `GPTEmbeddings`: Token + learnable Positional embeddings
2. `CausalSelfAttention`: Multi-head causal self-attention with scaled dot-product and causal mask
3. `GPTMLP`: $C \\to 4C \\to C$ Feed-Forward layer with GELU
4. `TransformerBlock`: Pre-LayerNorm residual block
5. `GPT`: Full language model with LM Head and Cross-Entropy loss
"""
)
add_code(
"""import math
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

@dataclass
class GPTConfig:
    vocab_size: int = 16384
    context_length: int = 256
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.1
    weight_tying: bool = False

class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embedding_dim: int, dropout: float = 0.0):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        positions = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        return self.dropout(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        mask = torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att_scores = att_scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        out = att_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))

class GPTMLP(nn.Module):
    def __init__(self, embedding_dim: int, ffn_multiplier: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = embedding_dim * ffn_multiplier
        self.fc_in = nn.Linear(embedding_dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(hidden_dim, embedding_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.act(self.fc_in(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int, ffn_multiplier: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, context_length, dropout)
        self.ln_2 = nn.LayerNorm(embedding_dim)
        self.mlp = GPTMLP(embedding_dim, ffn_multiplier, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embeddings = GPTEmbeddings(config.vocab_size, config.context_length, config.embedding_dim, config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embedding_dim, config.num_heads, config.context_length, config.ffn_multiplier, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total:,} ({total/1e6:.2f} M)")
        return total

config = GPTConfig()
model = GPT(config)
model.count_parameters()
"""
)

# Section 6: DataLoader & Training Loop
add_md("## 5. Dataset Loader & Optimizer Setup")
add_code(
"""class TokenDataset(Dataset):
    def __init__(self, bin_path: str, context_length: int = 256):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        self.n_examples = max(0, len(self.data) - context_length - 1)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx: int):
        T = self.context_length
        chunk = torch.from_numpy(self.data[idx : idx + T + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]

def infinite_iter(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def configure_optimizers(model, learning_rate=3e-4, weight_decay=0.1, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay.append(param)
            else:
                no_decay.append(param)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=learning_rate, betas=betas)

class CosineWarmupScheduler:
    def __init__(self, optimizer, learning_rate, min_lr, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def step(self, step: int):
        if step < self.warmup_steps:
            lr = self.learning_rate * (step + 1) / max(1, self.warmup_steps)
        elif step > self.max_steps:
            lr = self.min_lr
        else:
            decay_ratio = (step - self.warmup_steps) / max(1, (self.max_steps - self.warmup_steps))
            lr = self.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (self.learning_rate - self.min_lr)

        for p in self.optimizer.param_groups:
            p["lr"] = lr
        return lr

print("Dataset loader & Optimizer configured.")
"""
)

# Section 7: Pre-Training Execution
add_md("## 6. Pre-Training Run with Automatic Mixed Precision (AMP)")
add_code(
"""import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device.upper()}")

model = GPT(config).to(device)
optimizer = configure_optimizers(model, learning_rate=3e-4, weight_decay=0.1)

max_steps = 1000
warmup_steps = 100
eval_interval = 100
batch_size = 16
grad_accum_steps = 2  # Effective batch size = 32

scheduler = CosineWarmupScheduler(optimizer, learning_rate=3e-4, min_lr=3e-5, warmup_steps=warmup_steps, max_steps=max_steps)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

train_ds = TokenDataset("data/processed/train.bin", context_length=config.context_length)
val_ds = TokenDataset("data/processed/val.bin", context_length=config.context_length)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
train_iter = infinite_iter(train_loader)

print("Starting training loop...")
model.train()
t0 = time.time()
best_val_loss = float("inf")

for step in range(1, max_steps + 1):
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for _ in range(grad_accum_steps):
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            _, loss = model(x, y)
            loss = loss / grad_accum_steps
        accum_loss += loss.item() * grad_accum_steps
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    lr = scheduler.step(step)

    if step % 25 == 0 or step == 1:
        elapsed = time.time() - t0
        tok_per_sec = (25 * batch_size * grad_accum_steps * config.context_length) / max(1e-5, elapsed)
        print(f"Step {step:>5}/{max_steps} | Train Loss: {accum_loss:.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s")
        t0 = time.time()

    if step % eval_interval == 0 or step == max_steps:
        model.eval()
        val_loss, count = 0.0, 0
        with torch.no_grad():
            for i, (vx, vy) in enumerate(val_loader):
                if i >= 30:
                    break
                vx, vy = vx.to(device), vy.to(device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    _, vloss = model(vx, vy)
                val_loss += vloss.item()
                count += 1
        val_loss /= max(1, count)
        ppl = math.exp(min(val_loss, 20.0))
        print(f"--> [Eval Step {step}] Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")

        # Save Best Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "val_loss": val_loss,
                "step": step,
            }, "checkpoints/best.pt")
            print(f"Saved best model checkpoint to checkpoints/best.pt")
        model.train()

print("Pre-training finished!")
"""
)

# Section 8: Generation & Sampling
add_md(
"""## 7. Text Generation & Custom Sampling
Test the trained model with Temperature, Top-K, and Nucleus (Top-P) sampling.
"""
)
add_code(
"""@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9, device="cuda"):
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.endoftext_id]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.context_length else idx[:, -model.config.context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

        if next_token.item() == tokenizer.endoftext_id:
            break

    return tokenizer.decode(idx[0].tolist())

test_prompt = "Artificial intelligence and machine learning"
print("Prompt:", test_prompt)
print("-" * 60)
print(generate_text(model, tokenizer, test_prompt, max_new_tokens=120, temperature=0.8, top_k=50, top_p=0.9, device=device))
print("-" * 60)
"""
)

# Section 9: Publish to HuggingFace
add_md(
"""## 8. Publish Model & Tokenizer to Hugging Face Hub
Authenticate with your Hugging Face write token and publish your model weights, config, tokenizer, and model card directly to your Hugging Face account!
"""
)
add_code(
"""from huggingface_hub import HfApi, login, create_repo
import shutil

# 1. Login to Hugging Face (Get write token: https://huggingface.co/settings/tokens)
print("Please enter your Hugging Face write token:")
login()

# 2. Set your Hugging Face repo ID (e.g., 'your-username/tiny-llm-35m')
REPO_ID = input("Enter target repo ID (e.g., username/tiny-llm-35m): ").strip()
create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")

# 3. Save export bundle
export_dir = Path("export_hf")
export_dir.mkdir(parents=True, exist_ok=True)

# Save weights & config
torch.save(model.state_dict(), export_dir / "pytorch_model.bin")
with open(export_dir / "config.json", "w") as f:
    json.dump(config.__dict__, f, indent=2)

# Copy tokenizer files
for tok_file in ["full_tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"]:
    src = Path("artifacts/tokenizer") / tok_file
    if src.exists():
        shutil.copy(src, export_dir / tok_file)

# 4. Create README Model Card
model_card = f\"\"\"---
language:
- en
license: mit
tags:
- gpt2
- tiny-llm
- educational
- transformers-from-scratch
---

# Tiny-LLM-From-Scratch (35.8M)

A 35.8M parameter GPT-2 style decoder-only Transformer built from scratch in PyTorch and pre-trained on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

## Architecture Details
- **Architecture**: Decoder-only Transformer (Pre-LayerNorm)
- **Parameters**: 35,823,616
- **Context Length**: 256 tokens
- **Vocabulary**: 16,384 BPE tokens
- **Embedding Dimension**: 512
- **Heads**: 8 (head_dim = 64)
- **Layers**: 6
- **FFN Multiplier**: 4 (hidden dim = 2048)

## Sampling Capabilities
- Autoregressive generation with Temperature, Top-K, and Nucleus (Top-P) sampling.
\"\"\"

with open(export_dir / "README.md", "w") as f:
    f.write(model_card)

# 5. Upload everything to Hugging Face
api = HfApi()
api.upload_folder(
    folder_path=str(export_dir),
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"\\nModel successfully uploaded to Hugging Face Hub!")
print(f"Check it out at: https://huggingface.co/{REPO_ID}")
"""
)

with open("train_and_publish_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created train_and_publish_colab.ipynb successfully!")
