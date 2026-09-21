# Tensor Shapes Through the Model

> Notation used throughout this document:
> - **B** = batch size (number of sequences processed in parallel)
> - **T** = sequence length = context_length = 256
> - **C** = embedding_dim = 512
> - **H** = num_heads = 8
> - **D** = head_dim = C / H = 64
> - **V** = vocab_size = 16 384
> - **4C** = FFN hidden dim = 2048

---

## Input

```
token_ids: (B, T)     ← integers in [0, V)
```

---

## Embedding Layer

```
token_embed:  (B, T) → (B, T, C)    nn.Embedding(V, C)
pos_embed:    (T,)   → (T, C)       nn.Embedding(T, C)
                                     broadcast-adds to (B, T, C)
x:            (B, T, C)
```

---

## Inside Each Transformer Block

### LayerNorm 1
```
x_norm: (B, T, C)    same shape as input
```

### Q, K, V Projections (Causal Self-Attention)
```
# Combined QKV linear: C → 3C
qkv:     (B, T, 3C)  = (B, T, 1536)

# Split into Q, K, V
Q, K, V: each (B, T, C) = (B, T, 512)

# Reshape to separate attention heads
Q, K, V: each (B, H, T, D) = (B, 8, T, 64)
```

### Attention Scores
```
scores: Q @ K^T / sqrt(D)
      = (B, H, T, D) @ (B, H, D, T)
      = (B, H, T, T)   ← one score per (query position, key position)
```

### Causal Mask
```
# Upper triangle set to -inf so softmax → 0
mask:    (T, T)   upper-triangular boolean
scores:  (B, H, T, T)   with -inf above diagonal
```

### Attention Weights
```
weights: softmax(scores) → (B, H, T, T)
         each row sums to 1.0
```

### Weighted Sum (Context Vectors)
```
context: weights @ V
       = (B, H, T, T) @ (B, H, T, D)
       = (B, H, T, D)
```

### Merge Heads
```
# Transpose and reshape to merge all heads
context: (B, H, T, D) → (B, T, H*D) = (B, T, C)
```

### Output Projection
```
out:  Linear(C, C)
      (B, T, C) → (B, T, C)
```

### Residual Connection 1
```
x = x + out    # (B, T, C) + (B, T, C) = (B, T, C)
```

---

### LayerNorm 2
```
x_norm: (B, T, C)
```

### MLP (Feed-Forward Network)
```
h:  Linear(C, 4C)  → (B, T, 4C) = (B, T, 2048)
h:  GELU(h)        → (B, T, 4C)   non-linearity
x:  Linear(4C, C)  → (B, T, C)
```

### Residual Connection 2
```
x = x + x_mlp    # (B, T, C)
```

---

## After All Blocks

```
x:      (B, T, C)    after 6 Transformer blocks
x_norm: (B, T, C)    after final LayerNorm
logits: (B, T, V)    after LM Head Linear(C, V)
```

---

## Loss Computation

```
logits:       (B, T, V)
targets:      (B, T)       = input shifted by 1

# Flatten for cross_entropy
logits_flat:  (B*T, V)
targets_flat: (B*T,)

loss: scalar = cross_entropy(logits_flat, targets_flat)
```
