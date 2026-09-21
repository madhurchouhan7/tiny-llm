# Architecture Documentation — Tiny-LLM-From-Scratch

## 1. Overview & High-Level Flow

Tiny-LLM is a decoder-only autoregressive language model based on the GPT-2 architecture. It takes raw text, converts it into a sequence of subword tokens using Byte-Pair Encoding (BPE), and predicts the probability distribution of the next token at every sequence position.

```
Input Tokens: (B, T)
      │
      ▼
┌──────────────────────────────────────────────┐
│  GPTEmbeddings                               │
│  - Token Embeddings:    (B, T) -> (B, T, C)  │
│  - Positional Embeddings: (T,) -> (T, C)     │
│  - Sum & Dropout:       (B, T, C)            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  TransformerBlock × N (num_layers = 6)       │
│                                              │
│    x ──────────────────────────────(+)────── │
│    │                                ▲        │
│    ▼                                │        │
│  LayerNorm_1                        │        │
│    ▼                                │        │
│  CausalSelfAttention                │        │
│    │                                │        │
│    └────────────────────────────────┘        │
│    │                                         │
│    x ──────────────────────────────(+)────── │
│    │                                ▲        │
│    ▼                                │        │
│  LayerNorm_2                        │        │
│    ▼                                │        │
│  GPTMLP (Linear -> GELU -> Linear)  │        │
│    │                                │        │
│    └────────────────────────────────┘        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Final LayerNorm: (B, T, C) -> (B, T, C)     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Linear LM Head:  (B, T, C) -> (B, T, V)     │
└──────────────────────────────────────────────┘
```

---

## 2. Mathematical Breakdown

### 2.1 Multi-Head Causal Self-Attention

Given input activation $X \in \mathbb{R}^{B \times T \times C}$:
1. **Linear Projections**:
   $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
   where $W_Q, W_K, W_V \in \mathbb{R}^{C \times C}$.
2. **Head Splitting**:
   Reshape $Q, K, V$ into $H$ attention heads of head dimension $D = C / H$:
   $$Q, K, V \in \mathbb{R}^{B \times H \times T \times D}$$
3. **Scaled Dot-Product with Causal Masking**:
   $$\text{Scores} = \frac{Q K^T}{\sqrt{D}} \in \mathbb{R}^{B \times H \times T \times T}$$
   $$\text{MaskedScores}_{i, j} = \begin{cases} \text{Scores}_{i, j} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
   $$\text{AttentionWeights} = \text{Softmax}(\text{MaskedScores}, \text{dim}=-1)$$
4. **Context Aggregation & Output Projection**:
   $$\text{Context} = \text{AttentionWeights} \cdot V \in \mathbb{R}^{B \times H \times T \times D}$$
   $$\text{Output} = \text{Concat}(\text{Context}_1, \dots, \text{Context}_H) W_O \in \mathbb{R}^{B \times T \times C}$$

### 2.2 Feed-Forward Network (MLP)

$$\text{MLP}(X) = \text{GELU}(X W_1 + b_1) W_2 + b_2$$
where $W_1 \in \mathbb{R}^{C \times 4C}$ and $W_2 \in \mathbb{R}^{4C \times C}$.

### 2.3 Cross-Entropy Objective

Given input sequence $x = (t_0, t_1, \dots, t_{T-1})$ and target sequence $y = (t_1, t_2, \dots, t_T)$, the training objective minimizes negative log-likelihood:
$$\mathcal{L}_{\text{CE}} = -\frac{1}{B \cdot T} \sum_{b=1}^B \sum_{t=1}^T \log P(y_{b,t} \mid x_{b, \le t})$$
Perplexity is calculated as:
$$\text{PPL} = \exp(\mathcal{L}_{\text{CE}})$$
