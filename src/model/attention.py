"""
src/model/attention.py
──────────────────────
Multi-Head Causal Self-Attention implemented manually with PyTorch tensor ops.

─────────────────────────────────────────────────────────────────────────────
WHAT IS CAUSAL SELF-ATTENTION?
─────────────────────────────────────────────────────────────────────────────
In an autoregressive language model (like GPT), when predicting the token at
position `t`, the model is only allowed to attend to positions `0, 1, ..., t`.
It MUST NOT look ahead at future tokens `t+1, t+2, ...`.

This constraint is enforced mathematically using a **causal mask**:
  1. We project input X of shape (B, T, C) into Query (Q), Key (K), Value (V).
  2. We split Q, K, V into H attention heads of dimension D = C / H.
     Shape of Q, K, V: (B, H, T, D)
  3. We compute raw attention scores via scaled dot-product:
     Scores = (Q @ K.transpose(-2, -1)) / sqrt(D)
     Shape: (B, H, T, T)
  4. We apply a causal mask: any score where key_position > query_position
     is replaced with -infinity.
  5. Softmax along the last dimension:
     Weights = softmax(Scores, dim=-1)
     Since exp(-inf) = 0, future positions get exactly 0.0 attention weight!
  6. Compute context output:
     Context = Weights @ V
     Shape: (B, H, T, D)
  7. Concatenate all H heads and project back through a linear layer:
     Out = Linear(Context.transpose(1, 2).reshape(B, T, C))
     Shape: (B, T, C)

─────────────────────────────────────────────────────────────────────────────
TENSOR SHAPE SUMMARY
─────────────────────────────────────────────────────────────────────────────
Input x:                 (B, T, C)
QKV projection:          (B, T, 3*C)
Split into Q, K, V:      each (B, T, C)
Reshape into heads:      each (B, H, T, D) where D = C / H
Attention Scores:        (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
Masked Softmax Weights:  (B, H, T, T)
Context Vectors:         (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
Merged Heads:            (B, T, H*D) = (B, T, C)
Output Projection:       (B, T, C) -> (B, T, C)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention module.

    Parameters
    ----------
    embedding_dim : int
        Total hidden dimension C (e.g. 512).
    num_heads : int
        Number of attention heads H (e.g. 8). Must divide embedding_dim evenly.
    context_length : int
        Maximum sequence length T_max (e.g. 256) for creating the causal mask buffer.
    dropout : float
        Dropout probability applied to attention weights and output projection.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.context_length = context_length

        # Combined Q, K, V linear projection in a single weight matrix for efficiency.
        # Maps (B, T, C) -> (B, T, 3*C).
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)

        # Output projection linear layer: Maps merged heads (B, T, C) -> (B, T, C)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer: lower-triangular matrix of 1s of size (context_length, context_length).
        # We register it as a buffer so it is:
        #   1. Saved and loaded with model state_dict
        #   2. Moved automatically to whatever device (CPU/GPU) the model is on
        #   3. NOT treated as a trainable parameter (requires_grad=False)
        mask = torch.tril(torch.ones(context_length, context_length)).view(
            1, 1, context_length, context_length
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Multi-Head Causal Self-Attention.

        Parameters
        ----------
        x : torch.Tensor
            Input activations of shape (B, T, C).

        Returns
        -------
        torch.Tensor
            Attended output tensor of shape (B, T, C).
        """
        B, T, C = x.shape

        # 1. Project input to Q, K, V in one matrix multiplication:
        # (B, T, C) -> (B, T, 3*C)
        qkv = self.qkv_proj(x)

        # 2. Split into separate Query, Key, Value tensors:
        # each will have shape (B, T, C)
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. Reshape and transpose for multi-head attention:
        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. Scaled dot-product attention:
        # q: (B, H, T, D), k.transpose(-2, -1): (B, H, D, T)
        # att_scores: (B, H, T, T)
        scale = 1.0 / math.sqrt(self.head_dim)
        att_scores = (q @ k.transpose(-2, -1)) * scale

        # 5. Apply causal mask:
        # We slice the precomputed causal_mask buffer up to sequence length T.
        # Wherever mask == 0 (future tokens), fill with -infinity.
        mask_slice = self.causal_mask[:, :, :T, :T]
        att_scores = att_scores.masked_fill(mask_slice == 0, float("-inf"))

        # 6. Softmax over key dimension (last dimension) to get probabilities:
        # Each row sums to 1.0.
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        # 7. Weighted sum of values:
        # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        out = att_weights @ v

        # 8. Concatenate heads back into original dimension:
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 9. Output projection and residual dropout:
        # (B, T, C) -> (B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out
