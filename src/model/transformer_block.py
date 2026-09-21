"""
src/model/transformer_block.py
──────────────────────────────
GPT-2 Pre-LayerNorm Transformer Block.

─────────────────────────────────────────────────────────────────────────────
WHAT IS A TRANSFORMER BLOCK?
─────────────────────────────────────────────────────────────────────────────
A standard GPT-2 Transformer Block consists of two main sub-layers with
residual connections and pre-normalization:

  1. Multi-Head Causal Self-Attention (with pre-LayerNorm)
     x = x + Attention(LayerNorm1(x))

  2. Feed-Forward MLP (with pre-LayerNorm)
     x = x + MLP(LayerNorm2(x))

─────────────────────────────────────────────────────────────────────────────
PRE-NORM VS POST-NORM
─────────────────────────────────────────────────────────────────────────────
Original Transformer (Vaswani et al., 2017) used Post-LayerNorm:
  x = LayerNorm(x + Sublayer(x))

GPT-2 (Radford et al., 2019) switched to Pre-LayerNorm:
  x = x + Sublayer(LayerNorm(x))

Why Pre-Norm?
  - Pre-norm allows gradients to flow directly along the residual stream
    without being scaled down or distorted by layer normalization at every step.
  - This significantly improves training stability and makes deep networks
    (like 12 to 96 layers) much easier to optimize without complex learning rate warmups.

Tensor shape is strictly preserved:
  Input:  (B, T, C)
  Output: (B, T, C)
"""

import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention
from src.model.mlp import GPTMLP


class TransformerBlock(nn.Module):
    """
    A single Pre-LayerNorm Transformer block.

    Parameters
    ----------
    embedding_dim : int
        Hidden dimension C (e.g. 512).
    num_heads : int
        Number of attention heads H (e.g. 8).
    context_length : int
        Maximum sequence length T_max (e.g. 256).
    ffn_multiplier : int
        Expansion multiplier for MLP hidden dimension (default: 4 -> 4*C).
    dropout : float
        Dropout probability applied in attention and MLP.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Layer normalization applied before Attention
        self.ln_1 = nn.LayerNorm(embedding_dim)

        # Multi-head causal self-attention
        self.attn = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
        )

        # Layer normalization applied before MLP
        self.ln_2 = nn.LayerNorm(embedding_dim)

        # Feed-forward network (MLP)
        self.mlp = GPTMLP(
            embedding_dim=embedding_dim,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-norm residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input activations of shape (B, T, C).

        Returns
        -------
        torch.Tensor
            Output activations of shape (B, T, C).
        """
        # 1. Causal Self-Attention sub-layer with Pre-Norm and Residual Connection
        # x = x + Attention(LayerNorm(x))
        x = x + self.attn(self.ln_1(x))

        # 2. Feed-Forward MLP sub-layer with Pre-Norm and Residual Connection
        # x = x + MLP(LayerNorm(x))
        x = x + self.mlp(self.ln_2(x))

        return x
