"""
src/model/mlp.py
────────────────
Feed-Forward Network (MLP) for GPT Transformer Block.

─────────────────────────────────────────────────────────────────────────────
WHAT IS THE FEED-FORWARD NETWORK?
─────────────────────────────────────────────────────────────────────────────
In a Transformer, the Multi-Head Attention layer enables tokens to communicate
and mix information across time/positions.
The Feed-Forward Network (MLP) operates on each token representation *independently*.

It applies a two-layer non-linear transformation:
  1. Expansion: Linear projection from embedding dimension C to 4*C
  2. Activation: Non-linear activation function GELU (Gaussian Error Linear Unit)
  3. Contraction: Linear projection from 4*C back down to C
  4. Dropout for regularization

Mathematical formulation:
  MLP(x) = Linear_2(GELU(Linear_1(x)))

Tensor shape progression:
  Input x:      (B, T, C)
  Linear_1:     (B, T, 4*C)
  GELU:         (B, T, 4*C)
  Linear_2:     (B, T, C)
  Dropout:      (B, T, C)
"""

import torch
import torch.nn as nn


class GPTMLP(nn.Module):
    """
    Transformer Feed-Forward Network (MLP).

    Parameters
    ----------
    embedding_dim : int
        Input and output hidden dimension C (e.g. 512).
    ffn_multiplier : int
        Multiplier for intermediate hidden dimension (standard is 4 -> 4*C = 2048).
    dropout : float
        Dropout probability applied after second linear projection.
    """

    def __init__(
        self,
        embedding_dim: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = embedding_dim * ffn_multiplier

        # Up-projection: C -> 4*C
        self.fc_in = nn.Linear(embedding_dim, hidden_dim, bias=True)

        # Activation function: GELU (standard in modern GPTs)
        self.act = nn.GELU()

        # Down-projection: 4*C -> C
        self.fc_out = nn.Linear(hidden_dim, embedding_dim, bias=True)

        # Dropout on the residual branch
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input activations of shape (B, T, C).

        Returns
        -------
        torch.Tensor
            Output activations of shape (B, T, C).
        """
        # (B, T, C) -> (B, T, 4*C)
        h = self.fc_in(x)
        # Apply non-linearity
        h = self.act(h)
        # (B, T, 4*C) -> (B, T, C)
        h = self.fc_out(h)
        # Apply dropout
        out = self.dropout(h)
        return out
