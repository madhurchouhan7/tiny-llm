"""
src/model/embeddings.py
───────────────────────
Token and learnable Positional Embeddings for GPT architecture.

─────────────────────────────────────────────────────────────────────────────
WHAT ARE EMBEDDINGS?
─────────────────────────────────────────────────────────────────────────────
1. Token Embedding:
   - Takes integer token IDs of shape (B, T)
   - Looks up each token ID in a learnable weight matrix of shape (vocab_size, embedding_dim)
   - Outputs continuous representations of shape (B, T, embedding_dim)

2. Positional Embedding:
   - Transformers process all tokens in parallel and have no built-in notion of order or sequence.
   - Positional embeddings assign a unique learnable vector to each position index: 0, 1, 2, ..., T-1.
   - Positional embedding table has shape (context_length, embedding_dim).
   - For an input sequence of length T, we slice positions [0:T] and add them element-wise
     to the token embeddings.

Combined Output:
  x = TokenEmbedding(token_ids) + PositionalEmbedding(positions)
  Shape: (B, T, embedding_dim)
"""

import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    """
    Combined Token and Positional Embeddings layer.

    Parameters
    ----------
    vocab_size : int
        Size of the tokenizer vocabulary (e.g. 16384).
    context_length : int
        Maximum sequence length / context window size (e.g. 256).
    embedding_dim : int
        Hidden dimension / vector width (e.g. 512).
    dropout : float
        Dropout rate applied after adding token and position embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim

        # Token embedding lookup table: (V, C)
        # Given token index idx in [0, V-1], retrieves a C-dimensional vector.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Learnable positional embedding table: (T_max, C)
        # Given position index pos in [0, T_max-1], retrieves a C-dimensional vector.
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

        # Dropout for regularization during training
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute combined embeddings for a batch of token sequences.

        Parameters
        ----------
        idx : torch.Tensor
            Input token IDs of shape (B, T), where:
              - B is batch size
              - T is sequence length (T <= context_length)
            dtype must be torch.long (int64).

        Returns
        -------
        torch.Tensor
            Combined embeddings of shape (B, T, embedding_dim).
        """
        device = idx.device
        b, t = idx.shape

        if t > self.context_length:
            raise ValueError(
                f"Sequence length {t} exceeds maximum context length {self.context_length}"
            )

        # 1. Token embeddings: (B, T) -> (B, T, C)
        tok_emb = self.token_embedding(idx)

        # 2. Position indices: (T,) containing [0, 1, 2, ..., t-1]
        positions = torch.arange(0, t, dtype=torch.long, device=device)

        # 3. Position embeddings: (T,) -> (T, C)
        pos_emb = self.position_embedding(positions)

        # 4. Add token and position embeddings: (B, T, C) + (T, C) -> (B, T, C)
        # Broadcasting automatically adds pos_emb along the batch dimension B.
        x = tok_emb + pos_emb

        # 5. Apply dropout
        x = self.dropout(x)

        return x
