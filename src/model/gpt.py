"""
src/model/gpt.py
────────────────
Full GPT (Generative Pre-trained Transformer) decoder-only model.

─────────────────────────────────────────────────────────────────────────────
WHAT IS THE GPT MODEL?
─────────────────────────────────────────────────────────────────────────────
The complete architecture brings together all components:
  1. Embeddings: Token IDs (B, T) -> (B, T, C)
  2. Transformer Blocks: N stacked blocks with Pre-LayerNorm & Residuals (B, T, C) -> (B, T, C)
  3. Final LayerNorm: (B, T, C) -> (B, T, C)
  4. Language Modeling Head: Linear projection (B, T, C) -> (B, T, vocab_size)
  5. Cross-Entropy Loss: Optional loss calculation against shifted target tokens.

─────────────────────────────────────────────────────────────────────────────
WEIGHT TYING (OPTION)
─────────────────────────────────────────────────────────────────────────────
In standard GPT-2, the weights of the input token embedding matrix and the
output linear LM head are tied (shared):
  lm_head.weight = token_embedding.weight
This reduces parameter count by V*C (~8.4M params for 16k vocab & 512 dim).
We make weight tying configurable:
  - If weight_tying=False: Head has independent weights -> ~36M params (Option A)
  - If weight_tying=True: Head shares embedding weights -> ~27M params
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.embeddings import GPTEmbeddings
from src.model.transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    """Configuration class for GPT model architecture."""
    vocab_size: int = 16384
    context_length: int = 256
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.0
    weight_tying: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTConfig":
        """Create GPTConfig from a dictionary (e.g. loaded from YAML)."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


class GPT(nn.Module):
    """
    GPT Language Model.

    Parameters
    ----------
    config : GPTConfig
        Architecture configuration hyperparameters.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 1. Token & Positional Embeddings
        self.embeddings = GPTEmbeddings(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
        )

        # 2. Stack of N Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                context_length=config.context_length,
                ffn_multiplier=config.ffn_multiplier,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # 3. Final LayerNorm before language model head
        self.ln_f = nn.LayerNorm(config.embedding_dim)

        # 4. Language Model Head (projects hidden states to vocabulary logits)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Optional Weight Tying (GPT-2 style)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.token_embedding.weight

        # 5. Initialize weights properly (Gaussian with scaled std for residual projections)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize network weights according to standard GPT conventions:
          - Linear layers: N(0, 0.02)
          - Embeddings: N(0, 0.02)
          - LayerNorm: weights = 1.0, biases = 0.0
        """
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

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the GPT model.

        Parameters
        ----------
        idx : torch.Tensor
            Input token indices of shape (B, T).
        targets : Optional[torch.Tensor]
            Target token indices of shape (B, T) for computing cross-entropy loss.

        Returns
        -------
        logits : torch.Tensor
            Unnormalized vocabulary log-probabilities of shape (B, T, vocab_size).
        loss : Optional[torch.Tensor]
            Scalar cross-entropy loss if targets was provided, otherwise None.
        """
        B, T = idx.shape

        # Step 1: Compute token + positional embeddings -> (B, T, C)
        x = self.embeddings(idx)

        # Step 2: Pass through each Transformer block sequentially -> (B, T, C)
        for block in self.blocks:
            x = block(x)

        # Step 3: Apply final LayerNorm -> (B, T, C)
        x = self.ln_f(x)

        # Step 4: Compute logits via LM Head -> (B, T, V)
        logits = self.lm_head(x)

        # Step 5: Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten predictions and targets for cross_entropy:
            # logits: (B*T, V), targets: (B*T)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    def count_parameters(self) -> Dict[str, int]:
        """
        Calculate and return parameter counts broken down by component.

        Returns
        -------
        dict with:
          - total_params: Total parameters in the model
          - trainable_params: Total trainable parameters
          - token_embedding_params: Parameters in token embedding
          - pos_embedding_params: Parameters in positional embedding
          - transformer_block_params: Parameters across all blocks
          - final_ln_params: Parameters in final LayerNorm
          - lm_head_params: Parameters in output projection head
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        tok_emb = sum(p.numel() for p in self.embeddings.token_embedding.parameters())
        pos_emb = sum(p.numel() for p in self.embeddings.position_embedding.parameters())
        blocks = sum(p.numel() for p in self.blocks.parameters())
        ln_f = sum(p.numel() for p in self.ln_f.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())

        return {
            "total_params": total,
            "trainable_params": trainable,
            "token_embedding_params": tok_emb,
            "pos_embedding_params": pos_emb,
            "transformer_block_params": blocks,
            "final_ln_params": ln_f,
            "lm_head_params": lm_head,
        }

    def print_parameter_summary(self) -> None:
        """Print a formatted breakdown of model parameter counts."""
        p = self.count_parameters()
        print("─" * 60)
        print("  GPT Model Parameter Summary")
        print("─" * 60)
        print(f"  Token Embeddings       : {p['token_embedding_params']:>12,}")
        print(f"  Positional Embeddings  : {p['pos_embedding_params']:>12,}")
        print(f"  Transformer Blocks ({self.config.num_layers}) : {p['transformer_block_params']:>12,}")
        print(f"  Final LayerNorm        : {p['final_ln_params']:>12,}")
        print(f"  LM Head (Output Proj)  : {p['lm_head_params']:>12,}")
        print("─" * 60)
        print(f"  TOTAL PARAMETERS       : {p['total_params']:>12,} ({p['total_params']/1e6:.2f} M)")
        print(f"  Trainable Parameters   : {p['trainable_params']:>12,}")
        print("─" * 60)
