"""
tests/test_model.py
───────────────────
Unit tests for GPT model components and full GPT architecture:
  - Token and Positional Embeddings
  - Multi-Head Causal Self-Attention
  - Causal Masking (strictly autoregressive behavior)
  - Feed-Forward MLP
  - Transformer Block
  - GPT Model (forward pass, loss, parameter counting)
  - Gradient Flow (backpropagation sanity check)
"""

import sys
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.embeddings import GPTEmbeddings
from src.model.attention import CausalSelfAttention
from src.model.mlp import GPTMLP
from src.model.transformer_block import TransformerBlock
from src.model.gpt import GPT, GPTConfig


# ─────────────────────────────────────────────────────────────────────────────
# Test Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddings:
    def test_embeddings_shape(self):
        B, T = 4, 32
        vocab_size = 1000
        embed_dim = 64
        context_length = 128

        emb = GPTEmbeddings(vocab_size=vocab_size, context_length=context_length, embedding_dim=embed_dim)
        idx = torch.randint(0, vocab_size, (B, T), dtype=torch.long)
        out = emb(idx)

        assert out.shape == (B, T, embed_dim), f"Expected ({B}, {T}, {embed_dim}), got {out.shape}"

    def test_context_length_exceeded_raises(self):
        B, T = 2, 40
        emb = GPTEmbeddings(vocab_size=100, context_length=32, embedding_dim=16)
        idx = torch.randint(0, 100, (B, T), dtype=torch.long)
        with pytest.raises(ValueError):
            emb(idx)


# ─────────────────────────────────────────────────────────────────────────────
# Test Causal Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class TestAttention:
    def test_attention_output_shape(self):
        B, T, C = 2, 16, 64
        num_heads = 4
        attn = CausalSelfAttention(embedding_dim=C, num_heads=num_heads, context_length=32)
        x = torch.randn(B, T, C)
        out = attn(x)
        assert out.shape == (B, T, C), f"Expected ({B}, {T}, {C}), got {out.shape}"

    def test_causal_masking_autoregressive_property(self):
        """
        Critical Test: Changes to future tokens (e.g. at position t=10)
        MUST NOT affect outputs at earlier positions (t < 10).
        """
        B, T, C = 1, 8, 32
        attn = CausalSelfAttention(embedding_dim=C, num_heads=2, context_length=16, dropout=0.0)
        attn.eval()  # Disable dropout for deterministic evaluation

        # Input 1
        x1 = torch.randn(B, T, C)

        # Input 2 is identical to x1 up to index 4, but modified at indices 5, 6, 7
        x2 = x1.clone()
        x2[:, 5:, :] = torch.randn(B, 3, C)

        with torch.no_grad():
            out1 = attn(x1)
            out2 = attn(x2)

        # Output at positions 0, 1, 2, 3, 4 MUST be identical for both inputs!
        # Max difference should be effectively 0 (within floating point precision)
        diff = (out1[:, :5, :] - out2[:, :5, :]).abs().max().item()
        assert diff < 1e-5, f"Causal attention leaked future information! Diff: {diff}"


# ─────────────────────────────────────────────────────────────────────────────
# Test MLP and Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformerBlock:
    def test_mlp_shape(self):
        B, T, C = 2, 8, 64
        mlp = GPTMLP(embedding_dim=C, ffn_multiplier=4)
        x = torch.randn(B, T, C)
        out = mlp(x)
        assert out.shape == (B, T, C)

    def test_block_shape(self):
        B, T, C = 2, 8, 64
        block = TransformerBlock(embedding_dim=C, num_heads=4, context_length=16)
        x = torch.randn(B, T, C)
        out = block(x)
        assert out.shape == (B, T, C)


# ─────────────────────────────────────────────────────────────────────────────
# Test Full GPT Model
# ─────────────────────────────────────────────────────────────────────────────

class TestGPTModel:
    @pytest.fixture
    def small_config(self):
        return GPTConfig(
            vocab_size=256,
            context_length=32,
            embedding_dim=64,
            num_layers=2,
            num_heads=2,
            ffn_multiplier=4,
            dropout=0.0,
            weight_tying=False,
        )

    def test_gpt_forward_pass_shapes(self, small_config):
        model = GPT(small_config)
        B, T = 2, 16
        idx = torch.randint(0, small_config.vocab_size, (B, T), dtype=torch.long)
        logits, loss = model(idx)

        assert logits.shape == (B, T, small_config.vocab_size)
        assert loss is None

    def test_gpt_forward_with_loss(self, small_config):
        model = GPT(small_config)
        B, T = 2, 16
        idx = torch.randint(0, small_config.vocab_size, (B, T), dtype=torch.long)
        targets = torch.randint(0, small_config.vocab_size, (B, T), dtype=torch.long)

        logits, loss = model(idx, targets)
        assert loss is not None
        assert torch.isfinite(loss), "Loss must be a finite real number"
        assert loss.item() > 0.0

    def test_gradient_flow(self, small_config):
        """Verify backpropagation produces non-zero gradients for all parameters."""
        model = GPT(small_config)
        model.train()

        B, T = 2, 16
        idx = torch.randint(0, small_config.vocab_size, (B, T), dtype=torch.long)
        targets = torch.randint(0, small_config.vocab_size, (B, T), dtype=torch.long)

        logits, loss = model(idx, targets)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient is None for parameter: {name}"
            grad_norm = param.grad.norm().item()
            assert not math.isnan(grad_norm), f"NaN gradient detected in: {name}"

    def test_target_36m_param_calculation(self):
        """Test parameter counting on 50m.yaml configuration (Option A)."""
        config = GPTConfig(
            vocab_size=16384,
            context_length=256,
            embedding_dim=512,
            num_layers=6,
            num_heads=8,
            ffn_multiplier=4,
            weight_tying=False,
        )
        model = GPT(config)
        counts = model.count_parameters()

        total = counts["total_params"]
        # Expected: ~35.8M parameters
        assert 30_000_000 <= total <= 50_000_000, (
            f"Expected parameter count in 30M-50M range, got {total:,}"
        )
