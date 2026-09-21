"""
tests/test_overfit_batch.py
───────────────────────────
Phase 8: Single-batch tiny dataset overfitting test.

─────────────────────────────────────────────────────────────────────────────
PURPOSE OF THIS TEST
─────────────────────────────────────────────────────────────────────────────
Before attempting multi-million token pre-training, we verify that the model,
loss function, embeddings, causal self-attention, and optimizer can drive
cross-entropy loss to near zero on a single repeated sequence.

If a neural network cannot overfit a single batch, it has an architectural bug
(e.g., broken attention masking, improper gradient detaching, broken residual connections).
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gpt import GPT, GPTConfig
from src.training.optimizer import configure_optimizers


def test_overfit_single_batch():
    """Verify that the model drives training loss below 0.1 on a single fixed batch."""
    torch.manual_seed(42)

    # 1. Architecture setup
    config = GPTConfig(
        vocab_size=128,
        context_length=32,
        embedding_dim=64,
        num_layers=2,
        num_heads=2,
        ffn_multiplier=4,
        dropout=0.0,
        weight_tying=False,
    )
    model = GPT(config)
    model.train()

    # 2. Optimizer setup
    optimizer = configure_optimizers(model, learning_rate=1e-3, weight_decay=0.0)

    # 3. Create a single fixed synthetic batch: shape (B=2, T=32)
    B, T = 2, 32
    # Repeating token pattern
    x = torch.randint(0, config.vocab_size, (B, T), dtype=torch.long)
    # Target is x shifted by 1
    y = torch.roll(x, shifts=-1, dims=-1)

    initial_loss = None
    final_loss = None

    # 4. Train on this single batch for 200 iterations
    for step in range(200):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"\n[Overfit Test] Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")

    # Loss for random predictions on vocab=128 is ln(128) ≈ 4.85
    # The model should easily memorize this single batch and drive loss < 0.1
    assert final_loss < 0.1, f"Model failed to overfit single batch! Final loss: {final_loss:.4f}"
    print("[PASS] Single batch successfully overfit!")


if __name__ == "__main__":
    test_overfit_single_batch()
