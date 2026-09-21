"""
src/training/checkpoint.py
──────────────────────────
Checkpoint save and resume utilities for Tiny-LLM training.

─────────────────────────────────────────────────────────────────────────────
WHAT GOES INTO A CHECKPOINT?
─────────────────────────────────────────────────────────────────────────────
To enable full training resumption without loss of state, each checkpoint contains:
  1. 'step': Global gradient update step count
  2. 'model_state_dict': All weights and registered buffers (causal mask, etc.)
  3. 'optimizer_state_dict': AdamW momentum and variance buffers
  4. 'best_val_loss': Lowest validation loss achieved so far
  5. 'config': Full dictionary configuration (model and training hyperparameters)
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
    val_loss: float,
    best_val_loss: float,
    config: Dict[str, Any],
    is_best: bool = False,
) -> str:
    """
    Save training state checkpoint to disk.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save checkpoints.
    step : int
        Current training step.
    model : nn.Module
        GPT model.
    optimizer : Optimizer
        AdamW optimizer.
    val_loss : float
        Current validation loss.
    best_val_loss : float
        Best validation loss seen so far.
    config : dict
        Training and model configuration.
    is_best : bool
        If True, also saves a copy to 'best.pt'.

    Returns
    -------
    str
        Path to the saved latest checkpoint file.
    """
    save_path = Path(checkpoint_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    latest_file = save_path / "latest.pt"
    torch.save(state, latest_file)

    if is_best:
        best_file = save_path / "best.pt"
        torch.save(state, best_file)
        print(f"[Checkpoint] Saved new best checkpoint (val_loss: {val_loss:.4f}) -> {best_file}")

    return str(latest_file)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load model and optimizer states from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint file.
    model : nn.Module
        GPT model to load weights into.
    optimizer : Optional[Optimizer]
        Optimizer to load state into (optional if only running inference).
    device : str
        Target device ("cpu" or "cuda").

    Returns
    -------
    dict
        Metadata including 'step', 'best_val_loss', 'config'.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Checkpoint] Loaded model weights from step {checkpoint['step']}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[Checkpoint] Restored optimizer state.")

    return {
        "step": checkpoint.get("step", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "val_loss": checkpoint.get("val_loss", None),
        "config": checkpoint.get("config", {}),
    }
