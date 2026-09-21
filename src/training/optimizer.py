"""
src/training/optimizer.py
─────────────────────────
AdamW Optimizer with weight decay separation and Cosine Learning Rate Schedule.

─────────────────────────────────────────────────────────────────────────────
WEIGHT DECAY SEPARATION
─────────────────────────────────────────────────────────────────────────────
In standard GPT / Transformer training (nanoGPT, GPT-2, Llama):
  - 2D weight matrices (Linear layer weights, Embedding weights) receive weight decay.
  - 1D biases and LayerNorm scales/biases DO NOT receive weight decay.

Why?
  Decaying bias terms and LayerNorm scales hurts generalization and convergence.
  We inspect every parameter and group them into decayed and non-decayed sets.

─────────────────────────────────────────────────────────────────────────────
LEARNING RATE SCHEDULE: LINEAR WARMUP + COSINE DECAY
─────────────────────────────────────────────────────────────────────────────
1. Warmup Phase (step < warmup_steps):
   LR increases linearly from 0 to peak learning_rate:
   lr = peak_lr * (step + 1) / warmup_steps

2. Cosine Decay Phase (warmup_steps <= step <= max_steps):
   LR follows a cosine curve decaying from peak_lr down to min_lr:
   progress = (step - warmup_steps) / (max_steps - warmup_steps)
   lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))

3. Post-decay Phase (step > max_steps):
   lr = min_lr
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW


def configure_optimizers(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """
    Configure AdamW optimizer with weight decay applied ONLY to 2D tensors.

    Parameters
    ----------
    model : nn.Module
        GPT model instance.
    learning_rate : float
        Peak learning rate.
    weight_decay : float
        Weight decay coefficient for 2D matrices.
    betas : Tuple[float, float]
        Adam beta parameters (beta1, beta2).

    Returns
    -------
    AdamW
        Configured PyTorch AdamW optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 2D tensors (Linear weights, Embedding weights) -> Apply Weight Decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            # 1D tensors (Biases, LayerNorm scale and bias) -> No Weight Decay
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in no_decay_params)
    print(f"[Optimizer] Configured AdamW: {len(decay_params)} decayed tensors ({num_decay:,} params), "
          f"{len(no_decay_params)} non-decayed tensors ({num_nodecay:,} params)")

    return AdamW(optim_groups, lr=learning_rate, betas=betas, fused=False)


class CosineWarmupScheduler:
    """
    Learning Rate Scheduler with linear warmup and cosine decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be updated.
    learning_rate : float
        Peak learning rate.
    min_lr : float
        Minimum learning rate at the end of cosine decay.
    warmup_steps : int
        Number of steps for linear warmup.
    max_steps : int
        Total number of steps for the cosine decay cycle.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        """Calculate learning rate for the given training step."""
        # 1. Linear Warmup
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / max(1, self.warmup_steps)

        # 2. Post-max steps: floor at min_lr
        if step > self.max_steps:
            return self.min_lr

        # 3. Cosine Decay
        decay_ratio = (step - self.warmup_steps) / max(1, (self.max_steps - self.warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def step(self, step: int) -> float:
        """Update optimizer's learning rate for the current step and return current LR."""
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
