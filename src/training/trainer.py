"""
src/training/trainer.py
───────────────────────
Full-featured Trainer for Tiny-LLM pre-training.

─────────────────────────────────────────────────────────────────────────────
KEY FEATURES
─────────────────────────────────────────────────────────────────────────────
  1. Gradient Accumulation: Simulate large batch sizes with constrained VRAM.
  2. Mixed-Precision (AMP): Automatic torch.cuda.amp.autocast for fast FP16 execution on CUDA.
  3. Gradient Clipping: Prevent exploding gradients by clipping norm to grad_clip (e.g. 1.0).
  4. Periodic Validation: Evaluate cross-entropy loss and Perplexity = exp(loss).
  5. Resumable Checkpointing: Save/load optimizer, model, scheduler, and step states.
  6. Rich Logging: Step, loss, val_loss, perplexity, learning rate, tokens/sec, throughput.
"""

import time
import math
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.gpt import GPT, GPTConfig
from src.data.dataset import TokenDataset
from src.data.dataloader import make_dataloader, infinite_iter
from src.training.optimizer import configure_optimizers, CosineWarmupScheduler
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.generation import generate_text
from src.tokenizer import BPETokenizer


class Trainer:
    """
    Trainer class managing training, validation, checkpointing, and logging.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_cfg = config["model"]
        self.train_cfg = config["training"]
        self.paths_cfg = config["paths"]

        # Device Selection (CUDA if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Trainer] Using compute device: {self.device.upper()}")

        # Initialize GPT Model
        gpt_config = GPTConfig.from_dict(self.model_cfg)
        self.model = GPT(gpt_config).to(self.device)
        self.model.print_parameter_summary()

        # Optimizer & Scheduler
        self.optimizer = configure_optimizers(
            self.model,
            learning_rate=float(self.train_cfg["learning_rate"]),
            weight_decay=float(self.train_cfg["weight_decay"]),
            betas=(float(self.train_cfg["beta1"]), float(self.train_cfg["beta2"])),
        )

        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            learning_rate=float(self.train_cfg["learning_rate"]),
            min_lr=float(self.train_cfg["min_lr"]),
            warmup_steps=int(self.train_cfg["warmup_steps"]),
            max_steps=int(self.train_cfg["max_steps"]),
        )

        # Automatic Mixed Precision (AMP) Scaler
        self.use_amp = self.train_cfg.get("use_amp", False) and (self.device == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        print(f"[Trainer] Mixed Precision (AMP): {'ENABLED (FP16)' if self.use_amp else 'DISABLED (FP32)'}")

        # State tracking
        self.step = 0
        self.best_val_loss = float("inf")

    def evaluate(self, val_loader: DataLoader, eval_steps: int = 50) -> Dict[str, float]:
        """
        Compute validation loss and perplexity.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.
        eval_steps : int
            Number of batches to evaluate over.

        Returns
        -------
        dict with val_loss and perplexity.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= eval_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    _, loss = self.model(x, y)
                total_loss += loss.item()
                count += 1

        val_loss = total_loss / max(1, count)
        perplexity = math.exp(min(val_loss, 20.0))  # clamp to avoid overflow
        self.model.train()

        return {"val_loss": val_loss, "perplexity": perplexity}

    def train(self) -> None:
        """Execute main training loop."""
        print("\n" + "=" * 64)
        print("  Starting Tiny-LLM Training Run")
        print("=" * 64)

        data_dir = Path(self.paths_cfg["data_dir"])
        train_bin = data_dir / "train.bin"
        val_bin = data_dir / "val.bin"

        # Load Datasets
        train_ds = TokenDataset(str(train_bin), context_length=self.model_cfg["context_length"])
        val_ds = TokenDataset(str(val_bin), context_length=self.model_cfg["context_length"])

        batch_size = self.train_cfg["batch_size"]
        grad_accum_steps = self.train_cfg["gradient_accumulation_steps"]

        train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(self.device == "cuda"))
        val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(self.device == "cuda"))
        train_iter = infinite_iter(train_loader)

        max_steps = self.train_cfg["max_steps"]
        eval_interval = self.train_cfg["eval_interval"]
        save_interval = self.train_cfg["save_interval"]
        log_interval = self.train_cfg["log_interval"]
        grad_clip = self.train_cfg["grad_clip"]

        checkpoint_dir = self.paths_cfg["checkpoint_dir"]
        log_dir = Path(self.paths_cfg.get("log_dir", "experiments/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        self.model.train()
        t0 = time.time()
        tokens_processed = 0

        while self.step < max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # Gradient Accumulation Sub-steps
            for micro_step in range(grad_accum_steps):
                x, y = next(train_iter)
                x, y = x.to(self.device), y.to(self.device)
                tokens_processed += x.numel()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits, loss = self.model(x, y)
                    loss = loss / grad_accum_steps

                accum_loss += loss.item() * grad_accum_steps
                self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # Optimizer and Scaler step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # LR Scheduler update
            lr = self.scheduler.step(self.step)
            self.step += 1

            # Logging
            if self.step % log_interval == 0 or self.step == 1:
                t1 = time.time()
                elapsed = t1 - t0
                tok_per_sec = tokens_processed / max(1e-5, elapsed)
                print(f"Step {self.step:>6}/{max_steps} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | "
                      f"Throughput: {tok_per_sec:>7,.0f} tok/s")
                tokens_processed = 0
                t0 = time.time()

            # Periodic Validation
            if self.step % eval_interval == 0 or self.step == max_steps:
                metrics = self.evaluate(val_loader)
                val_loss = metrics["val_loss"]
                ppl = metrics["perplexity"]
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                print(f"--> [Validation Step {self.step}] Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f} "
                      f"{'(* Best *)' if is_best else ''}")

                # Save checkpoint on eval interval
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=self.step,
                    model=self.model,
                    optimizer=self.optimizer,
                    val_loss=val_loss,
                    best_val_loss=self.best_val_loss,
                    config=self.config,
                    is_best=is_best,
                )

            # Periodic Checkpointing
            elif self.step % save_interval == 0:
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=self.step,
                    model=self.model,
                    optimizer=self.optimizer,
                    val_loss=accum_loss,
                    best_val_loss=self.best_val_loss,
                    config=self.config,
                    is_best=False,
                )

        print("\n[Trainer] Pre-training completed successfully!")
