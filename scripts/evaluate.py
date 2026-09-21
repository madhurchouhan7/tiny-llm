"""
scripts/evaluate.py
───────────────────
Compute validation loss and perplexity on a dataset shard using a trained model checkpoint.

Usage:
  python scripts/evaluate.py --checkpoint checkpoints/tiny/best.pt --val-bin data/processed/tiny/val.bin
"""

import argparse
import sys
import math
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gpt import GPT, GPTConfig
from src.data.dataset import TokenDataset
from src.data.dataloader import make_dataloader
from src.training.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tiny-LLM on validation set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--val-bin",
        type=str,
        required=True,
        help="Path to validation .bin file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    config_dict = ckpt["config"]["model"]
    gpt_config = GPTConfig.from_dict(config_dict)

    model = GPT(gpt_config).to(device)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model.eval()

    val_ds = TokenDataset(args.val_bin, context_length=gpt_config.context_length)
    val_loader = make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\n[Evaluate] Running evaluation on {len(val_ds):,} examples...")
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    perplexity = math.exp(min(avg_loss, 20.0))

    print("─" * 60)
    print("  Evaluation Results")
    print("─" * 60)
    print(f"  Validation Loss : {avg_loss:.4f}")
    print(f"  Perplexity      : {perplexity:.2f}")
    print("─" * 60)


if __name__ == "__main__":
    main()
