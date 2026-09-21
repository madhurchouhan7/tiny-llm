"""
scripts/generate.py
───────────────────
Interactive and CLI text generation from a trained Tiny-LLM checkpoint.

Usage:
  python scripts/generate.py --checkpoint checkpoints/tiny/best.pt --prompt "Once upon a time"
"""

import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gpt import GPT, GPTConfig
from src.tokenizer import BPETokenizer
from src.generation import generate_text
from src.training.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text using trained Tiny-LLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="artifacts/tokenizer",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence",
        help="Conditioning text prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K sampling cutoff",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-P (nucleus) sampling threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer)

    # Load Checkpoint to inspect config
    ckpt = torch.load(args.checkpoint, map_location=device)
    config_dict = ckpt["config"]["model"]
    gpt_config = GPTConfig.from_dict(config_dict)

    # Build Model and load weights
    model = GPT(gpt_config)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model.eval()

    print("\n" + "=" * 64)
    print(f"Prompt: {repr(args.prompt)}")
    print("=" * 64)

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print("\nGenerated Output:")
    print("-" * 64)
    print(generated)
    print("-" * 64)


if __name__ == "__main__":
    main()
