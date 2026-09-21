"""
scripts/prepare_data.py
───────────────────────
Entry-point script for the preprocessing pipeline.

Calls Preprocessor.run() with parameters from a config file.

─────────────────────────────────────────────────────────────────────────────
HOW TO RUN
─────────────────────────────────────────────────────────────────────────────
From the project root:

  # Create a tiny 5M-token dataset (for pipeline validation)
  python scripts/prepare_data.py --config configs/tiny.yaml

  # Create a 100M-token dataset (for the full training run)
  python scripts/prepare_data.py --config configs/50m.yaml

  # Override the token target directly
  python scripts/prepare_data.py --config configs/tiny.yaml --target-tokens 1000000

─────────────────────────────────────────────────────────────────────────────
PREREQUISITES
─────────────────────────────────────────────────────────────────────────────
The tokenizer must be trained first:
  python scripts/train_tokenizer.py

─────────────────────────────────────────────────────────────────────────────
OUTPUT
─────────────────────────────────────────────────────────────────────────────
Creates in <config.paths.data_dir>/:
  train.bin       flat uint16 array of training token IDs
  val.bin         flat uint16 array of validation token IDs
  meta.json       metadata about the preprocessing run
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.data.preprocess import Preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess FineWeb-Edu into binary token shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny.yaml",
        help="Path to YAML config file (debug.yaml / tiny.yaml / 50m.yaml)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Override the target token count from config (e.g. 1000000)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override the FineWeb-Edu split (e.g. sample-10BT)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load YAML config ────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract paths from config
    tokenizer_dir = cfg["paths"]["tokenizer_dir"]
    data_dir      = cfg["paths"]["data_dir"]

    # Token target: from config training section, or CLI override
    # We estimate tokens needed as: batch_size × max_steps × context_length × 2
    # But a simple heuristic is: for tiny = 5M, for 50m = 100M
    # We let the user set --target-tokens explicitly or derive from config.
    if args.target_tokens is not None:
        target_tokens = args.target_tokens
    else:
        # Sensible defaults based on config name
        context_length = cfg["model"]["context_length"]
        batch_size     = cfg["training"]["batch_size"]
        grad_accum     = cfg["training"]["gradient_accumulation_steps"]
        max_steps      = cfg["training"]["max_steps"]
        effective_batch = batch_size * grad_accum
        # Target roughly 10 "epochs" over the training steps
        target_tokens = effective_batch * context_length * max_steps
        # Round up to nearest million
        target_tokens = ((target_tokens // 1_000_000) + 1) * 1_000_000

    dataset_name = args.dataset or "sample-10BT"

    print(f"[prepare_data] Config   : {config_path}")
    print(f"[prepare_data] Data dir : {data_dir}")
    print(f"[prepare_data] Target   : {target_tokens:,} training tokens")
    print(f"[prepare_data] Dataset  : {dataset_name}")

    # ── Run preprocessor ────────────────────────────────────────────────────
    preprocessor = Preprocessor(
        tokenizer_dir=tokenizer_dir,
        output_dir=data_dir,
        target_train_tokens=target_tokens,
    )
    preprocessor.run(dataset_name=dataset_name)

    print(f"\nDone! Data ready in: {data_dir}")
    print(f"Next: python scripts/train.py --config {args.config}")


if __name__ == "__main__":
    main()
