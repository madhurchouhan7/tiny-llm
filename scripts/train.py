"""
scripts/train.py
────────────────
Main training script for Tiny-LLM pre-training.

Usage:
  python scripts/train.py --config configs/tiny.yaml
  python scripts/train.py --config configs/50m.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny-LLM model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
