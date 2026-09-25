import argparse
import sys
from pathlib import Path
import json
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gpt import GPT, GPTConfig
from src.tokenizer import BPETokenizer
from src.generation import generate_stream

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )

    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("================================================")
    print("Tiny-LLM — Local Inference")
    print("================================================")
    print(f"Device: {device}")

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    config_path = PROJECT_ROOT / "output" / "config.json"
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = GPTConfig.from_dict(config_dict)
    except Exception as e:
        print(f"Error: Failed to load configuration: {e}")
        sys.exit(1)

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    try:
        model = GPT(config)
    except Exception as e:
        print(f"Error: Failed to initialize model: {e}")
        sys.exit(1)

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # --------------------------------------------------
    # Load trained weights
    # --------------------------------------------------

    weights_path = PROJECT_ROOT / "output" / "pytorch_model.bin"
    if not weights_path.exists():
        print(f"Error: Model weights not found at {weights_path}")
        sys.exit(1)

    try:
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error: Failed to load model weights: {e}")
        sys.exit(1)

    print("Model loaded successfully.")

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    tokenizer_path = PROJECT_ROOT / "output"
    try:
        tokenizer = BPETokenizer.load(str(tokenizer_path))
    except Exception as e:
        print(f"Error: Failed to load tokenizer: {e}")
        sys.exit(1)

    # --------------------------------------------------
    # Generate — streaming
    # --------------------------------------------------

    print()
    print("-" * 48)
    print(f"Prompt: {args.prompt}")
    print("-" * 48)

    prompt_ids = tokenizer.encode(args.prompt)

    if len(prompt_ids) == 0:
        prompt_ids = [tokenizer.endoftext_id]

    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    print("Generated:")
    print("-" * 48)

    # Print the prompt first
    print(args.prompt, end="", flush=True)

    # Generate token-by-token
    start_time = time.time()
    num_generated = 0
    
    try:
        for token_text in generate_stream(
            model=model,
            idx=idx,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=tokenizer.endoftext_id,
        ):
            print(token_text, end="", flush=True)
            num_generated += 1
    except Exception as e:
        print(f"\nError: Generation failed: {e}")
        sys.exit(1)

    print()
    print("-" * 48)
    
    end_time = time.time()
    gen_time = end_time - start_time
    speed = num_generated / gen_time if gen_time > 0 else 0
    
    print(f"Generated {num_generated} tokens")
    print(f"Generation time: {gen_time:.2f} s")
    print(f"Speed: {speed:.2f} tokens/sec")
    print("-" * 48)

if __name__ == "__main__":
    main()