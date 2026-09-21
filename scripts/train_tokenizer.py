"""
scripts/train_tokenizer.py
──────────────────────────
Train a Byte-Pair Encoding (BPE) tokenizer on a sample of FineWeb-Edu text
and save it to artifacts/tokenizer/.

─────────────────────────────────────────────────────────────────────────────
WHY TRAIN A CUSTOM TOKENIZER?
─────────────────────────────────────────────────────────────────────────────
We want a tokenizer that is:
  1. Tailored to the vocabulary of our training data (FineWeb-Edu is
     educational web text — math, science, Wikipedia, textbooks).
  2. Exactly the vocab size we configured (16 384 tokens).
  3. Not borrowed from another model (GPT-2's tokenizer was trained on
     a different corpus with a different vocab size).

─────────────────────────────────────────────────────────────────────────────
HOW TO RUN
─────────────────────────────────────────────────────────────────────────────
From the project root directory:

    python scripts/train_tokenizer.py

Optional arguments:
    --vocab-size   INT    Target vocabulary size (default: 16384)
    --num-docs     INT    Number of FineWeb-Edu documents to train on
                          (default: 50000; more docs = better vocabulary)
    --output-dir   STR    Where to save the tokenizer (default: artifacts/tokenizer)
    --dataset      STR    HuggingFace dataset split to use (default: sample-10BT)

─────────────────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT DOES
─────────────────────────────────────────────────────────────────────────────
1. Streams documents from FineWeb-Edu (never loads full dataset into RAM)
2. Extracts the 'text' field from each document
3. Passes texts to BPETokenizer.train()
4. Prints vocabulary statistics
5. Runs a quick round-trip sanity check
6. Saves the tokenizer to disk
"""

import argparse
import sys
import time
from pathlib import Path

# ── Make sure 'src' is importable when running from the project root ───────────
# Python's import system looks in sys.path. When you run:
#   python scripts/train_tokenizer.py
# Python's cwd is the project root, but 'src' is not automatically on the path.
# We add the project root to sys.path here so that "from src.tokenizer ..." works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on FineWeb-Edu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16_384,
        help="Target BPE vocabulary size",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=50_000,
        help=(
            "Number of documents from FineWeb-Edu to use for BPE training. "
            "More documents → better vocabulary coverage, but slower training. "
            "50k docs is usually sufficient for good English coverage."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/tokenizer",
        help="Directory to save the trained tokenizer files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample-10BT",
        help=(
            "FineWeb-Edu dataset split to stream from. "
            "Options: sample-10BT, sample-100BT, default"
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Text streaming from FineWeb-Edu
# ─────────────────────────────────────────────────────────────────────────────

def stream_fineweb_edu_texts(dataset_name: str, num_docs: int):
    """
    Stream text documents from FineWeb-Edu without loading the full dataset
    into RAM.

    Parameters
    ----------
    dataset_name : str
        The FineWeb-Edu subset to use (e.g. "sample-10BT").
    num_docs : int
        Maximum number of documents to yield.

    Yields
    ------
    str
        One document text at a time.
    """
    # Import here to avoid making datasets a hard dependency for
    # users who only import src modules.
    from datasets import load_dataset

    print(f"[stream] Loading FineWeb-Edu ({dataset_name}) in streaming mode...")
    print(f"[stream] Will use {num_docs:,} documents for tokenizer training")

    # streaming=True means documents are fetched one-by-one from HuggingFace.
    # The full dataset (100B tokens) is never downloaded.
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=dataset_name,
        split="train",
        streaming=True,
    )

    # Iterate through documents and yield just the text field
    count = 0
    for example in dataset:
        text = example.get("text", "")
        if not text or not text.strip():
            continue  # Skip empty documents
        yield text
        count += 1
        if count >= num_docs:
            break

    print(f"[stream] Streamed {count:,} documents from FineWeb-Edu")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check: round-trip encoding test
# ─────────────────────────────────────────────────────────────────────────────

def run_roundtrip_test(tokenizer: BPETokenizer) -> None:
    """
    Verify that decode(encode(text)) == text for several examples.

    This is the most basic correctness test for any tokenizer:
    If you encode text to token IDs and then decode those IDs back to text,
    you should get the original text. If this fails, something is wrong with
    the tokenizer configuration.
    """
    test_cases = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Héllo, this has non-ASCII characters: café, naïve, résumé.",
        "Numbers: 3.14159, 42, 1e-5, 0xFF",
    ]

    print("\n── Round-trip tests ───────────────────────────────────")
    all_passed = True
    for text in test_cases:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        passed = (decoded == text)
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False
        print(f"  {status}  [{len(ids)} tokens]  {repr(text[:50])}")
        if not passed:
            print(f"         Expected: {repr(text)}")
            print(f"         Got:      {repr(decoded)}")

    if all_passed:
        print("  All round-trip tests passed!")
    else:
        print("  WARNING: Some round-trip tests failed — check tokenizer config.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Compression statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_compression_stats(tokenizer: BPETokenizer, sample_texts: list) -> None:
    """
    Print compression ratio and token statistics.

    A compression ratio of ~3–5 characters per token is typical for English
    text with a 16k-token vocabulary. This tells us:
      - The model sees roughly 1 token per 4 characters of text
      - At context_length=256, we cover ~1024 characters per training example
    """
    stats = tokenizer.get_compression_stats(sample_texts[:1000])  # use first 1000

    print("── Tokenizer statistics ───────────────────────────────")
    print(f"  Vocabulary size         : {tokenizer.get_vocab_size():>10,}")
    print(f"  <|endoftext|> ID        : {tokenizer.endoftext_id:>10}")
    print(f"  <|pad|> ID              : {tokenizer.pad_id:>10}")
    print(f"  Docs sampled            : {stats['num_documents']:>10,}")
    print(f"  Avg chars / document    : {stats['avg_chars_per_doc']:>10.1f}")
    print(f"  Avg tokens / document   : {stats['avg_tokens_per_doc']:>10.1f}")
    print(f"  Compression ratio       : {stats['compression_ratio']:>10.2f}  chars/token")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Tiny-LLM: Training BPE Tokenizer")
    print("=" * 60)
    print(f"  Target vocab size : {args.vocab_size:,}")
    print(f"  Documents to use  : {args.num_docs:,}")
    print(f"  Output directory  : {args.output_dir}")
    print(f"  Dataset split     : {args.dataset}")
    print()

    # ── Step 1: Stream FineWeb-Edu texts ──────────────────────────────────
    # We collect them into a list because the BPE trainer needs to make
    # multiple passes over the data. In production you'd use a file-based
    # approach, but for tokenizer training 50k docs fits in RAM (~200MB).
    print("[1/4] Fetching training documents...")
    t0 = time.time()
    texts = list(stream_fineweb_edu_texts(args.dataset, args.num_docs))
    t1 = time.time()
    print(f"      Collected {len(texts):,} documents in {t1-t0:.1f}s")

    # ── Step 2: Train the tokenizer ────────────────────────────────────────
    print(f"\n[2/4] Training BPE tokenizer (vocab_size={args.vocab_size:,})...")
    t0 = time.time()
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(texts=iter(texts), show_progress=True)
    t1 = time.time()
    print(f"      Training complete in {t1-t0:.1f}s")

    # ── Step 3: Print statistics ───────────────────────────────────────────
    print(f"\n[3/4] Computing statistics...")
    print_compression_stats(tokenizer, texts)

    # ── Step 4: Run round-trip test ────────────────────────────────────────
    run_roundtrip_test(tokenizer)

    # ── Step 5: Save to disk ───────────────────────────────────────────────
    print(f"[4/4] Saving tokenizer to: {args.output_dir}")
    tokenizer.save(args.output_dir)
    print(f"\nDone! Tokenizer is ready for use.")
    print(f"Load it with:  BPETokenizer.load('{args.output_dir}')")


if __name__ == "__main__":
    main()
