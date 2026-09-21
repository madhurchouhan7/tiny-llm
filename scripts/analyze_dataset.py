"""
scripts/analyze_dataset.py
──────────────────────────
Examine a sample of FineWeb-Edu and report statistics that help us
decide how many documents to preprocess to reach our token target.

─────────────────────────────────────────────────────────────────────────────
WHY WE ANALYZE FIRST
─────────────────────────────────────────────────────────────────────────────
FineWeb-Edu contains billions of tokens. We do NOT want to blindly load
a fixed number of rows and hope we get enough tokens. Instead, we:

  1. Stream a sample (e.g. 5000 documents)
  2. Measure actual token counts using our trained BPE tokenizer
  3. Compute average tokens/document
  4. Extrapolate: "how many documents do we need for 100M tokens?"

This prevents the common mistake of preprocessing "50k rows" and discovering
much later that those rows only contained 20M tokens (or 500M tokens).

─────────────────────────────────────────────────────────────────────────────
HOW TO RUN
─────────────────────────────────────────────────────────────────────────────
  python scripts/analyze_dataset.py

Optional arguments:
  --num-docs     INT    Number of docs to sample (default: 5000)
  --dataset      STR    FineWeb-Edu split (default: sample-10BT)
  --tokenizer    STR    Path to trained tokenizer (default: artifacts/tokenizer)

─────────────────────────────────────────────────────────────────────────────
OUTPUT
─────────────────────────────────────────────────────────────────────────────
Prints a table like:

  ── FineWeb-Edu Sample Statistics ──────────────────────────────
  Documents examined           :      5,000
  Avg characters / document    :      2,847
  Median characters / document :      1,923
  Avg BPE tokens / document    :        712
  Median BPE tokens / document :        481
  Total BPE tokens in sample   :  3,560,000
  ──────────────────────────────────────────────────────────────
  To collect 100M tokens → need ≈ 140,449 documents
  To collect 200M tokens → need ≈ 280,899 documents
  ──────────────────────────────────────────────────────────────
"""

import argparse
import sys
import time
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse FineWeb-Edu to estimate document/token counts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-docs", type=int, default=5_000,
                        help="Number of documents to sample and measure")
    parser.add_argument("--dataset", type=str, default="sample-10BT",
                        help="FineWeb-Edu HuggingFace split name")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer",
                        help="Path to the trained BPETokenizer directory")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Document filtering helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_document(text: str, min_chars: int = 100, max_chars: int = 100_000) -> bool:
    """
    Return True if the document should be included in the dataset.

    We filter out:
      - Documents that are too short (< 100 chars) — usually junk/metadata
      - Documents that are extremely long (> 100k chars) — may be legal text,
        data dumps, or other non-educational content
      - Documents with no alphabetic content (pure numbers, symbols, code)

    Parameters
    ----------
    text : str
        The raw document text.
    min_chars : int
        Minimum acceptable character count.
    max_chars : int
        Maximum acceptable character count.

    Returns
    -------
    bool
        True if the document should be kept.
    """
    if not text or not text.strip():
        return False
    char_count = len(text)
    if char_count < min_chars or char_count > max_chars:
        return False
    # Require that at least 40% of characters are letters/spaces
    # This filters out binary-looking data or heavily formatted tables
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text[:500]) / min(500, char_count)
    if alpha_ratio < 0.4:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse(args: argparse.Namespace) -> None:
    print("=" * 64)
    print("  Tiny-LLM: FineWeb-Edu Dataset Analysis")
    print("=" * 64)

    # ── Load tokenizer ─────────────────────────────────────────────────────
    tokenizer_path = Path(args.tokenizer)
    if not (tokenizer_path / "tokenizer_config.json").exists():
        print(f"[!] Tokenizer not found at: {tokenizer_path}")
        print(f"    Run: python scripts/train_tokenizer.py  first.")
        print(f"    Falling back to character-count estimate (1 token ≈ 4 chars).")
        tokenizer = None
    else:
        print(f"[1/2] Loading tokenizer from: {tokenizer_path}")
        tokenizer = BPETokenizer.load(str(tokenizer_path))

    # ── Stream FineWeb-Edu ─────────────────────────────────────────────────
    print(f"[2/2] Streaming {args.num_docs:,} documents from FineWeb-Edu ({args.dataset})...\n")
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.dataset,
        split="train",
        streaming=True,
    )

    # Accumulators for statistics
    char_counts = []         # list of character counts per document
    token_counts = []        # list of BPE token counts per document
    total_filtered = 0       # documents skipped by filter
    total_examined = 0       # raw documents seen (before filter)

    t0 = time.time()
    for example in dataset:
        total_examined += 1
        text = example.get("text", "")

        # Apply quality filter
        if not is_valid_document(text):
            total_filtered += 1
            continue

        # Measure character count
        n_chars = len(text)
        char_counts.append(n_chars)

        # Measure token count
        if tokenizer is not None:
            ids = tokenizer.encode(text)
            token_counts.append(len(ids))
        else:
            # Rough estimate: 1 token ≈ 4 characters for English BPE
            token_counts.append(n_chars // 4)

        # Progress update every 500 docs
        if len(char_counts) % 500 == 0:
            elapsed = time.time() - t0
            rate = len(char_counts) / elapsed
            print(f"  {len(char_counts):>5,} / {args.num_docs:,} docs  "
                  f"({rate:.0f} docs/sec)  "
                  f"avg tokens so far: {sum(token_counts)/len(token_counts):.0f}")

        if len(char_counts) >= args.num_docs:
            break

    # ── Compute statistics ─────────────────────────────────────────────────
    n = len(char_counts)
    if n == 0:
        print("ERROR: No valid documents found. Check dataset access.")
        return

    avg_chars   = statistics.mean(char_counts)
    med_chars   = statistics.median(char_counts)
    avg_tokens  = statistics.mean(token_counts)
    med_tokens  = statistics.median(token_counts)
    total_tokens_sample = sum(token_counts)
    elapsed = time.time() - t0

    filter_rate = total_filtered / max(total_examined, 1) * 100

    # ── Print report ───────────────────────────────────────────────────────
    print()
    print("─" * 64)
    print("  FineWeb-Edu Sample Statistics")
    print("─" * 64)
    print(f"  Documents examined (raw)      : {total_examined:>12,}")
    print(f"  Documents filtered out        : {total_filtered:>12,}  ({filter_rate:.1f}%)")
    print(f"  Documents analysed            : {n:>12,}")
    print(f"  Analysis time                 : {elapsed:>11.1f}s")
    print()
    print(f"  Avg characters  / document   : {avg_chars:>12,.0f}")
    print(f"  Median chars    / document   : {med_chars:>12,.0f}")
    print(f"  Avg BPE tokens  / document   : {avg_tokens:>12,.0f}")
    print(f"  Median BPE tokens / document : {med_tokens:>12,.0f}")
    print()
    print(f"  BPE tokens in sample         : {total_tokens_sample:>12,}")

    if tokenizer is not None:
        comp_ratio = sum(char_counts) / max(total_tokens_sample, 1)
        print(f"  Compression ratio            : {comp_ratio:>11.2f}  chars/token")

    print()
    print("─" * 64)
    print("  Token Target Extrapolations")
    print("─" * 64)
    for target_millions in [50, 100, 200, 300]:
        target_tokens = target_millions * 1_000_000
        docs_needed = int(target_tokens / avg_tokens)
        print(f"  For {target_millions:>3}M tokens  →  need ≈ {docs_needed:>10,} documents")
    print("─" * 64)
    print()
    print("TIP: Pass --num-docs to this script to examine more documents")
    print("     for a more accurate estimate.")


def main() -> None:
    args = parse_args()
    analyse(args)


if __name__ == "__main__":
    main()
