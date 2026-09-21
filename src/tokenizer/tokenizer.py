"""
src/tokenizer/tokenizer.py
──────────────────────────
BPE (Byte-Pair Encoding) Tokenizer wrapper built on HuggingFace `tokenizers`.

─────────────────────────────────────────────────────────────────────────────
WHAT IS A TOKENIZER?
─────────────────────────────────────────────────────────────────────────────
A language model does not read raw text. It reads *integers* (token IDs).
The tokenizer is the bridge:

    Raw text  ──→  [token IDs]  ──→  model
    model     ──→  [token IDs]  ──→  Raw text

─────────────────────────────────────────────────────────────────────────────
WHAT IS BPE?
─────────────────────────────────────────────────────────────────────────────
Byte-Pair Encoding (BPE) is an algorithm that starts with a vocabulary of
individual bytes (256 entries) and then repeatedly *merges* the most
frequent pair of adjacent symbols until the target vocabulary size is
reached (in our case, 16 384 tokens).

Example evolution:
  Step 0: ['h', 'e', 'l', 'l', 'o']     (individual characters)
  ...BPE merges 'l' + 'l' → 'll' ...
  Step N: ['hello']                      (common word is one token)

This means common English words become a single token, while rare words
are split into subword pieces. It achieves a good balance between:
  - Vocabulary size (not too large, not too small)
  - Coverage (can represent ANY text, even unseen words)
  - Compression ratio (fewer tokens per character than character-level)

─────────────────────────────────────────────────────────────────────────────
WHY WE USE THE `tokenizers` LIBRARY INSTEAD OF IMPLEMENTING BPE OURSELVES
─────────────────────────────────────────────────────────────────────────────
Implementing BPE from scratch is a worthwhile exercise, but it would require
O(N²) Python loops over millions of documents — slow and error-prone.
The `tokenizers` library (written in Rust) does this efficiently and correctly.

Our goal is to understand the LANGUAGE MODEL, not recreate the tokenizer
library. So we use `tokenizers` but document every design choice below.

─────────────────────────────────────────────────────────────────────────────
DESIGN CHOICES (documented as required by the spec)
─────────────────────────────────────────────────────────────────────────────
1. Vocabulary size: 16 384 tokens
   - Powers of 2 are numerically clean and GPU-friendly
   - Large enough for English text (few unknown splits)
   - Small enough to keep the embedding table manageable (~8M params at dim=512)

2. Special tokens:
   - <|endoftext|>  (id=0): Marks document boundaries. Inserted between
                             documents during preprocessing so the model
                             learns not to bleed context across documents.
   - <|pad|>        (id=1): Padding token. Used to fill short sequences
                             to a fixed length when batching.

3. Pre-tokenization: ByteLevel
   - Text is first split by whitespace-like boundaries, then each piece
     is converted to its UTF-8 bytes represented as printable Unicode chars.
   - This means the tokenizer is *byte-level*: it can handle ANY text
     (even emojis, code, non-ASCII) without ever producing "unknown" tokens.
   - GPT-2 uses the same approach.

4. Normalization: None
   - We do not lowercase, strip accents, or apply Unicode NFKC.
   - Reason: We want the model to see mixed-case text as it appears in the
     training data. Normalization would destroy information (e.g. "CNN" ≠ "cnn").

5. Decoder: ByteLevel
   - Inverts the byte-level encoding so that decoded text looks like normal
     readable text (not raw byte representations).

─────────────────────────────────────────────────────────────────────────────
USAGE EXAMPLE
─────────────────────────────────────────────────────────────────────────────
    # Train a new tokenizer on a list of text strings
    tok = BPETokenizer(vocab_size=16384)
    tok.train(texts=["hello world", "this is a sentence", ...])

    # Encode text to token IDs
    ids = tok.encode("Hello, world!")     # e.g. [8234, 11, 995, 0]

    # Decode token IDs back to text
    text = tok.decode([8234, 11, 995, 0]) # "Hello, world!"

    # Save to disk (two files: vocab.json + merges.txt)
    tok.save("artifacts/tokenizer")

    # Load from disk later
    tok2 = BPETokenizer.load("artifacts/tokenizer")
"""

import os
import json
from pathlib import Path
from typing import List, Iterator, Optional

# ── HuggingFace tokenizers components ─────────────────────────────────────────
# We import individual building blocks and assemble them ourselves so it's
# clear what each piece does.

from tokenizers import Tokenizer               # The main Tokenizer container
from tokenizers.models import BPE              # The BPE merge algorithm
from tokenizers.trainers import BpeTrainer     # Drives the BPE merge iterations
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
# ByteLevelPreTokenizer splits text into "byte-level" tokens before BPE runs.
# Think of it as: text → raw bytes → printable Unicode representations

from tokenizers.decoders import ByteLevel as ByteLevelDecoder
# ByteLevelDecoder inverts the above transformation so decoded text
# looks like normal readable text.

from tokenizers.processors import TemplateProcessing
# TemplateProcessing lets us insert special tokens automatically.
# (We use this only optionally — document boundaries are usually inserted
#  manually in the preprocessing script.)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# File names used when saving/loading the tokenizer to/from disk
VOCAB_FILE = "vocab.json"    # Maps each token string → integer ID
MERGES_FILE = "merges.txt"   # Ordered list of BPE merge rules
CONFIG_FILE = "tokenizer_config.json"  # Our extra metadata (vocab_size, etc.)

# Special token strings
ENDOFTEXT_TOKEN = "<|endoftext|>"  # Document separator
PAD_TOKEN = "<|pad|>"              # Padding (for batching)


# ─────────────────────────────────────────────────────────────────────────────
# BPETokenizer class
# ─────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    A Byte-Pair Encoding tokenizer trained from scratch using HuggingFace
    `tokenizers`.

    This class wraps the low-level `tokenizers.Tokenizer` object and exposes
    a simple, educational API:

        encode(text)  → list of integer token IDs
        decode(ids)   → text string
        train(texts)  → train a new vocabulary from scratch
        save(dir)     → persist to disk
        load(dir)     → restore from disk (class method)

    After training, the tokenizer holds:
        - vocab_size: number of unique tokens (e.g. 16 384)
        - endoftext_id: integer ID of the <|endoftext|> token
        - pad_id: integer ID of the <|pad|> token
    """

    def __init__(self, vocab_size: int = 16_384):
        """
        Create a new (untrained) BPETokenizer.

        Parameters
        ----------
        vocab_size : int
            Target vocabulary size. The BPE trainer will keep merging pairs
            until this many unique tokens exist.
            Default: 16 384 (= 2^14, a power-of-two vocab).
        """
        self.vocab_size = vocab_size

        # The underlying HuggingFace Tokenizer object.
        # We build it fresh here; it gets populated during .train() or .load().
        # BPE() with no arguments means "empty model — no merges yet".
        self._tokenizer: Optional[Tokenizer] = None

        # Convenient shortcuts to special token IDs (filled after training)
        self.endoftext_id: Optional[int] = None
        self.pad_id: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        texts: Iterator[str],
        show_progress: bool = True,
    ) -> None:
        """
        Train a new BPE vocabulary from scratch on the provided text stream.

        How this works step by step:
          1. Start with a vocabulary of 256 "byte tokens" (one per possible
             byte value, represented as printable Unicode characters).
          2. Add the 2 special tokens (<|endoftext|> and <|pad|>).
          3. Repeatedly find the most frequent ADJACENT pair of tokens in
             the corpus and merge them into a new single token.
          4. Repeat step 3 until vocab_size is reached.

        Parameters
        ----------
        texts : Iterator[str]
            An iterable of raw text strings. This can be a list in memory
            or a lazy generator (important for large corpora — we never need
            to load everything at once).
        show_progress : bool
            Whether to show a progress bar during training.
        """
        # ── Step 1: Create a fresh BPE model ──────────────────────────────
        # We do NOT set unk_token here. Here's why:
        #
        # A ByteLevel pre-tokenizer converts every character to its byte
        # representation. Since there are only 256 possible byte values and
        # BPE starts with ALL 256 bytes already in the vocabulary, there is
        # NO possible input character that cannot be encoded. Therefore, the
        # concept of an "unknown token" is unnecessary and the tokenizers
        # library would error if we specified one but it ended up unused.
        tokenizer = Tokenizer(BPE())

        # ── Step 2: ByteLevel pre-tokenizer ───────────────────────────────
        # Before the BPE algorithm runs, text is split by this pre-tokenizer.
        # ByteLevel:
        #   - Converts each character to its UTF-8 byte representation
        #   - Represents bytes as printable Unicode characters (Ġ, Ā, ĉ, etc.)
        #   - add_prefix_space=False: the first word does NOT get an extra
        #     leading space prefix (simpler round-trips)
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)

        # ── Step 3: ByteLevel decoder ─────────────────────────────────────
        # When we call decode(), this undoes the byte→Unicode mapping so
        # the output looks like normal text.
        # add_prefix_space=False must match the pre_tokenizer setting.
        tokenizer.decoder = ByteLevelDecoder(add_prefix_space=False)

        # ── Step 4: Configure the BPE trainer ─────────────────────────────
        # BpeTrainer controls how the training loop behaves.
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            # Special tokens are inserted into the vocabulary FIRST
            # (before any BPE merges) so they get low IDs (0, 1, ...).
            special_tokens=[ENDOFTEXT_TOKEN, PAD_TOKEN],
            # initial_alphabet: CRITICAL — this tells the trainer to include
            # ALL 256 possible byte representations in the base vocabulary,
            # even if some bytes never appear in the training corpus.
            #
            # Without this, a byte like 0xFF that never appears in the corpus
            # would be absent from the vocabulary, making it impossible to
            # encode text containing that byte later. This would break the
            # "encode any text" guarantee of byte-level BPE.
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
            # Show training progress if requested
            show_progress=show_progress,
            # minimum_frequency: a pair must appear at least this often
            # to be considered for merging. Keeps vocabulary clean.
            min_frequency=2,
        )

        # ── Step 5: Run BPE training ───────────────────────────────────────
        # `train_from_iterator` feeds texts one by one to the trainer.
        # This is memory-efficient: we never load the whole corpus at once.
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # ── Step 6: Store results ─────────────────────────────────────────
        self._tokenizer = tokenizer
        self._cache_special_token_ids()

        # Print a brief summary so the user knows what was trained
        actual_vocab_size = self._tokenizer.get_vocab_size()
        print(f"[BPETokenizer] Training complete.")
        print(f"  Vocabulary size : {actual_vocab_size:,}")
        print(f"  <|endoftext|> id: {self.endoftext_id}")
        print(f"  <|pad|> id      : {self.pad_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Encoding and Decoding
    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """
        Convert a string of text into a list of integer token IDs.

        Example:
            encode("Hello world")  →  [8234, 995]
                                       ↑      ↑
                                       "Hello" "world"  (approximate)

        Parameters
        ----------
        text : str
            Any UTF-8 text string.

        Returns
        -------
        List[int]
            Token IDs that the model will use as input.
        """
        self._require_trained()
        # `encode()` returns a HuggingFace Encoding object; `.ids` gives the list
        return self._tokenizer.encode(text).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a list of strings in one call (faster than calling encode()
        in a Python loop because the underlying Rust code parallelises it).

        Parameters
        ----------
        texts : List[str]
            A list of text strings.

        Returns
        -------
        List[List[int]]
            One list of token IDs per input string.
        """
        self._require_trained()
        # encode_batch returns a list of Encoding objects; we extract .ids
        encodings = self._tokenizer.encode_batch(texts)
        return [e.ids for e in encodings]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a list of token IDs back to a human-readable string.

        Example:
            decode([8234, 995])  →  "Hello world"

        Parameters
        ----------
        ids : List[int]
            Token IDs produced by encode() or by the model during generation.
        skip_special_tokens : bool
            If True (default), the <|endoftext|> and <|pad|> tokens are
            removed from the decoded string. Set to False if you want to see
            where document boundaries are.

        Returns
        -------
        str
            Decoded human-readable text.
        """
        self._require_trained()
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # ─────────────────────────────────────────────────────────────────────────
    # Save and Load
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """
        Save the trained tokenizer to disk.

        Three files are written to `directory`:
          vocab.json   — maps each token string to its integer ID
          merges.txt   — the ordered list of BPE merge rules
                         (these two are the standard GPT-2 tokenizer format)
          tokenizer_config.json — our extra metadata (vocab_size, etc.)

        Parameters
        ----------
        directory : str
            Path to the directory where files will be saved.
            The directory is created if it does not exist.
        """
        self._require_trained()
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the tokenizer in HuggingFace's standard format (vocab + merges)
        # This format is widely compatible with other tools.
        self._tokenizer.model.save(str(save_dir))

        # The model.save() above writes vocab.json + merges.txt.
        # We also save the full tokenizer config (including pre-tokenizer,
        # decoder, and special tokens) as a single JSON for easy loading.
        full_tokenizer_path = save_dir / "full_tokenizer.json"
        self._tokenizer.save(str(full_tokenizer_path))

        # Save our own metadata so we can reconstruct the Python object
        config = {
            "vocab_size": self.vocab_size,
            "endoftext_id": self.endoftext_id,
            "pad_id": self.pad_id,
            "endoftext_token": ENDOFTEXT_TOKEN,
            "pad_token": PAD_TOKEN,
        }
        config_path = save_dir / CONFIG_FILE
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"[BPETokenizer] Saved to: {save_dir}")
        print(f"  {VOCAB_FILE}")
        print(f"  {MERGES_FILE}")
        print(f"  full_tokenizer.json")
        print(f"  {CONFIG_FILE}")

    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        """
        Load a previously trained tokenizer from disk.

        This is a *class method*, meaning you call it on the class, not on
        an instance:

            tok = BPETokenizer.load("artifacts/tokenizer")

        Parameters
        ----------
        directory : str
            Path to the directory that was created by .save().

        Returns
        -------
        BPETokenizer
            A fully initialised tokenizer, ready to encode/decode.

        Raises
        ------
        FileNotFoundError
            If the directory or required files don't exist.
        """
        load_dir = Path(directory)

        # Check that the directory and config file exist
        config_path = load_dir / CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(
                f"Tokenizer config not found at {config_path}. "
                f"Run scripts/train_tokenizer.py first."
            )

        # Load our metadata
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Reconstruct the BPETokenizer object
        instance = cls(vocab_size=config["vocab_size"])

        # Load the full tokenizer (with pre-tokenizer, decoder, etc.)
        full_tokenizer_path = load_dir / "full_tokenizer.json"
        instance._tokenizer = Tokenizer.from_file(str(full_tokenizer_path))

        # Restore cached special token IDs
        instance.endoftext_id = config["endoftext_id"]
        instance.pad_id = config["pad_id"]

        print(f"[BPETokenizer] Loaded from: {load_dir}")
        print(f"  Vocabulary size : {instance.get_vocab_size():,}")
        return instance

    # ─────────────────────────────────────────────────────────────────────────
    # Utility methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_vocab_size(self) -> int:
        """Return the actual number of tokens in the trained vocabulary."""
        self._require_trained()
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        """
        Look up the integer ID for a token string.
        Returns None if the token is not in the vocabulary.
        """
        self._require_trained()
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """
        Look up the token string for an integer ID.
        Returns None if the ID is out of range.
        """
        self._require_trained()
        return self._tokenizer.id_to_token(token_id)

    def get_compression_stats(self, sample_texts: List[str]) -> dict:
        """
        Compute tokenizer statistics on a sample of texts.

        Returns a dictionary with:
          avg_chars_per_token : average number of characters per token
                                (higher = better compression)
          avg_tokens_per_doc  : average token count per document
          avg_chars_per_doc   : average character count per document
          compression_ratio   : characters / tokens (same as avg_chars_per_token)

        A compression ratio around 3–5 chars/token is typical for English
        BPE tokenizers with vocab size ~16k.

        Parameters
        ----------
        sample_texts : List[str]
            A representative sample of documents to measure on.
        """
        self._require_trained()
        total_chars = 0
        total_tokens = 0
        for text in sample_texts:
            total_chars += len(text)
            total_tokens += len(self.encode(text))

        n = max(len(sample_texts), 1)  # avoid division by zero
        return {
            "num_documents": n,
            "avg_chars_per_doc": total_chars / n,
            "avg_tokens_per_doc": total_tokens / n,
            "compression_ratio": total_chars / max(total_tokens, 1),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _require_trained(self) -> None:
        """Raise an error if the tokenizer hasn't been trained or loaded yet."""
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer has not been trained or loaded yet. "
                "Call .train() or BPETokenizer.load() first."
            )

    def _cache_special_token_ids(self) -> None:
        """
        Cache the integer IDs of special tokens after training or loading.
        This avoids repeated dictionary lookups during preprocessing.
        """
        self.endoftext_id = self._tokenizer.token_to_id(ENDOFTEXT_TOKEN)
        self.pad_id = self._tokenizer.token_to_id(PAD_TOKEN)

        if self.endoftext_id is None:
            raise RuntimeError(
                f"Special token '{ENDOFTEXT_TOKEN}' not found in vocabulary. "
                "This should not happen — check the BpeTrainer special_tokens list."
            )

    def __repr__(self) -> str:
        trained = self._tokenizer is not None
        vocab = self.get_vocab_size() if trained else "untrained"
        return f"BPETokenizer(vocab_size={vocab})"
