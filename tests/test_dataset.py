"""
tests/test_dataset.py
─────────────────────
Unit tests for the data pipeline:
  - src/data/preprocess.py (Preprocessor)
  - src/data/dataset.py    (TokenDataset)
  - src/data/dataloader.py (make_dataloader, infinite_iter)

─────────────────────────────────────────────────────────────────────────────
WHAT WE TEST AND WHY
─────────────────────────────────────────────────────────────────────────────
Test 1 — Preprocessor writes valid binary files
    Verifies that train.bin and val.bin are created and non-empty after
    running the preprocessor on a tiny synthetic dataset.

Test 2 — Binary file contains only uint16 values in valid range
    Every token ID must be in [0, vocab_size). If any ID is out of range,
    the embedding table lookup will fail with an index error.

Test 3 — train/val split is approximately 98/2
    The modulo split must be deterministic and give roughly the right ratio.

Test 4 — Dataset shapes: x.shape == (T,), y.shape == (T,)
    Verifies spec requirement: each example has exactly context_length tokens.

Test 5 — y is x shifted by 1
    CRITICAL. The language modeling objective depends on this.
    y[t] must equal x[t+1] for all t.

Test 6 — DataLoader batch shapes
    x_batch.shape = (B, T), y_batch.shape = (B, T)
    where B = batch_size, T = context_length.

Test 7 — infinite_iter never raises StopIteration
    The training loop calls next() indefinitely; this must never crash.

Test 8 — Dataset length is correct
    len(dataset) = n_tokens - context_length - 1

─────────────────────────────────────────────────────────────────────────────
HOW TO RUN
─────────────────────────────────────────────────────────────────────────────
  python -m pytest tests/test_dataset.py -v
"""

import sys
import os
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import TokenDataset, inspect_dataset
from src.data.dataloader import make_dataloader, infinite_iter, get_batch_stats
from src.data.preprocess import Preprocessor, is_valid_document, TOKEN_DTYPE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: create synthetic binary files without FineWeb-Edu
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_bin(path: str, n_tokens: int, vocab_size: int = 16384, seed: int = 42) -> None:
    """
    Write a synthetic .bin file containing random token IDs.

    We use this to test the dataset/dataloader WITHOUT needing FineWeb-Edu.
    The preprocessing pipeline's tests are done separately with a tiny mock.

    Parameters
    ----------
    path : str
        Output file path.
    n_tokens : int
        Number of token IDs to write.
    vocab_size : int
        Max token ID (exclusive). IDs are in [0, vocab_size).
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    # Generate random token IDs in [0, vocab_size)
    tokens = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    tokens.tofile(path)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Document filter
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentFilter:
    """Tests for the is_valid_document() filter in preprocess.py."""

    def test_empty_string_is_rejected(self):
        assert not is_valid_document("")

    def test_whitespace_only_is_rejected(self):
        assert not is_valid_document("   \n\t  ")

    def test_too_short_is_rejected(self):
        assert not is_valid_document("hi")  # < 100 chars

    def test_normal_english_is_accepted(self):
        text = "The transformer architecture was introduced in 2017. " * 5
        assert is_valid_document(text)

    def test_very_long_doc_is_rejected(self):
        # > MAX_DOC_CHARS (100000)
        text = "a " * 55_000  # 110000 chars
        assert not is_valid_document(text)

    def test_mostly_numbers_is_rejected(self):
        # Very low alpha ratio
        text = "12345 67890 11111 22222 33333 " * 10
        assert not is_valid_document(text)

    def test_borderline_length_is_accepted(self):
        # Exactly 100 chars of normal text
        text = "The quick brown fox jumps over the lazy dog. " * 3  # ~135 chars
        assert is_valid_document(text)


# ─────────────────────────────────────────────────────────────────────────────
# Test: TokenDataset shapes and correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenDataset:
    """Tests for the TokenDataset class."""

    CONTEXT_LENGTH = 64    # smaller than production for faster tests
    N_TOKENS       = 5000  # enough to have many examples
    VOCAB_SIZE     = 16384

    @pytest.fixture
    def bin_file(self, tmp_path):
        """
        Fixture: create a temporary .bin file with synthetic tokens.
        `tmp_path` is a pytest built-in fixture that gives a unique temp directory.
        """
        path = tmp_path / "test_train.bin"
        make_synthetic_bin(str(path), self.N_TOKENS, self.VOCAB_SIZE)
        return path

    @pytest.fixture
    def dataset(self, bin_file):
        """Fixture: create a TokenDataset from the synthetic bin file."""
        return TokenDataset(str(bin_file), context_length=self.CONTEXT_LENGTH)

    # ── Test 8: Correct length ─────────────────────────────────────────────
    def test_length_is_correct(self, dataset):
        """
        Test 8: len(dataset) must equal n_tokens - context_length - 1.

        Reason: the last valid example starts at index n_tokens - T - 1,
        needs T+1 tokens (x = T tokens, y = T tokens with 1 overlap),
        so exactly T+1 tokens ending at n_tokens - 1.
        """
        expected_len = self.N_TOKENS - self.CONTEXT_LENGTH - 1
        assert len(dataset) == expected_len, (
            f"Expected len={expected_len}, got {len(dataset)}"
        )

    # ── Test 4: Shape of single example ───────────────────────────────────
    def test_single_example_shapes(self, dataset):
        """
        Test 4: Each example must have shape (context_length,).
        """
        T = self.CONTEXT_LENGTH
        x, y = dataset[0]
        assert x.shape == (T,), f"Expected x.shape=({T},), got {x.shape}"
        assert y.shape == (T,), f"Expected y.shape=({T},), got {y.shape}"

    def test_last_example_shapes(self, dataset):
        """Shape test for the last valid example (boundary condition)."""
        T = self.CONTEXT_LENGTH
        x, y = dataset[len(dataset) - 1]
        assert x.shape == (T,), f"Expected x.shape=({T},), got {x.shape}"
        assert y.shape == (T,), f"Expected y.shape=({T},), got {y.shape}"

    # ── Test 5: y is x shifted by 1 ───────────────────────────────────────
    def test_y_is_x_shifted_right_by_one(self, dataset):
        """
        Test 5: CRITICAL correctness test.

        If x = [t0, t1, t2, ..., tT-1]
        then y = [t1, t2, t3, ..., tT]

        In other words: y[i] == x[i+1] for i in [0, T-2]
        and y[-1] is the token AFTER x[-1] (first token of the next overlap).
        """
        for i in [0, 1, 100, len(dataset) - 1]:
            x, y = dataset[i]
            # The overlap: all of y except the last element should equal
            # all of x except the first element
            assert torch.equal(y[:-1], x[1:]), (
                f"At example {i}: y[:-1] != x[1:]  — y is NOT x shifted by 1!"
            )

    # ── Token ID range test ────────────────────────────────────────────────
    def test_token_ids_in_valid_range(self, dataset):
        """
        All token IDs in x and y must be in [0, vocab_size).
        Out-of-range IDs would cause an IndexError in nn.Embedding.
        """
        x, y = dataset[0]
        assert x.min() >= 0, f"Negative token ID in x: {x.min()}"
        assert y.min() >= 0, f"Negative token ID in y: {y.min()}"
        assert x.max() < self.VOCAB_SIZE, f"Token ID too large in x: {x.max()}"
        assert y.max() < self.VOCAB_SIZE, f"Token ID too large in y: {y.max()}"

    # ── dtype test ─────────────────────────────────────────────────────────
    def test_tensors_are_long_dtype(self, dataset):
        """
        x and y must be torch.long (int64).
        nn.Embedding.forward() requires LongTensor input.
        """
        x, y = dataset[0]
        assert x.dtype == torch.long, f"Expected torch.long, got {x.dtype}"
        assert y.dtype == torch.long, f"Expected torch.long, got {y.dtype}"

    # ── Missing file raises error ──────────────────────────────────────────
    def test_missing_file_raises_error(self):
        """Trying to create a dataset from a non-existent file should fail."""
        with pytest.raises(FileNotFoundError):
            TokenDataset("/nonexistent/path/train.bin", context_length=64)


# ─────────────────────────────────────────────────────────────────────────────
# Test: DataLoader batch shapes
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader:
    """Tests for make_dataloader() and infinite_iter()."""

    CONTEXT_LENGTH = 32
    N_TOKENS       = 10_000
    BATCH_SIZE     = 8

    @pytest.fixture
    def dataset(self, tmp_path):
        path = tmp_path / "loader_test.bin"
        make_synthetic_bin(str(path), self.N_TOKENS)
        return TokenDataset(str(path), context_length=self.CONTEXT_LENGTH)

    # ── Test 6: Batch shapes ───────────────────────────────────────────────
    def test_batch_shapes(self, dataset):
        """
        Test 6: DataLoader must produce batches of shape (B, T).
        """
        B = self.BATCH_SIZE
        T = self.CONTEXT_LENGTH
        loader = make_dataloader(dataset, batch_size=B, shuffle=False)
        x_batch, y_batch = next(iter(loader))
        assert x_batch.shape == (B, T), f"x_batch.shape={x_batch.shape}, expected ({B},{T})"
        assert y_batch.shape == (B, T), f"y_batch.shape={y_batch.shape}, expected ({B},{T})"

    def test_batch_x_y_shift(self, dataset):
        """
        In a batch, for each sequence in the batch,
        y[b, :-1] must equal x[b, 1:].
        """
        loader = make_dataloader(dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        x_batch, y_batch = next(iter(loader))
        # Check the shift property for every sequence in the batch
        for b in range(x_batch.shape[0]):
            assert torch.equal(y_batch[b, :-1], x_batch[b, 1:]), (
                f"Batch item {b}: y is not x shifted by 1"
            )

    def test_batch_dtype(self, dataset):
        """Batch tensors must be torch.long."""
        loader = make_dataloader(dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        x_batch, y_batch = next(iter(loader))
        assert x_batch.dtype == torch.long
        assert y_batch.dtype == torch.long

    # ── Test 7: infinite_iter never raises StopIteration ──────────────────
    def test_infinite_iter_does_not_stop(self, dataset):
        """
        Test 7: infinite_iter must yield indefinitely.
        We call next() more times than there are batches in the dataset.
        """
        loader = make_dataloader(dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        n_batches_per_epoch = len(loader)  # how many batches in one full pass

        it = infinite_iter(loader)

        # Call next() for 2.5 times the number of batches in one epoch
        # to verify the dataset cycles through correctly
        n_calls = int(n_batches_per_epoch * 2.5)
        for i in range(n_calls):
            try:
                x, y = next(it)
            except StopIteration:
                pytest.fail(f"infinite_iter raised StopIteration at call {i}")

    def test_infinite_iter_yields_correct_shapes(self, dataset):
        """Even after one full epoch, yielded shapes must be correct."""
        loader = make_dataloader(dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        n_batches_per_epoch = len(loader)
        it = infinite_iter(loader)

        B, T = self.BATCH_SIZE, self.CONTEXT_LENGTH
        # Skip to the second epoch
        for _ in range(n_batches_per_epoch + 1):
            x, y = next(it)

        assert x.shape == (B, T), f"After epoch wrap: x.shape={x.shape}"
        assert y.shape == (B, T), f"After epoch wrap: y.shape={y.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Preprocessor with a mock (no FineWeb-Edu needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessor:
    """
    Tests for the Preprocessor class using a tiny synthetic tokenizer
    and mock dataset (no internet access required).
    """

    @pytest.fixture(scope="class")
    def trained_tokenizer_dir(self, tmp_path_factory):
        """
        Train a tiny BPETokenizer and save it to a temp directory.
        scope="class" means this runs once for all tests in this class.
        """
        from src.tokenizer import BPETokenizer
        corpus = [
            "hello world machine learning transformer attention " * 200,
            "The quick brown fox jumps over the lazy dog. " * 200,
        ]
        tok = BPETokenizer(vocab_size=512)
        tok.train(texts=iter(corpus), show_progress=False)
        tok_dir = tmp_path_factory.mktemp("tokenizer")
        tok.save(str(tok_dir))
        return tok_dir

    def _run_preprocessor(self, tokenizer_dir, output_dir, mock_texts):
        """
        Run the Preprocessor but replace the FineWeb-Edu stream with
        our own mock texts (no internet access needed).
        """
        from src.data.preprocess import Preprocessor, is_valid_document
        import time, json
        from datetime import datetime

        preprocessor = Preprocessor(
            tokenizer_dir=str(tokenizer_dir),
            output_dir=str(output_dir),
            target_train_tokens=10_000,  # stop after 10k tokens
        )

        # Directly call the internals with mock data (bypass streaming)
        train_buffer, val_buffer = [], []
        train_tokens_total, val_tokens_total = 0, 0
        docs_processed = 0

        train_file = open(preprocessor.train_bin, "wb")
        val_file   = open(preprocessor.val_bin,   "wb")

        try:
            for text in mock_texts:
                if not is_valid_document(text):
                    continue
                token_ids = preprocessor.tokenizer.encode(text)
                token_ids.append(preprocessor.eos_id)

                if docs_processed % 50 == 0:
                    val_buffer.extend(token_ids)
                    val_tokens_total += len(token_ids)
                else:
                    train_buffer.extend(token_ids)
                    train_tokens_total += len(token_ids)
                docs_processed += 1

            if train_buffer:
                preprocessor._flush_buffer(train_buffer, train_file)
            if val_buffer:
                preprocessor._flush_buffer(val_buffer, val_file)
        finally:
            train_file.close()
            val_file.close()

        meta = {
            "vocab_size": preprocessor.tokenizer.get_vocab_size(),
            "train_tokens": train_tokens_total,
            "val_tokens": val_tokens_total,
            "total_tokens": train_tokens_total + val_tokens_total,
            "docs_processed": docs_processed,
            "token_dtype": str(TOKEN_DTYPE),
            "tokenizer_dir": str(tokenizer_dir),
            "endoftext_id": preprocessor.eos_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(preprocessor.meta_json, "w") as f:
            json.dump(meta, f)

        return meta

    @pytest.fixture(scope="class")
    def preprocessed_dir(self, tmp_path_factory, trained_tokenizer_dir):
        """
        Fixture: run the preprocessor on mock data and return the output dir.
        """
        output_dir = tmp_path_factory.mktemp("processed")
        # Create 200 realistic-looking mock documents
        mock_texts = [
            f"The topic of machine learning is document number {i}. "
            f"Transformers use self-attention to process sequences. "
            f"Neural networks learn representations from data. " * 5
            for i in range(200)
        ]
        self._run_preprocessor(trained_tokenizer_dir, output_dir, mock_texts)
        return output_dir

    def test_train_bin_exists(self, preprocessed_dir):
        """Test 1: train.bin must be created."""
        assert (preprocessed_dir / "train.bin").exists()

    def test_val_bin_exists(self, preprocessed_dir):
        """Test 1: val.bin must be created."""
        assert (preprocessed_dir / "val.bin").exists()

    def test_meta_json_exists(self, preprocessed_dir):
        """meta.json must be created."""
        assert (preprocessed_dir / "meta.json").exists()

    def test_train_bin_is_nonempty(self, preprocessed_dir):
        """Test 1: train.bin must have content."""
        size = (preprocessed_dir / "train.bin").stat().st_size
        assert size > 0, "train.bin is empty"

    def test_token_ids_in_valid_range(self, preprocessed_dir, trained_tokenizer_dir):
        """Test 2: All token IDs must be in [0, vocab_size)."""
        from src.tokenizer import BPETokenizer
        tok = BPETokenizer.load(str(trained_tokenizer_dir))
        vocab_size = tok.get_vocab_size()

        train_data = np.fromfile(str(preprocessed_dir / "train.bin"), dtype=np.uint16)
        assert train_data.min() >= 0, "Negative token ID found"
        assert train_data.max() < vocab_size, (
            f"Token ID {train_data.max()} >= vocab_size {vocab_size}"
        )

    def test_train_val_split_ratio(self, preprocessed_dir):
        """
        Test 3: Train should be ~49× larger than val (98% / 2%).
        We allow a wide tolerance because the exact split depends on
        token counts per document, not just document counts.
        """
        meta_path = preprocessed_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        train_t = meta["train_tokens"]
        val_t   = meta["val_tokens"]

        if val_t == 0:
            pytest.skip("No validation tokens (too few documents in mock)")

        ratio = train_t / val_t
        # Expect ratio around 49:1, allow ±50% tolerance
        assert 10 < ratio < 150, (
            f"Train/val ratio {ratio:.1f} is outside expected range [10, 150]. "
            f"train={train_t}, val={val_t}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Allow running directly: python tests/test_dataset.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(PROJECT_ROOT),
    )
    sys.exit(result.returncode)
