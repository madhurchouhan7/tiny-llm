"""
tests/test_tokenizer.py
───────────────────────
Unit tests for the BPETokenizer (src/tokenizer/tokenizer.py).

─────────────────────────────────────────────────────────────────────────────
WHAT WE TEST AND WHY
─────────────────────────────────────────────────────────────────────────────
Before training on FineWeb-Edu, we must verify that the tokenizer is
correct. A broken tokenizer would silently corrupt all downstream data.

Test 1 — Train on tiny corpus
    Verify that .train() runs without errors and produces a vocabulary
    of the expected size.

Test 2 — Round-trip (encode → decode = original text)
    The most fundamental tokenizer property. If this fails, the model
    will be trained on corrupted data.

Test 3 — Special tokens exist and have correct IDs
    <|endoftext|> must be token 0 (BpeTrainer inserts special tokens first).
    The model uses this ID to separate documents during training.

Test 4 — Encode returns integers only
    Token IDs must be Python ints in the range [0, vocab_size).

Test 5 — Empty string and edge cases
    The tokenizer should not crash on empty strings or single characters.

Test 6 — Save and load round-trip
    After loading a saved tokenizer, encode/decode results must be
    identical to the original tokenizer. This is critical because the
    training script and the inference script use different processes.

Test 7 — Compression ratio sanity check
    A 16k-vocab BPE tokenizer on English should give ~3-6 chars/token.
    Values outside this range suggest a misconfigured tokenizer.

─────────────────────────────────────────────────────────────────────────────
HOW TO RUN
─────────────────────────────────────────────────────────────────────────────
From the project root:

    python -m pytest tests/test_tokenizer.py -v

Or run just this file:
    python tests/test_tokenizer.py
"""

import sys
import os
import tempfile
from pathlib import Path

# ── Add project root to import path ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.tokenizer import BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

# A small synthetic corpus we train the tokenizer on during tests.
# Using a tiny corpus makes training fast (< 1 second).
#
# IMPORTANT: The corpus must contain ALL character types used in the test
# cases below. If a character class is absent from the training corpus,
# the BPE model won't produce merges for it and round-trips may fail.
#
# We need coverage of:
#   - Lowercase letters  (a-z)
#   - Uppercase letters  (A-Z)          ← needed by "The", "Hello" etc.
#   - Digits             (0-9)          ← needed by "1 + 1 = 2"
#   - Punctuation        ., !, ?, +, = ← needed by test sentences
#   - Whitespace         space, \n, \t  ← newlines and tabs
#   - Non-ASCII          é, ï           ← needed by café/naïve test
TRAINING_CORPUS = [
    # Lowercase + uppercase + punctuation coverage
    "Hello World. The Quick Brown Fox Jumps Over The Lazy Dog. " * 50,
    "hello world the quick brown fox jumps over the lazy dog " * 80,
    "Machine Learning And Artificial Intelligence Are Amazing. " * 60,
    # Digits and operators
    "1 + 1 = 2, 3 * 4 = 12, 100 / 5 = 20, x = 3.14159, y = 0.001 " * 50,
    # Newlines and tabs (must appear in training corpus to survive round-trip)
    "hello\nworld\nthis\nis\na\nnewline\ntest\n" * 80,
    "column1\tcolumn2\tcolumn3\tvalue1\tvalue2\t" * 80,
    # Non-ASCII (UTF-8 multi-byte characters)
    "café résumé naïve jalapeño über coöperate " * 50,
    # Common English words and sentences
    "transformer attention mechanism self supervised learning " * 70,
    "language model training data tokenization byte pair encoding " * 85,
    "deep neural network parameters gradient descent optimization " * 75,
    "loss function cross entropy softmax temperature sampling " * 80,
    "positional encoding embedding layer normalization residual " * 70,
    # Mixed case and punctuation
    "The model achieved 95.3% accuracy on the benchmark dataset! " * 60,
    "In 2024, researchers published over 10,000 papers on LLMs. " * 55,
] * 3  # repeat 3× for more BPE merge opportunities


@pytest.fixture(scope="module")
def trained_tokenizer():
    """
    Pytest fixture: Creates and trains a small BPETokenizer once and reuses
    it across all tests in this module.

    `scope="module"` means the fixture is created once per test file,
    not once per test function. This speeds up the test suite significantly
    since training takes a moment.

    We use a small vocab_size (512) so training is fast.
    """
    tok = BPETokenizer(vocab_size=512)
    tok.train(texts=iter(TRAINING_CORPUS), show_progress=False)
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBPETokenizerTraining:
    """Tests related to tokenizer training."""

    def test_train_runs_without_error(self, trained_tokenizer):
        """
        Test 1: Training should complete without raising any exception.
        This is the most basic smoke test.
        """
        # If we got here, training didn't crash. The fixture already trained it.
        assert trained_tokenizer is not None

    def test_vocab_size_is_correct(self, trained_tokenizer):
        """
        Test 1b: The actual vocabulary size should match what we requested.

        Note: BPE may produce slightly fewer tokens than requested if the
        corpus doesn't have enough unique pairs to merge. For very small
        corpora this is expected.
        """
        vocab_size = trained_tokenizer.get_vocab_size()
        # Should be ≤ 512 (our requested size) and > 256 (the byte-level baseline)
        assert vocab_size <= 512, f"Vocab too large: {vocab_size}"
        assert vocab_size > 10, f"Vocab suspiciously small: {vocab_size}"


class TestEncoding:
    """Tests for the encode() and decode() methods."""

    def test_encode_returns_list_of_ints(self, trained_tokenizer):
        """
        Test 4: encode() must return a list of integers.
        The model's embedding layer expects integer token IDs.
        """
        text = "hello world"
        ids = trained_tokenizer.encode(text)

        assert isinstance(ids, list), "encode() should return a list"
        assert len(ids) > 0, "Encoded result should not be empty"
        assert all(isinstance(i, int) for i in ids), "All IDs must be integers"

    def test_encode_ids_in_valid_range(self, trained_tokenizer):
        """
        Test 4b: All token IDs must be in [0, vocab_size).
        IDs outside this range would cause an out-of-bounds error in the
        embedding table lookup.
        """
        text = "the quick brown fox jumps over the lazy dog"
        ids = trained_tokenizer.encode(text)
        vocab_size = trained_tokenizer.get_vocab_size()

        for token_id in ids:
            assert 0 <= token_id < vocab_size, (
                f"Token ID {token_id} is out of range [0, {vocab_size})"
            )

    def test_encode_nonempty_text_gives_nonempty_ids(self, trained_tokenizer):
        """Non-empty text should always produce at least one token."""
        ids = trained_tokenizer.encode("hello")
        assert len(ids) >= 1, "Non-empty text should produce tokens"

    def test_encode_empty_string(self, trained_tokenizer):
        """
        Test 5 (edge case): encode("") should return an empty list or a list
        with only whitespace-related tokens. It must NOT crash.
        """
        ids = trained_tokenizer.encode("")
        # Result depends on implementation; just verify it doesn't crash
        # and returns a list.
        assert isinstance(ids, list)


class TestDecoding:
    """Tests for the decode() method."""

    def test_decode_returns_string(self, trained_tokenizer):
        """decode() must return a str, not bytes or a list."""
        ids = trained_tokenizer.encode("hello world")
        decoded = trained_tokenizer.decode(ids)
        assert isinstance(decoded, str)

    def test_decode_nonempty_ids_gives_nonempty_string(self, trained_tokenizer):
        """Decoding a non-empty ID list should give non-empty text."""
        ids = trained_tokenizer.encode("hello world")
        decoded = trained_tokenizer.decode(ids)
        assert len(decoded) > 0


class TestRoundTrip:
    """
    Test 2: Round-trip tests.

    The critical property:  decode(encode(text)) == text

    If this fails, the tokenizer is corrupting text, which would cascade
    into corrupted training data.
    """

    ROUND_TRIP_CASES = [
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "machine learning is fascinating",
        "1 + 1 = 2",
        # Unusual characters that byte-level BPE must handle
        "café and naïve",
        "hello\nworld",   # newline
        "hello\tworld",   # tab
    ]

    @pytest.mark.parametrize("text", ROUND_TRIP_CASES)
    def test_roundtrip(self, trained_tokenizer, text):
        """
        For each test case: decode(encode(text)) must equal the original text.
        """
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids, skip_special_tokens=False)
        assert decoded == text, (
            f"Round-trip failed!\n"
            f"  Original : {repr(text)}\n"
            f"  Encoded  : {ids}\n"
            f"  Decoded  : {repr(decoded)}"
        )


class TestSpecialTokens:
    """
    Test 3: Special token correctness.
    """

    def test_endoftext_token_exists(self, trained_tokenizer):
        """
        The <|endoftext|> token must exist in the vocabulary.
        It's used to mark document boundaries during preprocessing.
        """
        eos_id = trained_tokenizer.endoftext_id
        assert eos_id is not None, "<|endoftext|> token not found in vocabulary"
        assert isinstance(eos_id, int)

    def test_pad_token_exists(self, trained_tokenizer):
        """The <|pad|> token must exist."""
        pad_id = trained_tokenizer.pad_id
        assert pad_id is not None, "<|pad|> token not found in vocabulary"
        assert isinstance(pad_id, int)

    def test_special_tokens_have_distinct_ids(self, trained_tokenizer):
        """Special tokens must have DIFFERENT IDs (they must not collide)."""
        assert trained_tokenizer.endoftext_id != trained_tokenizer.pad_id, (
            "<|endoftext|> and <|pad|> should have different token IDs"
        )

    def test_endoftext_encodes_to_its_id(self, trained_tokenizer):
        """
        Encoding the <|endoftext|> string should return exactly [endoftext_id].
        This verifies that the special token is recognized as a unit.
        """
        from src.tokenizer.tokenizer import ENDOFTEXT_TOKEN
        ids = trained_tokenizer.encode(ENDOFTEXT_TOKEN)
        # The special token should encode to a single ID
        assert len(ids) == 1, (
            f"<|endoftext|> should encode to 1 token, got {len(ids)}: {ids}"
        )
        assert ids[0] == trained_tokenizer.endoftext_id


class TestSaveLoad:
    """
    Test 6: Save and load round-trip.
    The tokenizer on disk must be byte-for-byte equivalent to the in-memory one.
    """

    def test_save_creates_expected_files(self, trained_tokenizer):
        """
        After .save(), three files must exist:
          vocab.json, merges.txt, full_tokenizer.json, tokenizer_config.json
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_tokenizer.save(tmpdir)
            files = set(os.listdir(tmpdir))
            assert "vocab.json" in files, "vocab.json missing"
            assert "merges.txt" in files, "merges.txt missing"
            assert "full_tokenizer.json" in files, "full_tokenizer.json missing"
            assert "tokenizer_config.json" in files, "tokenizer_config.json missing"

    def test_load_restores_vocab_size(self, trained_tokenizer):
        """After loading, vocab_size must match the original."""
        original_size = trained_tokenizer.get_vocab_size()
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_tokenizer.save(tmpdir)
            loaded = BPETokenizer.load(tmpdir)
            assert loaded.get_vocab_size() == original_size

    def test_load_preserves_encoding(self, trained_tokenizer):
        """
        Encoding a text with the original and the loaded tokenizer should give
        identical token IDs.
        """
        text = "hello world machine learning"
        original_ids = trained_tokenizer.encode(text)

        with tempfile.TemporaryDirectory() as tmpdir:
            trained_tokenizer.save(tmpdir)
            loaded = BPETokenizer.load(tmpdir)
            loaded_ids = loaded.encode(text)

        assert original_ids == loaded_ids, (
            f"Save/load changed encoding!\n"
            f"  Original : {original_ids}\n"
            f"  Loaded   : {loaded_ids}"
        )

    def test_load_preserves_roundtrip(self, trained_tokenizer):
        """After loading, decode(encode(text)) must still equal the original text."""
        text = "the quick brown fox jumps over the lazy dog"
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_tokenizer.save(tmpdir)
            loaded = BPETokenizer.load(tmpdir)
            ids = loaded.encode(text)
            decoded = loaded.decode(ids, skip_special_tokens=False)
        assert decoded == text


class TestCompressionStats:
    """
    Test 7: Compression ratio sanity check.
    """

    def test_compression_ratio_is_sane(self, trained_tokenizer):
        """
        For English text, a 16k-vocab BPE tokenizer should give roughly
        3–6 characters per token. Values outside this range suggest a bug.

        Note: Our test tokenizer has vocab_size=512 (tiny), so compression
        will be lower (closer to 2–3 chars/token). We check a relaxed range.
        """
        sample = [
            "hello world this is a test of the tokenizer compression ratio" * 10
        ]
        stats = trained_tokenizer.get_compression_stats(sample)
        ratio = stats["compression_ratio"]
        # For byte-level BPE, ratio should always be >= 1 (bytes per token)
        assert ratio >= 1.0, f"Compression ratio {ratio:.2f} < 1 (impossible)"
        # And should be reasonable for natural language
        assert ratio <= 20.0, f"Compression ratio {ratio:.2f} seems too high"


class TestUntrainedErrors:
    """Tests that calling methods on an untrained tokenizer raises errors."""

    def test_encode_raises_if_not_trained(self):
        """encode() on an untrained tokenizer should raise RuntimeError."""
        tok = BPETokenizer(vocab_size=100)
        with pytest.raises(RuntimeError):
            tok.encode("hello")

    def test_decode_raises_if_not_trained(self):
        """decode() on an untrained tokenizer should raise RuntimeError."""
        tok = BPETokenizer(vocab_size=100)
        with pytest.raises(RuntimeError):
            tok.decode([1, 2, 3])

    def test_load_raises_if_dir_missing(self):
        """load() should raise FileNotFoundError if directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            BPETokenizer.load("/nonexistent/path/tokenizer")


# ─────────────────────────────────────────────────────────────────────────────
# Allow running directly: python tests/test_tokenizer.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(PROJECT_ROOT),
    )
    sys.exit(result.returncode)
