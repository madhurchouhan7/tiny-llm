# src/tokenizer/__init__.py
# Makes 'src.tokenizer' a Python package.
# Exposes the main BPETokenizer class at the package level so callers can write:
#   from src.tokenizer import BPETokenizer
from .tokenizer import BPETokenizer

__all__ = ["BPETokenizer"]
