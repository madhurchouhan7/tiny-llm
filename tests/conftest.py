# tests/conftest.py
"""
Pytest configuration for Windows DLL loading.
Ensures PyTorch's native DLL libraries in torch/lib are in the Windows DLL search path.
"""

import os
import sys
from pathlib import Path

# Add torch lib to Windows DLL directory search path
torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if torch_lib.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(torch_lib))
