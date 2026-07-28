"""Pytest path setup: make the ``python/`` directory importable so the tests
can ``import tensixflow`` regardless of the invocation directory."""

import sys
from pathlib import Path

# parents: [0]=tests, [1]=tensixflow, [2]=python
_PYTHON_DIR = Path(__file__).resolve().parents[2]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
