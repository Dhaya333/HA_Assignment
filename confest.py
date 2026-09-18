"""
Ensures the project root is on sys.path so `from src...` and
`from config...` imports resolve during test collection, regardless
of how pytest is invoked or which directory it's run from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))