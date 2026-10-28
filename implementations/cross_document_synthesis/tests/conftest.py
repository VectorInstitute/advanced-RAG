"""Make the RI #2 implementation importable for its runner tests."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
