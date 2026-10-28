"""Make the RI scripts importable by name, as they are when run from their directory."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
