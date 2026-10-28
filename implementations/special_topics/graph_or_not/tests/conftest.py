"""Make ``gon`` and the runner importable, as from a notebook in this folder."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
