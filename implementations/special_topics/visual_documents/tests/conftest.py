"""Make ``vdocs`` importable, as it is when a notebook runs from its folder."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
