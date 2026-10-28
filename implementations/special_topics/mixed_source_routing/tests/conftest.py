"""Import ``routing`` and ``build_reference`` from the notebook's folder."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
