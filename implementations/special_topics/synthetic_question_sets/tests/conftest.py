"""Make ``sqsets`` and the explore script importable, as from the notebook folder."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
