"""Import ``stability`` and the runner from the topic's folder, as the runner does."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
