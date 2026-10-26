# tests/conftest.py
import sys
from pathlib import Path

# Make the project root importable so `import aligner`, `import app`, …
# resolve correctly when pytest runs from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))