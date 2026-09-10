"""CLI wrapper for deterministic subject-level dataset splits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data.splits import main

if __name__ == "__main__":
    raise SystemExit(main())
