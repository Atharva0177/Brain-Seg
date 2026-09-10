"""CLI wrapper for sampled NIfTI verification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.stages.verify_volumes import main

if __name__ == "__main__":
    raise SystemExit(main())
