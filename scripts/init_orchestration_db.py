"""Create PostgreSQL orchestration tables for BrainSeg."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.orchestration import create_orchestration_schema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "BRAINSEG_DATABASE_URL",
            "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg",
        ),
    )
    args = parser.parse_args()
    create_orchestration_schema(args.database_url)
    print("Orchestration schema created or already exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
