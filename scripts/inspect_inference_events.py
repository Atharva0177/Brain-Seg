"""Print recent PostgreSQL inference events."""

import argparse
import json
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db.models import Base
from app.db.orchestration import InferenceEvent  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "BRAINSEG_DATABASE_URL",
            "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg",
        ),
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    engine = create_engine(args.database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    query = text(
        "select model_name, model_version, status, latency_ms, device, created_at from inference_events order by created_at desc limit :limit"
    )
    with engine.connect() as connection:
        rows = [dict(row._mapping) for row in connection.execute(query, {"limit": args.limit})]
    print(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
