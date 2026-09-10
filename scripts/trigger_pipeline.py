"""Trigger the primary BrainSeg pipeline through ordered Celery stages."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import Base
from app.db.orchestration import PipelineRun
from pipeline.tasks.pipeline import run_ordered_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "BRAINSEG_DATABASE_URL",
            "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg",
        ),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/full-training-high-memory/best-model.pt")
    )
    parser.add_argument("--async", dest="asynchronous", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Primary checkpoint not found: {args.checkpoint}")
    engine = create_engine(args.database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    run_id = uuid.uuid4().hex
    with Session(engine) as session, session.begin():
        session.add(
            PipelineRun(
                run_id=run_id,
                trigger="cli",
                status="running",
                started_at=datetime.now(UTC),
                metadata_json={
                    "checkpoint": str(args.checkpoint),
                    "model": "brainseg-segmentation:1",
                },
            )
        )

    stages = [
        ("verify_primary_model", []),
        ("evaluate_primary_model", ["verify_primary_model"]),
        ("publish_primary_results", ["evaluate_primary_model"]),
    ]
    results = []
    previous = None
    for name, dependencies in stages:
        dependencies = dependencies or ([previous] if previous else [])
        task = run_ordered_stage.delay(
            run_id, name, dependencies, {"checkpoint": str(args.checkpoint)}
        )
        results.append({"stage": name, "task_id": task.id})
        if not args.asynchronous:
            task.get(timeout=300)
        previous = name
    print(json.dumps({"status": "submitted", "run_id": run_id, "stages": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
