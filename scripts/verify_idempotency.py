"""Verify content-hash stage idempotency through the Celery worker."""

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
from pipeline.tasks.pipeline import idempotent_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/idempotency-check.json"))
    parser.add_argument("--async", dest="asynchronous", action="store_true")
    args = parser.parse_args()
    run_id = uuid.uuid4().hex
    database_url = os.getenv(
        "BRAINSEG_DATABASE_URL",
        "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg",
    )
    engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.add(
            PipelineRun(
                run_id=run_id,
                trigger="idempotency-test",
                status="running",
                started_at=datetime.now(UTC),
                metadata_json={"purpose": "BRATS-063"},
            )
        )
    artifact = Path("artifacts/idempotency-probe.txt")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("probe", encoding="utf-8")
    first_payload = {"input": "unchanged", "version": 1}
    second = idempotent_stage.delay(run_id, "probe", str(artifact), first_payload)
    first_result = second.get(timeout=60) if not args.asynchronous else {"task_id": second.id}
    # Write matching metadata as a stand-in for a completed stage output.
    from pipeline.tasks.policies import content_hash, write_artifact_metadata

    write_artifact_metadata(artifact, content_hash(first_payload))
    if args.asynchronous:
        result = {"status": "submitted", "task_id": second.id}
    else:
        third = idempotent_stage.delay(run_id, "probe", str(artifact), first_payload)
        unchanged_result = third.get(timeout=60)
        changed = idempotent_stage.delay(
            run_id, "probe", str(artifact), {"input": "changed", "version": 2}
        )
        changed_result = changed.get(timeout=60)
        result = {
            "status": "completed",
            "first": first_result,
            "unchanged": unchanged_result,
            "changed": changed_result,
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
