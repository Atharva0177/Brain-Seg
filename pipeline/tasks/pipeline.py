"""Tracked Celery wrappers for the completed BrainSeg pipeline stages."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base
from app.db.orchestration import PipelineStage

from .celery_app import celery_app
from .policies import PipelineLogicError, artifact_is_current, content_hash

DATABASE_URL = os.getenv(
    "BRAINSEG_DATABASE_URL",
    "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg",
)


def _record_stage(run_id: str, name: str, status: str, started: datetime, **kwargs: Any) -> None:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        stage = session.query(PipelineStage).filter_by(run_id=run_id, stage_name=name).one_or_none()
        if stage is None:
            stage = PipelineStage(
                stage_id=f"{run_id}-{name}",
                run_id=run_id,
                stage_name=name,
                status=status,
                started_at=started,
            )
            session.add(stage)
        stage.status = status
        stage.finished_at = kwargs.get("finished_at")
        stage.duration_seconds = kwargs.get("duration_seconds")
        stage.resource_usage = kwargs.get("resource_usage", {})
        stage.error_message = kwargs.get("error_message")


def _stage_status(run_id: str, name: str) -> str | None:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        stage = session.query(PipelineStage).filter_by(run_id=run_id, stage_name=name).one_or_none()
        return stage.status if stage else None


def tracked_stage(run_id: str, name: str):
    def decorator(function):
        def wrapped(*args, **kwargs):
            started = datetime.now(UTC)
            start_clock = time.monotonic()
            _record_stage(run_id, name, "running", started)
            try:
                result = function(*args, **kwargs)
                finished = datetime.now(UTC)
                _record_stage(
                    run_id,
                    name,
                    "completed",
                    started,
                    finished_at=finished,
                    duration_seconds=time.monotonic() - start_clock,
                )
                return result
            except Exception as error:
                finished = datetime.now(UTC)
                _record_stage(
                    run_id,
                    name,
                    "failed",
                    started,
                    finished_at=finished,
                    duration_seconds=time.monotonic() - start_clock,
                    error_message=str(error),
                )
                raise

        return wrapped

    return decorator


@celery_app.task(bind=True, name="brainseg.health_check")
def health_check(self) -> dict[str, str]:
    return {"status": "ok", "task": self.name}


@celery_app.task(
    bind=True,
    name="brainseg.run_stage",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def run_stage(
    self, run_id: str, stage_name: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generic stage boundary used until each stage receives a dedicated task body."""

    return _execute_stage(run_id, stage_name, payload)


def _execute_stage(
    run_id: str, stage_name: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute one stage body without creating a nested Celery result wait."""

    started = datetime.now(UTC)
    clock = time.monotonic()
    _record_stage(run_id, stage_name, "running", started)
    try:
        result = {"status": "accepted", "stage": stage_name, "payload": payload or {}}
        _record_stage(
            run_id,
            stage_name,
            "completed",
            started,
            finished_at=datetime.now(UTC),
            duration_seconds=time.monotonic() - clock,
        )
        return result
    except Exception as error:
        _record_stage(
            run_id,
            stage_name,
            "failed",
            started,
            finished_at=datetime.now(UTC),
            duration_seconds=time.monotonic() - clock,
            error_message=str(error),
        )
        raise


@celery_app.task(bind=True, name="brainseg.run_ordered_stage")
def run_ordered_stage(
    self,
    run_id: str,
    stage_name: str,
    upstream_stages: list[str] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a stage only when every declared upstream stage completed."""

    upstream_stages = upstream_stages or []
    blocked = {
        stage: _stage_status(run_id, stage)
        for stage in upstream_stages
        if _stage_status(run_id, stage) != "completed"
    }
    if blocked:
        raise RuntimeError(f"Stage '{stage_name}' blocked by incomplete upstream stages: {blocked}")
    return _execute_stage(run_id, stage_name, payload)


@celery_app.task(bind=True, name="brainseg.idempotent_stage")
def idempotent_stage(
    self,
    run_id: str,
    stage_name: str,
    output_path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Skip an unchanged stage output or record a new deterministic execution."""

    expected_hash = content_hash(payload)
    output = Path(output_path)
    if artifact_is_current(output, expected_hash):
        _record_stage(
            run_id,
            stage_name,
            "skipped",
            datetime.now(UTC),
            finished_at=datetime.now(UTC),
            resource_usage={"reason": "content_hash_match"},
        )
        return {"status": "skipped", "stage": stage_name, "content_hash": expected_hash}
    if payload.get("logic_error"):
        raise PipelineLogicError(str(payload["logic_error"]))
    result = _execute_stage(run_id, stage_name, payload)
    return {**result, "content_hash": expected_hash}
