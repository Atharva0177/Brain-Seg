"""PostgreSQL orchestration state for BrainSeg pipeline runs and stages."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from .models import Base


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trigger: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    stages: Mapped[list[PipelineStage]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class PipelineStage(Base):
    __tablename__ = "pipeline_stages"

    stage_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    stage_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None] = mapped_column()
    resource_usage: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    run: Mapped[PipelineRun] = relationship(back_populates="stages")
    artifacts: Mapped[list[PipelineArtifact]] = relationship(
        back_populates="stage", cascade="all, delete-orphan"
    )


class PipelineArtifact(Base):
    __tablename__ = "pipeline_artifacts"

    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    stage_id: Mapped[str] = mapped_column(
        ForeignKey("pipeline_stages.stage_id", ondelete="CASCADE"), nullable=False
    )
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    stage: Mapped[PipelineStage] = relationship(back_populates="artifacts")


class InferenceEvent(Base):
    __tablename__ = "inference_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    latency_ms: Mapped[float | None] = mapped_column()
    device: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def create_orchestration_schema(database_url: str) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)


def create_pipeline_run(
    database_url: str, trigger: str, metadata: dict[str, Any] | None = None
) -> str:
    create_orchestration_schema(database_url)
    run_id = uuid.uuid4().hex
    with Session(create_engine(database_url, pool_pre_ping=True)) as session, session.begin():
        session.add(
            PipelineRun(
                run_id=run_id,
                trigger=trigger,
                status="running",
                started_at=datetime.now(UTC),
                metadata_json=metadata or {},
            )
        )
    return run_id
