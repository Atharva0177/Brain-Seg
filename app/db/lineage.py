"""Persist the canonical model and artifact lineage."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import String, Text, create_engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from .models import Base
from .persistence import manifest_id


class ModelLineage(Base):
    __tablename__ = "model_lineage"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    manifest_id: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_version_id: Mapped[str] = mapped_column(String(64), nullable=False)
    mlflow_model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mlflow_model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation_path: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)


def persist_primary_lineage(
    database_url: str,
    primary_model: dict[str, Any],
    manifest: dict[str, Any],
    dataset_version_id: str,
) -> dict[str, Any]:
    engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    source_id = manifest_id(manifest)
    encoded = json.dumps(
        {"manifest_id": source_id, "primary_model": primary_model},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    model_id = hashlib.sha256(encoded).hexdigest()
    with Session(engine) as session, session.begin():
        if session.get(ModelLineage, model_id) is not None:
            return {"status": "skipped", "model_id": model_id}
        session.add(
            ModelLineage(
                model_id=model_id,
                manifest_id=source_id,
                dataset_version_id=dataset_version_id,
                mlflow_model_name=primary_model["mlflow_registered_model"],
                mlflow_model_version=str(primary_model["mlflow_registered_version"]),
                checkpoint_path=primary_model["checkpoint"],
                evaluation_path=primary_model["evaluation_artifact"],
                metadata_json=json.dumps(primary_model, sort_keys=True),
                created_at=datetime.now(UTC),
            )
        )
    return {"status": "persisted", "model_id": model_id, "manifest_id": source_id}
