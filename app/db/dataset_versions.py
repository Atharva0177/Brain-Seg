"""Persist preprocessing dataset versions and lineage."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from .models import Base
from .persistence import manifest_id


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    manifest_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_manifests.manifest_id"), nullable=False
    )
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(Text, nullable=False)
    cache_root: Mapped[str] = mapped_column(Text, nullable=False)
    subject_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ready")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def preprocessing_config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_version_id(source_manifest_id: str, config_hash: str) -> str:
    return hashlib.sha256(f"{source_manifest_id}:{config_hash}".encode()).hexdigest()


def persist_dataset_version(
    manifest: dict[str, Any],
    database_url: str,
    config: dict[str, Any],
    cache_root: str,
) -> dict[str, Any]:
    engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    source_id = manifest_id(manifest)
    config_hash = preprocessing_config_hash(config)
    version_id = dataset_version_id(source_id, config_hash)
    with Session(engine) as session, session.begin():
        if session.get(DatasetVersion, version_id) is not None:
            return {"status": "skipped", "version_id": version_id, "reason": "already persisted"}
        session.add(
            DatasetVersion(
                version_id=version_id,
                manifest_id=source_id,
                config_hash=config_hash,
                config_json=json.dumps(config, sort_keys=True),
                cache_root=cache_root,
                subject_count=int(manifest.get("subject_count", 0)),
                status="ready",
                created_at=datetime.now(UTC),
            )
        )
    return {
        "status": "persisted",
        "version_id": version_id,
        "manifest_id": source_id,
        "config_hash": config_hash,
    }
