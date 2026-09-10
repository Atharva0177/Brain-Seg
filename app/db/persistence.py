"""Persist generated dataset manifests in PostgreSQL."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .models import Base, DatasetManifest, DatasetManifestFile


def manifest_id(manifest: dict[str, Any]) -> str:
    """Return a stable ID based on dataset contents, not generation time/path."""

    identity = {
        key: manifest.get(key)
        for key in ("schema_version", "file_count", "total_size_bytes", "subject_count", "files")
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def persist_manifest(manifest: dict[str, Any], database_url: str, dataset: str) -> dict[str, Any]:
    """Create or replace one content-addressed manifest and its file records."""

    engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    current_id = manifest_id(manifest)
    entries = manifest.get("files", [])

    with Session(engine) as session, session.begin():
        existing = session.get(DatasetManifest, current_id)
        if existing is not None:
            return {
                "status": "skipped",
                "manifest_id": current_id,
                "file_count": len(existing.files),
                "reason": "manifest already persisted",
            }

        record = DatasetManifest(
            manifest_id=current_id,
            dataset=dataset,
            root=str(manifest["root"]),
            status="verified",
            file_count=int(manifest["file_count"]),
            subject_count=int(manifest["subject_count"]),
            total_size_bytes=int(manifest["total_size_bytes"]),
            manifest_json=manifest,
            created_at=datetime.now(UTC),
        )
        record.files = [
            DatasetManifestFile(
                relative_path=str(entry["path"]),
                subject_id=entry.get("subject_id"),
                modality=entry.get("modality"),
                file_type=str(entry["file_type"]),
                size_bytes=int(entry["size_bytes"]),
                sha256=str(entry["sha256"]),
            )
            for entry in entries
        ]
        session.add(record)

    return {"status": "persisted", "manifest_id": current_id, "file_count": len(entries)}


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
