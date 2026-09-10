"""Retry, failure, and idempotency policies for pipeline tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class PipelineLogicError(RuntimeError):
    """Non-transient failure that must not be silently retried."""


def content_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_is_current(path: Path, expected_hash: str) -> bool:
    metadata = path.with_suffix(path.suffix + ".metadata.json")
    if not path.exists() or not metadata.exists():
        return False
    try:
        stored = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return stored.get("content_hash") == expected_hash


def write_artifact_metadata(path: Path, expected_hash: str) -> None:
    metadata = path.with_suffix(path.suffix + ".metadata.json")
    metadata.write_text(
        json.dumps({"content_hash": expected_hash}, indent=2) + "\n", encoding="utf-8"
    )
