"""Enforce sanity-gate results before model selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def require_passed_sanity_gate(result: dict[str, Any]) -> None:
    if result.get("status") != "passed":
        raise RuntimeError(
            "Model selection is blocked: the overfit sanity gate did not pass. "
            f"status={result.get('status', 'missing')}"
        )


def require_sanity_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Model selection is blocked: sanity artifact is missing: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Model selection is blocked: invalid sanity artifact: {path}"
        ) from error
    require_passed_sanity_gate(result)
    return result
