import json
from pathlib import Path

import pytest

from pipeline.models.gate import require_passed_sanity_gate, require_sanity_artifact


def test_failed_sanity_gate_blocks_selection() -> None:
    with pytest.raises(RuntimeError, match="blocked"):
        require_passed_sanity_gate({"status": "failed"})


def test_missing_sanity_artifact_blocks_selection(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing"):
        require_sanity_artifact(tmp_path / "missing.json")


def test_passed_sanity_artifact_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "sanity.json"
    path.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    assert require_sanity_artifact(path)["status"] == "passed"
