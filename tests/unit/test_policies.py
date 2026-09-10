from pathlib import Path

from pipeline.tasks.policies import artifact_is_current, content_hash, write_artifact_metadata


def test_content_hash_is_deterministic() -> None:
    assert content_hash({"b": 2, "a": 1}) == content_hash({"a": 1, "b": 2})


def test_artifact_hash_policy(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    assert not artifact_is_current(artifact, "hash")
    write_artifact_metadata(artifact, "hash")
    assert artifact_is_current(artifact, "hash")
