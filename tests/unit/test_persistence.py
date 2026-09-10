from app.db.persistence import manifest_id


def test_manifest_id_ignores_generation_metadata() -> None:
    base = {
        "schema_version": 1,
        "file_count": 1,
        "total_size_bytes": 4,
        "subject_count": 1,
        "files": [{"path": "a.nii", "sha256": "hash"}],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "root": "C:/one",
    }
    changed_metadata = {**base, "generated_at": "2026-01-02T00:00:00+00:00", "root": "D:/two"}
    assert manifest_id(base) == manifest_id(changed_metadata)
