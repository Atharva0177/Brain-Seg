from pathlib import Path

import pytest

from pipeline.stages import verify_volumes


def test_verify_volume_reports_geometry_with_mocked_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    class Header:
        def get_zooms(self) -> tuple[float, ...]:
            return (1.0, 1.0, 1.0)

    class Image:
        shape = (240, 240, 155)
        affine = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        header = Header()

    monkeypatch.setattr(verify_volumes, "_load_nifti", lambda _: Image())
    result = verify_volumes.verify_volume(Path("sample.nii"))
    assert result["shape"] == (240, 240, 155)
    assert result["spacing"] == (1.0, 1.0, 1.0)


def test_missing_nibabel_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "nibabel", None)
    with pytest.raises(RuntimeError, match="nibabel is required"):
        verify_volumes._load_nifti(Path("sample.nii"))
