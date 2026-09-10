import json
from pathlib import Path

import pytest

from pipeline.stages.download import DownloadConfig, download_dataset


def test_download_requires_dataset_slug(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="KAGGLE_DATASET"):
        download_dataset(
            DownloadConfig(dataset="", output_dir=tmp_path),
            environ={"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"},
        )


def test_download_dry_run_does_not_execute_or_write_marker(tmp_path: Path) -> None:
    result = download_dataset(
        DownloadConfig(dataset="owner/brats", output_dir=tmp_path),
        environ={"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"},
        dry_run=True,
    )
    assert result["status"] == "dry-run"
    assert result["command"][-1] == "--unzip"
    assert not (tmp_path / ".download-complete.json").exists()


def test_download_command_uses_supported_kaggle_dataset_flag(tmp_path: Path) -> None:
    result = download_dataset(
        DownloadConfig(dataset="owner/brats", output_dir=tmp_path),
        environ={"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"},
        dry_run=True,
    )
    assert result["command"][:4] == ["kaggle", "datasets", "download", "-d"]


def test_matching_marker_skips_download(tmp_path: Path) -> None:
    marker = tmp_path / ".download-complete.json"
    marker.write_text(json.dumps({"dataset": "owner/brats"}), encoding="utf-8")
    result = download_dataset(
        DownloadConfig(dataset="owner/brats", output_dir=tmp_path),
        environ={},
    )
    assert result["status"] == "skipped"
