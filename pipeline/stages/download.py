"""Unattended, idempotent BraTS dataset acquisition."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class DownloadConfig:
    dataset: str
    output_dir: Path = Path("data/raw")
    kaggle_command: str = "kaggle"


def _load_dotenv(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from the project .env without overriding env vars."""

    loaded = dict(os.environ if environ is None else environ)
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    env_path = next((path for path in candidates if path.is_file()), None)
    if env_path is None:
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in loaded:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        loaded[key] = value
    return loaded


def _credentials_present(environ: dict[str, str]) -> bool:
    return bool(
        (environ.get("KAGGLE_USERNAME") or environ.get("BRAINSEG_KAGGLE_USERNAME"))
        and (environ.get("KAGGLE_KEY") or environ.get("BRAINSEG_KAGGLE_KEY"))
    )


def _completion_path(output_dir: Path) -> Path:
    return output_dir / ".download-complete.json"


def _is_complete(config: DownloadConfig) -> bool:
    marker = _completion_path(config.output_dir)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("dataset") == config.dataset


def _download_command(config: DownloadConfig, destination: Path) -> list[str]:
    return [
        config.kaggle_command,
        "datasets",
        "download",
        "-d",
        config.dataset,
        "--path",
        str(destination),
        "--unzip",
    ]


def _staging_directory(output_dir: Path) -> Path:
    """Create staging beside the final output so archive/extraction stay on one drive."""

    staging = output_dir.parent / f".brainseg-download-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def download_dataset(
    config: DownloadConfig,
    *,
    environ: dict[str, str] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Download a dataset once and return an auditable stage result.

    The stage intentionally leaves checksums and expected file-count validation to
    the manifest stage. It never marks a failed or partial subprocess as complete.
    """

    environment = _load_dotenv(environ)
    output_dir = config.output_dir.resolve()

    resolved_config = DownloadConfig(
        dataset=config.dataset,
        output_dir=output_dir,
        kaggle_command=config.kaggle_command,
    )

    if not config.dataset.strip():
        raise ValueError("KAGGLE_DATASET must contain the Kaggle owner/dataset slug.")

    command = _download_command(resolved_config, output_dir)
    if dry_run:
        return {"status": "dry-run", "command": command, **asdict(resolved_config)}

    if _is_complete(resolved_config):
        print(f"Dataset already downloaded: {output_dir}")
        print("Nothing to download. Remove the completion marker to force a fresh acquisition.")
        return {
            "status": "skipped",
            "reason": "matching download marker",
            **asdict(resolved_config),
        }

    if not _credentials_present(environment):
        raise RuntimeError("Kaggle credentials are missing from the environment.")

    print(f"Downloading Kaggle dataset '{config.dataset}'")
    print(f"Destination: {output_dir}")
    print("Kaggle's live progress bar will appear below. Keep this terminal open.")

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_path = _staging_directory(output_dir)
    try:
        try:
            # Inherit terminal stdout/stderr so Kaggle's progress bar renders live.
            subprocess.run(
                _download_command(resolved_config, staging_path),
                check=True,
                env=environment,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "The Kaggle CLI was not found. Activate the brainseg Conda environment "
                "and install requirements.txt before running this script."
            ) from error
        except OSError as error:
            if getattr(error, "winerror", None) == 112 or getattr(error, "errno", None) == 28:
                raise RuntimeError(
                    f"Not enough disk space while downloading/extracting into {staging_path}. "
                    "Choose a drive with more free space using --output-dir."
                ) from error
            raise RuntimeError(
                "The Kaggle download process could not access its staging disk."
            ) from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Kaggle download failed. The dataset was not marked complete; "
                "fix the error and rerun the command."
            ) from error

        print(f"Download and extraction complete in {staging_path}.")
        print("Moving extracted files into data/raw...")
        for source in staging_path.iterdir():
            destination = output_dir / source.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(source), str(destination))
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path, ignore_errors=True)

    print("Dataset files moved successfully. Writing completion marker...")
    marker = {
        "dataset": config.dataset,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "status": "downloaded",
    }
    _completion_path(output_dir).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset ready at: {output_dir}")
    return {"status": "downloaded", **marker, **asdict(resolved_config)}


def main() -> int:
    environment = _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=environment.get("KAGGLE_DATASET") or environment.get("BRAINSEG_KAGGLE_DATASET", ""),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the completion marker and download again.",
    )
    args = parser.parse_args()

    if args.force and args.output_dir.exists():
        marker = _completion_path(args.output_dir.resolve())
        if marker.exists():
            marker.unlink()

    result = download_dataset(
        DownloadConfig(dataset=args.dataset, output_dir=args.output_dir), dry_run=args.dry_run
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
