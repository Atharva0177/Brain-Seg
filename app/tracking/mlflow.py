"""Centralized MLflow configuration for BrainSeg experiments."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow


def _configure_console_encoding() -> None:
    """Prevent MLflow's Unicode run links from breaking Windows cp1252 shells."""

    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class TrackingConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "brainseg"
    log_system_metrics: bool = True
    system_metrics_interval_seconds: float = 1.0
    system_metrics_samples_before_logging: int = 1


def configure_tracking(config: TrackingConfig) -> str:
    """Set the tracking URI and return the resolved MLflow experiment ID."""

    _configure_console_encoding()
    # Set these before start_run creates MLflow's background monitor. A short
    # smoke run should still emit at least one system-metrics sample.
    import os

    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = str(
        config.system_metrics_interval_seconds
    )
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING"] = str(
        config.system_metrics_samples_before_logging
    )
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(config.experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(config.experiment_name)
    return str(experiment_id)


@contextmanager
def start_run(
    config: TrackingConfig,
    *,
    run_name: str,
    tags: dict[str, str] | None = None,
    parameters: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Start a consistently configured run and log its initial metadata."""

    experiment_id = configure_tracking(config)
    with mlflow.start_run(
        run_name=run_name,
        tags=tags,
        log_system_metrics=config.log_system_metrics,
    ) as run:
        mlflow.set_tag("brainseg.experiment_id", experiment_id)
        if parameters:
            mlflow.log_params({key: str(value) for key, value in parameters.items()})
        yield run


def log_json_artifact(payload: dict[str, Any], artifact_file: str) -> Path:
    """Write a JSON artifact in a temporary file and log it to the active run."""

    import json
    import tempfile

    with tempfile.TemporaryDirectory(prefix="brainseg-mlflow-") as directory:
        path = Path(directory) / artifact_file
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(path), artifact_path="diagnostics")
    return path
