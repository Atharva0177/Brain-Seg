"""Verify MLflow health and create a sample BrainSeg tracking run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tracking.mlflow import TrackingConfig, log_json_artifact, start_run


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri", default=os.getenv("BRAINSEG_MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    parser.add_argument("--experiment", default="brainseg")
    args = parser.parse_args()
    config = TrackingConfig(args.tracking_uri, args.experiment)

    experiment_id = None
    with start_run(
        config,
        run_name="configuration-health-check",
        tags={"purpose": "infrastructure-verification", "project": "brainseg"},
        parameters={"phase": "BRATS-028", "source": "verify_mlflow.py"},
    ) as run:
        experiment_id = run.info.experiment_id
        mlflow.log_metric("health_check", 1.0)
        log_json_artifact({"status": "passed", "tracking_uri": args.tracking_uri}, "health.json")

    client = MlflowClient(tracking_uri=args.tracking_uri)
    stored = client.get_run(run.info.run_id)
    print(
        json.dumps(
            {
                "status": "passed",
                "tracking_uri": args.tracking_uri,
                "experiment_id": experiment_id,
                "run_id": run.info.run_id,
                "run_status": stored.info.status,
                "artifact_uri": stored.info.artifact_uri,
                "metric_health_check": stored.data.metrics.get("health_check"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
