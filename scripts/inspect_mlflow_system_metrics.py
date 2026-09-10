"""List system metrics recorded for an MLflow run."""

from __future__ import annotations

import argparse
import json
import os

import mlflow
from mlflow.tracking import MlflowClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("BRAINSEG_MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    run = MlflowClient(tracking_uri=args.tracking_uri).get_run(args.run_id)
    system_metrics = {
        key: value
        for key, value in run.data.metrics.items()
        if key.startswith("system/") or "system" in key.lower()
    }
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": run.info.status,
                "metric_count": len(run.data.metrics),
                "system_metric_count": len(system_metrics),
                "system_metrics": system_metrics,
                "all_metric_keys": sorted(run.data.metrics),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
