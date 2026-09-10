"""Run a bounded Optuna model-selection smoke sweep."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tracking.mlflow import TrackingConfig
from pipeline.model_selection.search import SearchConfig, run_search


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--sanity-artifact", type=Path, default=Path("artifacts/sanity-gate.json"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-selection.json"))
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-gpu-seconds", type=float, default=600)
    parser.add_argument(
        "--tracking-uri", default=os.getenv("BRAINSEG_MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = run_search(
        manifest,
        args.sanity_artifact,
        cache_root=args.cache_root,
        tracking=TrackingConfig(args.tracking_uri),
        config=SearchConfig(
            trial_count=args.trials, trial_epochs=args.epochs, max_gpu_seconds=args.max_gpu_seconds
        ),
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "best_trial": result["best_trial"],
                "trial_count": len(result["trials"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
