"""Register a trained checkpoint in the MLflow model registry."""

import argparse
import json
import os
import sys
from pathlib import Path

from mlflow.models import infer_signature
from mlflow.pytorch import log_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tracking.mlflow import TrackingConfig, start_run
from pipeline.training.full import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_result", type=Path)
    parser.add_argument("--model-name", default="brainseg-segmentation")
    parser.add_argument(
        "--tracking-uri", default=os.getenv("BRAINSEG_MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    args = parser.parse_args()
    result = json.loads(args.training_result.read_text(encoding="utf-8"))
    model = build_model(result["selected"])
    checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with start_run(
        TrackingConfig(args.tracking_uri),
        run_name="model-registration",
        tags={"purpose": "registry"},
    ) as run:
        example = torch.zeros(
            1,
            4,
            int(result["selected"]["patch_size"]),
            int(result["selected"]["patch_size"]),
            int(result["selected"]["patch_size"]),
        )
        log_model(
            model,
            artifact_path="model",
            registered_model_name=args.model_name,
            signature=infer_signature(example.numpy(), model(example).detach().numpy()),
        )
        print(
            json.dumps(
                {"status": "registered", "run_id": run.info.run_id, "model_name": args.model_name},
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    import torch

    raise SystemExit(main())
