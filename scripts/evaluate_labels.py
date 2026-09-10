"""Run region metrics for a prediction label NIfTI against a target label NIfTI."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.models.metrics import label_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError("nibabel is required for NIfTI metric evaluation.") from error
    prediction_image = nib.load(str(args.prediction))
    target_image = nib.load(str(args.target))
    prediction = prediction_image.get_fdata().astype("uint8")
    target = target_image.get_fdata().astype("uint8")
    spacing = target_image.header.get_zooms()[:3]
    print(json.dumps(label_metrics(prediction, target, spacing), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
