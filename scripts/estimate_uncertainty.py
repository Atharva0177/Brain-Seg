"""Estimate MC-dropout uncertainty for selected held-out subjects."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.training.artifacts import uncertainty_for_subject


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("training_result", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("subject_ids", nargs="+")
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/uncertainty.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = json.loads(args.training_result.read_text(encoding="utf-8"))["selected"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [
        uncertainty_for_subject(
            manifest,
            selected,
            args.checkpoint,
            subject_id,
            cache_root=args.cache_root,
            device=device,
            passes=args.passes,
        )
        for subject_id in args.subject_ids
    ]
    payload = {"status": "completed", "results": results, "device": str(device)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
