"""Generate prediction overlays for the worst held-out subjects."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.training.artifacts import generate_failure_gallery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("training_result", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/failure-gallery"))
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    result = generate_failure_gallery(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.evaluation.read_text(encoding="utf-8")),
        json.loads(args.training_result.read_text(encoding="utf-8"))["selected"],
        args.checkpoint,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        count=args.count,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
