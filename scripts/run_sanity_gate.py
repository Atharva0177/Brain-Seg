"""Run the overfit-single-batch sanity gate on a cached subject patch."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data.cache import preprocess_subject_cached
from pipeline.data.patches import sample_patch
from pipeline.models.sanity import SanityConfig, run_sanity_gate, write_sanity_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("subject_id")
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/sanity-gate.json"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-final-loss-ratio", type=float, default=0.30)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subject, _ = preprocess_subject_cached(manifest, args.subject_id, args.cache_root)
    patch = sample_patch(
        subject.images, subject.labels, tuple(args.patch_size), foreground_probability=1.0, rng=None
    )
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    result = run_sanity_gate(
        patch.image,
        patch.label,
        device,
        SanityConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_final_loss_ratio=args.max_final_loss_ratio,
        ),
    )
    write_sanity_result(result, args.output)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "device", "initial_loss", "final")}, indent=2
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
