"""Measure foreground-biased patch sampling on one cached subject."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data.cache import preprocess_subject_cached
from pipeline.data.patches import sample_patches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("subject_id")
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subject, _ = preprocess_subject_cached(manifest, args.subject_id, args.cache_root)
    samples = sample_patches(
        subject.images,
        subject.labels,
        args.count,
        patch_size=tuple(args.patch_size),
        seed=args.seed,
    )
    foreground = sum(sample.center_type == "foreground" for sample in samples)
    print(
        json.dumps(
            {
                "subject_id": args.subject_id,
                "count": len(samples),
                "patch_size": args.patch_size,
                "foreground_center_count": foreground,
                "random_center_count": len(samples) - foreground,
                "foreground_center_fraction": foreground / len(samples),
                "mean_tumor_fraction": sum(sample.tumor_fraction for sample in samples)
                / len(samples),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
