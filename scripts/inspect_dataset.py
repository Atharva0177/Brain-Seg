"""Load one sample from the lazy cached dataset."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data.dataset import CachedSubjectDataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    args = parser.parse_args()
    dataset = CachedSubjectDataset(args.manifest, args.splits, args.split, args.cache_root)
    sample = dataset[0]
    print(
        json.dumps(
            {
                "length": len(dataset),
                "subject_id": sample["subject_id"],
                "image_shape": tuple(sample["image"].shape),
                "label_shape": tuple(sample["label"].shape),
                "cache_status": sample["cache_status"],
                "label_values": sorted(int(value) for value in sample["label"].unique()),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
