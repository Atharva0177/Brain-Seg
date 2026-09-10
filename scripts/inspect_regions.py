"""Report WT/TC/ET voxel counts for one cached subject."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data.cache import preprocess_subject_cached
from pipeline.models.regions import labels_to_regions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("subject_id")
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subject, _ = preprocess_subject_cached(manifest, args.subject_id, args.cache_root)
    regions = labels_to_regions(subject.labels)
    print(json.dumps({name: int(mask.sum()) for name, mask in regions.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
