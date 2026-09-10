"""Load one multimodal subject and print its data/label summary."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data.subjects import composite_regions, load_subject


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("subject_id")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subject = load_subject(manifest, args.subject_id)
    regions = composite_regions(subject.labels)
    print(
        json.dumps(
            {
                "subject_id": subject.subject_id,
                "images_shape": subject.images.shape,
                "labels_shape": subject.labels.shape,
                "label_values": sorted(int(value) for value in set(subject.labels.flat)),
                "spacing": subject.spacing,
                "region_voxels": {name: int(mask.sum()) for name, mask in regions.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
