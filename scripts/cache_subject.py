"""Preprocess one subject into the content-addressed cache."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data.cache import preprocess_subject_cached


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("subject_id")
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--margin", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subject, status = preprocess_subject_cached(
        manifest, args.subject_id, args.cache_root, margin=args.margin
    )
    print(
        json.dumps(
            {
                "status": status,
                "subject_id": subject.subject_id,
                "images_shape": subject.images.shape,
                "labels_shape": subject.labels.shape,
                "cache_root": str(args.cache_root.resolve()),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
