"""Inspect normalization and cropping for one subject."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data.preprocess import normalize_and_crop, preprocessing_summary
from pipeline.data.subjects import load_subject


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("subject_id")
    parser.add_argument("--margin", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subject = load_subject(manifest, args.subject_id)
    result = normalize_and_crop(subject, margin=args.margin)
    print(json.dumps(preprocessing_summary(result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
