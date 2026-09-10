"""Run a real inference smoke test against the canonical model."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.inference import InferenceService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data_manifest.json"))
    parser.add_argument("--subject-id", default="BraTS20_Training_275")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    from pipeline.data.cache import preprocess_subject_cached

    subject, _ = preprocess_subject_cached(manifest, args.subject_id, args.subject_cache)
    result = InferenceService(tracking_uri=args.tracking_uri).predict(subject.images)
    print(
        json.dumps({key: value for key, value in result.items() if key != "segmentation"}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
