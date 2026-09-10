"""Pre-cache every training subject before a full training run."""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data.cache import preprocess_subject_cached
from pipeline.training.data import load_training_subject_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subject_ids = load_training_subject_ids(args.splits)
    hits = misses = 0
    started = time.monotonic()
    for subject_id in tqdm(subject_ids, desc="pre-caching training subjects", unit="subject"):
        _, status = preprocess_subject_cached(manifest, subject_id, args.cache_root)
        if status == "hit":
            hits += 1
        else:
            misses += 1
    print(
        json.dumps(
            {
                "status": "completed",
                "subjects": len(subject_ids),
                "cache_hits": hits,
                "cache_misses": misses,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
