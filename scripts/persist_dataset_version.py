"""Persist preprocessing configuration and dataset lineage in PostgreSQL."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.dataset_versions import persist_dataset_version
from app.db.persistence import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", default="data/cache")
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--database-url", default=os.getenv("BRAINSEG_DATABASE_URL"))
    args = parser.parse_args()
    if not args.database_url:
        raise RuntimeError("BRAINSEG_DATABASE_URL must be set for dataset-version persistence.")
    config = {
        "schema_version": 1,
        "normalization": "foreground-zscore-v1",
        "crop_margin": args.margin,
        "modalities": ["t1", "t1ce", "t2", "flair"],
    }
    result = persist_dataset_version(
        load_manifest(args.manifest), args.database_url, config, args.cache_root
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
