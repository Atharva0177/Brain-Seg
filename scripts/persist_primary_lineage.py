"""Persist the canonical primary-model lineage in PostgreSQL."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.lineage import persist_primary_lineage
from app.db.persistence import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_model", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-version-id", required=True)
    parser.add_argument("--database-url", default=os.getenv("BRAINSEG_DATABASE_URL"))
    args = parser.parse_args()
    if not args.database_url:
        raise RuntimeError("BRAINSEG_DATABASE_URL is required.")
    result = persist_primary_lineage(
        args.database_url,
        json.loads(args.primary_model.read_text(encoding="utf-8")),
        load_manifest(args.manifest),
        args.dataset_version_id,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
