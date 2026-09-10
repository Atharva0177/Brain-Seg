"""Persist a generated data manifest in PostgreSQL."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.persistence import load_manifest, persist_manifest


def load_local_env() -> dict[str, str]:
    values = dict(os.environ)
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in values:
            values[key] = value
    return values


def main() -> int:
    environment = load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--dataset",
        default=environment.get("KAGGLE_DATASET")
        or environment.get("BRAINSEG_KAGGLE_DATASET", "unknown"),
    )
    parser.add_argument(
        "--database-url",
        default=environment.get(
            "BRAINSEG_DATABASE_URL",
            "postgresql+psycopg://brainseg:brainseg@localhost:5432/brainseg",
        ),
    )
    args = parser.parse_args()
    result = persist_manifest(load_manifest(args.manifest), args.database_url, args.dataset)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
