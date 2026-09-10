"""Validate the release artifact index and source-controlled release metadata."""

import json
from pathlib import Path


def main() -> int:
    index_path = Path("artifacts/release-index.json")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    required = [
        "canonical_model",
        "checkpoint",
        "training_result",
        "evaluation",
        "failure_gallery",
        "uncertainty",
        "dataset_manifest",
        "split_manifest",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"Release index missing keys: {missing}")
    print(
        json.dumps(
            {"status": "passed", "model": data["canonical_model"], "required_keys": len(required)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
