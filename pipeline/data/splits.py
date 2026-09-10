"""Create reproducible subject-level splits from a verified BraTS manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    def validate(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(fraction <= 0 for fraction in fractions):
            raise ValueError("Split fractions must all be greater than zero.")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("Split fractions must sum to 1.0.")


def _cohort(entry: dict[str, Any]) -> str | None:
    subject = str(entry.get("subject_id") or "")
    path = str(entry.get("path") or "").lower()
    if "validation" in path or "validation" in subject.lower():
        return "validation"
    if "training" in path or "training" in subject.lower():
        return "training"
    return None


def _manifest_identity(manifest: dict[str, Any]) -> str:
    identity = {
        "file_count": manifest.get("file_count"),
        "total_size_bytes": manifest.get("total_size_bytes"),
        "files": [
            {"path": entry.get("path"), "sha256": entry.get("sha256")}
            for entry in manifest.get("files", [])
            if isinstance(entry, dict)
        ],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def labeled_training_subjects(manifest: dict[str, Any]) -> list[str]:
    """Return training subjects with a segmentation mask, sorted and deduplicated."""

    subjects = {
        str(entry["subject_id"])
        for entry in manifest.get("files", [])
        if isinstance(entry, dict)
        and entry.get("file_type") == "seg"
        and _cohort(entry) == "training"
        and entry.get("subject_id")
    }
    return sorted(subjects)


def create_subject_splits(
    manifest: dict[str, Any], config: SplitConfig = SplitConfig()
) -> dict[str, Any]:
    config.validate()
    subjects = labeled_training_subjects(manifest)
    if len(subjects) != 369:
        raise ValueError(
            f"Expected 369 labeled training subjects, found {len(subjects)}. "
            "Run strict dataset validation before splitting."
        )

    shuffled = list(subjects)
    random.Random(config.seed).shuffle(shuffled)
    validation_count = round(len(shuffled) * config.validation_fraction)
    test_count = round(len(shuffled) * config.test_fraction)
    train_count = len(shuffled) - validation_count - test_count

    splits = {
        "train": sorted(shuffled[:train_count]),
        "validation": sorted(shuffled[train_count : train_count + validation_count]),
        "test": sorted(shuffled[train_count + validation_count :]),
    }
    if len({subject for values in splits.values() for subject in values}) != len(subjects):
        raise RuntimeError("Subject split construction produced duplicate assignments.")

    return {
        "schema_version": 1,
        "source_manifest_id": _manifest_identity(manifest),
        "seed": config.seed,
        "fractions": {
            "train": config.train_fraction,
            "validation": config.validation_fraction,
            "test": config.test_fraction,
        },
        "counts": {name: len(values) for name, values in splits.items()},
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/split_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = create_subject_splits(manifest, SplitConfig(seed=args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "splits"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
