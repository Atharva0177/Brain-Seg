"""Validate subject split coverage, leakage, cohort membership, and determinism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .splits import SplitConfig, create_subject_splits, labeled_training_subjects


def validate_splits(manifest: dict[str, Any], split_manifest: dict[str, Any]) -> dict[str, Any]:
    expected_subjects = set(labeled_training_subjects(manifest))
    splits = split_manifest.get("splits", {})
    train = set(splits.get("train", []))
    validation = set(splits.get("validation", []))
    test = set(splits.get("test", []))
    split_sets = [train, validation, test]

    regenerated = create_subject_splits(
        manifest,
        SplitConfig(
            seed=int(split_manifest["seed"]),
            train_fraction=float(split_manifest["fractions"]["train"]),
            validation_fraction=float(split_manifest["fractions"]["validation"]),
            test_fraction=float(split_manifest["fractions"]["test"]),
        ),
    )

    checks = {
        "all_subjects_are_labeled_training_subjects": all(
            current <= expected_subjects for current in split_sets
        ),
        "full_subject_coverage": train | validation | test == expected_subjects,
        "train_validation_disjoint": train.isdisjoint(validation),
        "train_test_disjoint": train.isdisjoint(test),
        "validation_test_disjoint": validation.isdisjoint(test),
        "deterministic_regeneration": regenerated["splits"] == splits,
        "recorded_counts_match": split_manifest.get("counts")
        == {"train": len(train), "validation": len(validation), "test": len(test)},
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "observed_counts": {
            "expected_labeled_training": len(expected_subjects),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/split-validation.json")
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    result = validate_splits(manifest, split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
