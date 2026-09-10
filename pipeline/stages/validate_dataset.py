"""Validate BraTS dataset composition before preprocessing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetExpectations:
    training_subjects: int = 369
    validation_subjects: int = 125
    training_nifti_files: int = 1845
    validation_nifti_files: int = 500
    csv_files: int = 4


def _cohort(entry: dict[str, object]) -> str | None:
    subject = str(entry.get("subject_id") or "")
    path = str(entry.get("path") or "").lower()
    if "validation" in path or "validation" in subject.lower():
        return "validation"
    if "training" in path or "training" in subject.lower():
        return "training"
    return None


def validate_manifest(
    manifest: dict[str, object],
    expectations: DatasetExpectations = DatasetExpectations(),
    *,
    strict: bool = True,
) -> dict[str, object]:
    entries = [entry for entry in manifest.get("files", []) if isinstance(entry, dict)]
    nifti_entries = [entry for entry in entries if entry.get("file_type") != "csv"]
    csv_entries = [entry for entry in entries if entry.get("file_type") == "csv"]
    training_nifti = [entry for entry in nifti_entries if _cohort(entry) == "training"]
    validation_nifti = [entry for entry in nifti_entries if _cohort(entry) == "validation"]
    training_subjects = {entry.get("subject_id") for entry in training_nifti}
    validation_subjects = {entry.get("subject_id") for entry in validation_nifti}

    checks = {
        "training_subjects": len(training_subjects) == expectations.training_subjects,
        "validation_subjects": len(validation_subjects) == expectations.validation_subjects,
        "training_nifti_files": len(training_nifti) == expectations.training_nifti_files,
        "validation_nifti_files": len(validation_nifti) == expectations.validation_nifti_files,
        "csv_files": len(csv_entries) == expectations.csv_files,
        "disjoint_subjects": training_subjects.isdisjoint(validation_subjects),
    }
    failures = [name for name, passed in checks.items() if not passed]
    result = {
        "status": "passed" if not failures else "failed",
        "strict": strict,
        "checks": checks,
        "failures": failures,
        "observed": {
            "training_subjects": len(training_subjects),
            "validation_subjects": len(validation_subjects),
            "training_nifti_files": len(training_nifti),
            "validation_nifti_files": len(validation_nifti),
            "csv_files": len(csv_entries),
        },
        "expected": asdict(expectations),
    }
    if strict and failures:
        raise ValueError(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset-validation.json"))
    parser.add_argument("--relaxed", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = validate_manifest(manifest, strict=not args.relaxed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
