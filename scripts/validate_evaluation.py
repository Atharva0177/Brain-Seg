"""Validate evaluation artifacts for suspicious or invalid results."""

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    result = json.loads(args.evaluation.read_text(encoding="utf-8"))
    failures = []
    if result.get("status") != "completed":
        failures.append("evaluation status is not completed")
    if not result.get("subject_results"):
        failures.append("no subject results")
    for subject in result.get("subject_results", []):
        for region, metrics in subject.get("metrics", {}).items():
            if not 0 <= metrics["dice"] <= 1:
                failures.append(f"invalid Dice for {subject['subject_id']}:{region}")
            if metrics["dice"] > 0.98:
                failures.append(f"suspiciously high Dice for {subject['subject_id']}:{region}")
            # Empty-region HD95 is reported separately; it must not contaminate
            # finite aggregate statistics but is not itself a fatal artifact error.
    for region, metrics in result.get("aggregate", {}).items():
        for metric_name in ("dice", "hd95"):
            for statistic_name in ("mean", "std"):
                value = float(metrics[metric_name][statistic_name])
                if not math.isfinite(value) and metrics.get("valid_count", 0) > 0:
                    failures.append(
                        f"non-finite aggregate {metric_name}.{statistic_name} for {region}"
                    )
    print(
        json.dumps(
            {"status": "passed" if not failures else "failed", "failures": failures}, indent=2
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
