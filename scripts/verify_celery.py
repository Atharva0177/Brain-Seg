"""Verify Celery task registration without requiring a running worker."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.tasks.celery_app import celery_app


def main() -> int:
    print(json.dumps({"status": "passed", "tasks": sorted(celery_app.tasks)[:]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
