"""Celery application configuration."""

import os

from celery import Celery

celery_app = Celery(
    "brainseg",
    broker=os.getenv("BRAINSEG_REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("BRAINSEG_REDIS_URL", "redis://localhost:6379/0"),
)
celery_app.conf.task_track_started = True
celery_app.conf.result_expires = 86400

# Import task definitions during app bootstrap so registration is deterministic
# for workers, CLI inspection, and unit tests.
from . import pipeline as _pipeline  # noqa: E402,F401
