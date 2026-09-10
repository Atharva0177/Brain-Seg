"""Canonical BrainSeg model loading and inference service."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base
from app.db.orchestration import InferenceEvent
from pipeline.models.inference import sliding_window_logits
from pipeline.models.regions import logits_to_labels
from pipeline.models.training import load_checkpoint
from pipeline.training.full import build_model


class InferenceService:
    def __init__(
        self,
        metadata_path: Path | None = None,
        tracking_uri: str | None = None,
        model_alias: str | None = None,
    ) -> None:
        metadata_path = metadata_path or Path(
            os.getenv("BRAINSEG_MODEL_METADATA", "artifacts/primary-model.json")
        )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_source = "local_checkpoint"
        self.model = build_model(self.metadata).to(self.device)
        registry_uri = tracking_uri or os.getenv("BRAINSEG_MLFLOW_TRACKING_URI")
        if registry_uri:
            try:
                mlflow.set_tracking_uri(registry_uri)
                version = model_alias or str(self.metadata["mlflow_registered_version"])
                registered = mlflow.pytorch.load_model(
                    f"models:/{self.metadata['mlflow_registered_model']}/{version}",
                    map_location=self.device,
                )
                self.model = registered.to(self.device)
                self.model_source = "mlflow_registry"
            except Exception:
                load_checkpoint(
                    Path(self.metadata["checkpoint"]), self.model, map_location=self.device
                )
        else:
            load_checkpoint(Path(self.metadata["checkpoint"]), self.model, map_location=self.device)
        self.model.eval()

    def _record(self, status: str, latency_ms: float | None, error: str | None = None) -> None:
        database_url = os.getenv(
            "BRAINSEG_DATABASE_URL",
            "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg",
        )
        engine = create_engine(database_url, pool_pre_ping=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session, session.begin():
            session.add(
                InferenceEvent(
                    event_id=uuid.uuid4().hex,
                    model_name=self.metadata["mlflow_registered_model"],
                    model_version=str(self.metadata["mlflow_registered_version"]),
                    status=status,
                    latency_ms=latency_ms,
                    device=str(self.device),
                    error_message=error,
                    created_at=datetime.now(UTC),
                )
            )

    def predict(self, images: np.ndarray) -> dict[str, Any]:
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 4 or images.shape[0] != 4:
            raise ValueError("Expected images with shape (4, X, Y, Z).")
        started = time.perf_counter()
        inputs = torch.from_numpy(images).unsqueeze(0)
        try:
            logits = sliding_window_logits(
                self.model,
                inputs,
                window_size=(int(self.metadata["patch_size"]),) * 3,
                device=self.device,
            )
            labels = logits_to_labels(logits).squeeze(0).cpu().numpy().astype(np.uint8)
            latency_ms = (time.perf_counter() - started) * 1000
            self._record("completed", latency_ms)
            return {
                "segmentation": labels,
                "shape": list(labels.shape),
                "model_name": self.metadata["mlflow_registered_model"],
                "model_version": self.metadata["mlflow_registered_version"],
                "latency_ms": latency_ms,
                "device": str(self.device),
                "model_source": self.model_source,
            }
        except Exception as error:
            self._record("failed", (time.perf_counter() - started) * 1000, str(error))
            raise
