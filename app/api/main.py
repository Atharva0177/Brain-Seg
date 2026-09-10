"""FastAPI inference service for the canonical BrainSeg model."""

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from .inference import InferenceService

app = FastAPI(title="BrainSeg API", version="0.1.0")
service: InferenceService | None = None


class PredictionRequest(BaseModel):
    images: list[list[list[list[float]]]] = Field(..., description="T1, T1ce, T2, FLAIR channels")


def get_service() -> InferenceService:
    global service
    if service is None:
        service = InferenceService()
    return service


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _database_url() -> str:
    return os.getenv(
        "BRAINSEG_DATABASE_URL",
        "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg",
    )


def _artifact_root() -> Path:
    return Path(os.getenv("BRAINSEG_ARTIFACT_ROOT", "artifacts"))


def _artifact_path(relative: str) -> Path:
    path = _artifact_root() / relative
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "artifact_unavailable",
                "path": str(path),
                "message": "Mount generated BrainSeg artifacts into the API container.",
            },
        )
    return path


@app.get("/analytics/summary")
def analytics_summary() -> dict[str, Any]:
    engine = create_engine(_database_url(), pool_pre_ping=True)
    with engine.connect() as connection:
        runs = [
            dict(row._mapping)
            for row in connection.execute(
                text(
                    "select run_id, trigger, status, started_at, finished_at from pipeline_runs order by started_at desc limit 20"
                )
            )
        ]
        stages = [
            dict(row._mapping)
            for row in connection.execute(
                text(
                    "select run_id, stage_name, status, duration_seconds, started_at, finished_at from pipeline_stages order by started_at desc limit 50"
                )
            )
        ]
        inference = [
            dict(row._mapping)
            for row in connection.execute(
                text(
                    "select model_name, model_version, status, latency_ms, device, created_at from inference_events order by created_at desc limit 20"
                )
            )
        ]
    return {"runs": runs, "stages": stages, "inference": inference}


@app.get("/analytics/model")
def model_analytics() -> dict[str, Any]:
    primary = json.loads(_artifact_path("primary-model.json").read_text(encoding="utf-8"))
    training = json.loads(
        _artifact_path("full-training-high-memory/training-result.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        _artifact_path("evaluation/evaluation.json").read_text(encoding="utf-8")
    )
    selection_path = _artifact_root() / "model-selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        selection = {"best_trial": {"params": training["selected"]}}
    trials = []
    try:
        client = mlflow.tracking.MlflowClient(
            tracking_uri=os.getenv("BRAINSEG_MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        experiment = client.get_experiment_by_name("brainseg")
        if experiment:
            for run in client.search_runs(
                [experiment.experiment_id], order_by=["metrics.composite_dice DESC"]
            ):
                trials.append(
                    {
                        "run_id": run.info.run_id,
                        "status": run.info.status,
                        "params": run.data.params,
                        "metrics": run.data.metrics,
                    }
                )
    except Exception:
        pass
    return {
        "primary": primary,
        "training": training,
        "evaluation": evaluation["aggregate"],
        "trials": trials,
        "selection": selection.get("best_trial"),
    }


@app.get("/training/overview")
def training_overview() -> dict[str, Any]:
    primary = json.loads(_artifact_path("primary-model.json").read_text(encoding="utf-8"))
    training_path = _artifact_path("full-training-high-memory/training-result.json")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    evaluation = json.loads(
        _artifact_path("evaluation/evaluation.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(_artifact_path("data_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads(_artifact_path("dataset-validation.json").read_text(encoding="utf-8"))
    split = json.loads(_artifact_path("split_manifest.json").read_text(encoding="utf-8"))
    return {
        "dataset": {
            "files": manifest["file_count"],
            "subjects": manifest["subject_count"],
            "size_bytes": manifest["total_size_bytes"],
            "validation": validation["status"],
        },
        "preprocessing": {
            "cache_root": "data/cache",
            "dataset_version": "c36885b173a5705fc5c17f1844e45400d5619e7a2a4a661b516671efca90eb10",
            "status": "ready",
        },
        "splits": {name: len(values) for name, values in split["splits"].items()},
        "model": {
            key: primary[key]
            for key in (
                "model_name",
                "architecture",
                "channels",
                "patch_size",
                "training_subjects",
                "epochs",
            )
        },
        "training": training,
        "evaluation": evaluation["aggregate"],
    }


@app.post("/training/start")
def start_training() -> dict[str, Any]:
    import uuid

    from pipeline.tasks.pipeline import run_ordered_stage

    run_id = uuid.uuid4().hex
    stages = ["verify_primary_model", "evaluate_primary_model", "publish_primary_results"]
    task_ids = []
    previous: list[str] = []
    try:
        for stage in stages:
            task = run_ordered_stage.delay(run_id, stage, previous, {"mode": "primary-model"})
            task_ids.append(task.id)
            previous = [stage]
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail={"error": "pipeline_worker_unavailable", "message": str(error)},
        ) from error
    return {"status": "submitted", "run_id": run_id, "task_ids": task_ids}


@app.post("/training/config")
def save_training_config(config: dict[str, Any]) -> dict[str, Any]:
    allowed = {"epochs", "patch_size", "batch_size", "learning_rate"}
    unknown = set(config) - allowed
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unsupported parameters: {sorted(unknown)}")
    if "epochs" in config and not 1 <= int(config["epochs"]) <= 1000:
        raise HTTPException(status_code=400, detail="epochs must be between 1 and 1000")
    if "batch_size" in config and not 1 <= int(config["batch_size"]) <= 16:
        raise HTTPException(status_code=400, detail="batch_size must be between 1 and 16")
    path = _artifact_root() / "requested-training-config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "saved",
        "config": config,
        "note": "Saved for a future run; canonical model unchanged.",
    }


@app.get("/viewer/subjects")
def viewer_subjects(split_name: str = "test") -> dict[str, Any]:
    split_manifest = json.loads(_artifact_path("split_manifest.json").read_text(encoding="utf-8"))
    if split_name not in split_manifest["splits"]:
        raise HTTPException(status_code=400, detail="split must be train, validation, or test")
    return {"split": split_name, "subjects": split_manifest["splits"][split_name]}


@app.get("/viewer/{subject_id}/{slice_index}")
def viewer_slice(subject_id: str, slice_index: int) -> dict[str, Any]:
    from pipeline.data.cache import preprocess_subject_cached
    from pipeline.models.inference import sliding_window_logits
    from pipeline.models.regions import logits_to_labels
    from pipeline.training.artifacts import tta_uncertainty_for_subject

    manifest = json.loads(_artifact_path("data_manifest.json").read_text(encoding="utf-8"))
    subject, _ = preprocess_subject_cached(manifest, subject_id, Path("data/cache"))
    if slice_index < 0 or slice_index >= subject.labels.shape[2]:
        raise HTTPException(status_code=400, detail="slice_index is outside the subject volume")
    active_service = get_service()
    prediction = (
        logits_to_labels(
            sliding_window_logits(
                active_service.model,
                torch.from_numpy(subject.images).float().unsqueeze(0),
                window_size=(int(active_service.metadata["patch_size"]),) * 3,
                device=active_service.device,
            )
        )
        .squeeze(0)
        .cpu()
        .numpy()
    )
    image_slice = subject.images[:, :, :, slice_index]
    uncertainty = tta_uncertainty_for_subject(
        manifest,
        active_service.metadata,
        Path(active_service.metadata["checkpoint"]),
        subject_id,
        cache_root=Path("data/cache"),
        device=active_service.device,
    )
    return {
        "subject_id": subject_id,
        "slice_index": slice_index,
        "slice_count": int(subject.labels.shape[2]),
        "modalities": {
            name: image_slice[index].tolist()
            for index, name in enumerate(("t1", "t1ce", "t2", "flair"))
        },
        "ground_truth": subject.labels[:, :, slice_index].tolist(),
        "prediction": prediction[:, :, slice_index].tolist(),
        "uncertainty": {
            "entropy_mean": uncertainty["entropy_mean"],
            "entropy_max": uncertainty["entropy_max"],
        },
    }


@app.get("/about")
def about() -> dict[str, Any]:
    return {
        "project": "BrainSeg",
        "purpose": "Research and portfolio demonstration of automated 3D brain tumor segmentation.",
        "clinical_disclaimer": "Not a medical device. Not validated for diagnosis, treatment, or clinical decision-making.",
        "limitations": [
            "BraTS benchmark results do not establish clinical performance.",
            "Uncertainty estimates indicate model disagreement, not clinical risk.",
            "Performance varies by tumor region and acquisition/domain shift.",
        ],
        "citations": [
            "Menze et al. (2015), BRATS, IEEE TMI.",
            "Bakas et al. (2017), Scientific Data.",
            "Bakas et al. (2018), arXiv.",
        ],
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        result = get_service().predict(np.asarray(request.images, dtype=np.float32))
        segmentation = result.pop("segmentation")
        result["segmentation_values"] = np.unique(segmentation).astype(int).tolist()
        result["segmentation"] = segmentation.tolist()
        return result
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
