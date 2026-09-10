# BrainSeg Command Runbook

Chronological command reference for the backend-only BrainSeg project.

Run commands from the repository root:

```powershell
cd E:\BrainSeg
```

All commands below assume Windows PowerShell and the Conda environment named `brainseg` unless stated otherwise.

## 1. Environment Setup

Create the dedicated Python environment:

```powershell
conda create -n brainseg python=3.12 -y
conda activate brainseg
```

Install the CUDA-compatible PyTorch build for the local GPU using the official PyTorch selector, then install project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify Python, PyTorch, CUDA, and the GPU:

```powershell
python --version
python scripts/verify_gpu.py
```

Expected target environment:

- Python 3.12
- RTX 5070
- `torch.cuda.is_available() == true`
- CUDA-capable PyTorch build

Copy local configuration and edit it with local credentials/URLs. Never commit the resulting `.env`:

```powershell
Copy-Item .env.example .env
```

## 2. Docker Backend Services

Validate Compose syntax without starting anything:

```powershell
docker compose config --quiet
```

Start PostgreSQL, Redis, and MLflow:

```powershell
docker compose up -d postgres redis mlflow
```

Start the backend API:

```powershell
docker compose up -d --build api
```

Check service status:

```powershell
docker compose ps
```

Expected backend services:

- PostgreSQL: host port `5433`
- Redis: host port `6379`
- MLflow: host port `5000`
- FastAPI: host port `8000`

Check API and MLflow health:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health
Invoke-WebRequest -UseBasicParsing http://localhost:5000/version
```

Inspect backend logs:

```powershell
docker compose logs --tail 100 postgres redis mlflow api
```

Stop services without deleting persistent volumes:

```powershell
docker compose down
```

Do not use `docker compose down -v` unless you intentionally want to delete PostgreSQL, Redis, and MLflow volumes.

## 3. Dataset Acquisition

Configure Kaggle credentials in `.env` without committing them:

```text
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
KAGGLE_DATASET=owner/dataset-slug
```

Download BraTS with live terminal progress. The download is approximately 42.8 GB and should only be run once:

```powershell
python scripts/download_data.py
```

The default destination is:

```text
data/raw
```

The script writes a completion marker only after a successful download:

```text
data/raw/.download-complete.json
```

Force a deliberate redownload:

```powershell
python scripts/download_data.py --force
```

Check the downloaded file count:

```powershell
$count = (Get-ChildItem data/raw -Recurse -File |
  Where-Object { $_.Name -match '\.nii(\.gz)?$|\.csv$' }).Count

"Dataset files: $count"
```

Expected dataset file count:

```text
2349
```

## 4. Manifest and Dataset Validation

Build the SHA-256 manifest. This may take several minutes because it hashes the full raw dataset:

```powershell
python scripts/build_manifest.py `
  data/raw `
  --output artifacts/data_manifest.json
```

Validate BraTS composition:

```powershell
python scripts/validate_dataset.py `
  artifacts/data_manifest.json `
  --output artifacts/dataset-validation.json
```

Expected strict validation:

- Training subjects: `369`
- Validation subjects: `125`
- Training NIfTI files: `1845`
- Validation NIfTI files: `500`
- CSV files: `4`
- No subject overlap

Verify sampled NIfTI geometry:

```powershell
python scripts/verify_volumes.py `
  data/raw `
  --sample-count 5 `
  --seed 42 `
  --output artifacts/volume-verification.json
```

Expected sampled volume shape:

240 x 240 x 155
```

Persist the manifest in PostgreSQL. Docker PostgreSQL uses host port `5433` because native Windows PostgreSQL owns `5432`:

```powershell
python scripts/persist_manifest.py `
  artifacts/data_manifest.json `
  --database-url "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg"
```

Running the same command again should return `status: skipped`.

## 5. Subject Splits and Preprocessing

Create the deterministic subject split:

```powershell
python scripts/create_splits.py `
  artifacts/data_manifest.json `
  --seed 42 `
  --output artifacts/split_manifest.json
```

Expected split:

- Train: `259`
- Validation: `55`
- Test: `55`

Validate split coverage and leakage:

```powershell
python scripts/validate_splits.py `
  artifacts/data_manifest.json `
  artifacts/split_manifest.json `
  --output artifacts/split-validation.json
```

Inspect a real multimodal subject:

```powershell
python scripts/inspect_subject.py `
  artifacts/data_manifest.json `
  BraTS20_Training_001
```

Expected image shape:

4 x 240 x 240 x 155
```

Inspect normalization and foreground cropping:

```powershell
python scripts/preprocess_subject.py `
  artifacts/data_manifest.json `
  BraTS20_Training_001 `
  --margin 8
```

Cache one subject:

```powershell
python scripts/cache_subject.py `
  artifacts/data_manifest.json `
  BraTS20_Training_001 `
  --cache-root data/cache `
  --margin 8
```

The first call should report `miss`; an identical second call should report `hit`.

Pre-cache all training subjects before full training:

```powershell
python scripts/precache_training.py `
  artifacts/data_manifest.json `
  artifacts/split_manifest.json `
  --cache-root data/cache
```

Persist the preprocessing dataset version:

```powershell
python scripts/persist_dataset_version.py `
  artifacts/data_manifest.json `
  --cache-root data/cache `
  --margin 8 `
  --database-url "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg"
```

## 6. Dataset Visualization

Generate split charts, label distributions, intensity histograms, and labeled subject montages:

```powershell
python scripts/visualize_dataset.py `
  artifacts/data_manifest.json `
  artifacts/split_manifest.json `
  --output-dir artifacts/dataset-visualization `
  --subjects 6 `
  --seed 42
```

Outputs:

- `REPORT.md`
- `summary.json`
- `split-composition.png`
- `label-distribution.png`
- `intensity-distributions.png`
- One labeled montage per sampled subject

## 7. Model Verification and Training

Verify the baseline 3D U-Net:

```powershell
python -m pytest tests/unit/test_unet3d.py -q
python scripts/verify_unet.py --patch-size 64 64 64
```

Verify the Attention U-Net:

```powershell
python -m pytest tests/unit/test_attention_unet3d.py -q
python scripts/verify_attention_unet.py --patch-size 64 64 64
```

Verify region conversion, metrics, losses, inference, and training utilities:

```powershell
python -m pytest tests/unit/test_regions.py tests/unit/test_metrics.py tests/unit/test_losses.py tests/unit/test_inference.py tests/unit/test_training.py -q
python scripts/inspect_regions.py artifacts/data_manifest.json BraTS20_Training_001 --cache-root data/cache
python scripts/verify_inference.py
python scripts/verify_training_utils.py
```

Verify patch sampling and lazy dataset access:

```powershell
python scripts/sample_patches.py artifacts/data_manifest.json BraTS20_Training_001 --cache-root data/cache --count 100 --patch-size 128 128 128 --seed 42
python scripts/inspect_dataset.py artifacts/data_manifest.json artifacts/split_manifest.json --split train --cache-root data/cache
```

Run baseline U-Net tests:

```powershell
python -m pytest tests/unit/test_unet3d.py -q
```

Run Attention U-Net tests:

```powershell
python -m pytest tests/unit/test_attention_unet3d.py -q
```

Run the CUDA model smoke test:

```powershell
python scripts/verify_unet.py --patch-size 64 64 64
python scripts/verify_attention_unet.py --patch-size 64 64 64
```

Run the sanity gate before model selection:

```powershell
python scripts/run_sanity_gate.py `
  artifacts/data_manifest.json `
  BraTS20_Training_001 `
  --cache-root data/cache `
  --output artifacts/sanity-gate.json `
  --epochs 500 `
  --learning-rate 0.01 `
  --patch-size 32 32 32
```

Run bounded Optuna selection:

```powershell
python scripts/run_model_selection.py `
  artifacts/data_manifest.json `
  --sanity-artifact artifacts/sanity-gate.json `
  --cache-root data/cache `
  --output artifacts/model-selection.json `
  --trials 2 `
  --epochs 1 `
  --max-gpu-seconds 600 `
  --tracking-uri http://localhost:5000
```

Train the canonical high-memory Attention U-Net:

```powershell
python scripts/train_full.py `
  artifacts/data_manifest.json `
  artifacts/split_manifest.json `
  artifacts/model-selection.json `
  --cache-root data/cache `
  --output-dir artifacts/full-training-high-memory `
  --epochs 10 `
  --profile high-memory `
  --num-workers 4 `
  --device cuda `
  --tracking-uri http://localhost:5000
```

The canonical checkpoint is:

artifacts/full-training-high-memory/best-model.pt
```

## 8. Evaluation and Artifacts

Evaluate two label volumes directly when a prediction NIfTI is available:

```powershell
python scripts/evaluate_labels.py prediction.nii target.nii
```

Inspect MLflow training/system metrics:

```powershell
python scripts/inspect_mlflow_system_metrics.py RUN_ID --tracking-uri http://localhost:5000
```

Register the canonical model:

```powershell
python scripts/register_model.py `
  artifacts/full-training-high-memory/training-result.json `
  --model-name brainseg-segmentation `
  --tracking-uri http://localhost:5000
```

Evaluate all 55 held-out subjects:

```powershell
python scripts/evaluate_full.py `
  artifacts/data_manifest.json `
  artifacts/split_manifest.json `
  artifacts/model-selection.json `
  artifacts/full-training-high-memory/best-model.pt `
  --cache-root data/cache `
  --output-dir artifacts/evaluation `
  --device cuda
```

Validate evaluation metrics:

```powershell
python scripts/validate_evaluation.py artifacts/evaluation/evaluation.json
```

Generate worst-case overlays:

```powershell
python scripts/generate_failure_gallery.py `
  artifacts/data_manifest.json `
  artifacts/evaluation/evaluation.json `
  artifacts/full-training-high-memory/training-result.json `
  artifacts/full-training-high-memory/best-model.pt `
  --cache-root data/cache `
  --output-dir artifacts/failure-gallery `
  --count 5
```

Generate TTA uncertainty:

```powershell
python scripts/estimate_tta_uncertainty.py `
  artifacts/data_manifest.json `
  artifacts/full-training-high-memory/training-result.json `
  artifacts/full-training-high-memory/best-model.pt `
  BraTS20_Training_275 `
  BraTS20_Training_310 `
  --cache-root data/cache `
  --output artifacts/uncertainty-tta.json
```

## 9. Local Inference

Provide a folder containing all four co-registered NIfTI modalities:

```text
subject-folder/
  subject_t1.nii
  subject_t1ce.nii
  subject_t2.nii
  subject_flair.nii
```

Run inference:

```powershell
python scripts/infer_subject.py `
  subject-folder `
  --output-dir artifacts/inference
```

Outputs:

- Full-size segmentation NIfTI
- Central-slice overlay PNG
- JSON latency/model summary

## 10. Backend Orchestration

Verify registered Celery tasks and orchestration policies:

```powershell
python scripts/verify_celery.py
python -m pytest tests/unit/test_celery.py tests/unit/test_policies.py tests/unit/test_pipeline_policies.py -q
```

Initialize PostgreSQL orchestration tables:

```powershell
python scripts/init_orchestration_db.py `
  --database-url "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg"
```

Start a Celery worker:

```powershell
python -m celery -A pipeline.tasks.celery_app.celery_app worker --loglevel=INFO --pool=solo
```

Trigger the post-training primary pipeline:

```powershell
python scripts/trigger_pipeline.py `
  --database-url "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg"
```

Verify idempotency:

```powershell
python scripts/verify_idempotency.py
```

Persist canonical primary-model lineage:

```powershell
python scripts/persist_primary_lineage.py `
  artifacts/primary-model.json `
  artifacts/data_manifest.json `
  --dataset-version-id c36885b173a5705fc5c17f1844e45400d5619e7a2a4a661b516671efca90eb10 `
  --database-url "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg"
```

Inspect PostgreSQL inference telemetry:

```powershell
python scripts/inspect_inference_events.py `
  --database-url "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg"
```

Run the API locally without Docker when the Conda environment and artifacts are available:

```powershell
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

In another terminal:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health
Invoke-WebRequest -UseBasicParsing http://localhost:8000/analytics/summary
Invoke-WebRequest -UseBasicParsing http://localhost:8000/analytics/model
Invoke-WebRequest -UseBasicParsing http://localhost:8000/viewer/subjects
```

## 11. Release Validation

Run all tests:

```powershell
python -m pytest tests -q
```

Validate the release index:

```powershell
python scripts/validate_release_index.py
```

Run the complete test suite with the project interpreter:

```powershell
python -m pytest tests -q
```

The project is backend-only. Supported interfaces are FastAPI, MLflow, PostgreSQL, Redis/Celery, and CLI/artifact workflows.
