# BrainSeg

BrainSeg is a backend-only, end-to-end 3D brain-tumor segmentation and MLOps portfolio project built on the BraTS 2020 dataset.

The system downloads and validates multimodal MRI data, creates subject-safe splits, preprocesses and caches volumes, trains and evaluates a 3D Attention U-Net, registers the model in MLflow, records lineage in PostgreSQL, exposes inference through FastAPI, and produces reproducible visualization artifacts.

This repository is a research/portfolio system. It is **not a medical device** and must not be used for diagnosis, treatment, prognosis, or clinical decision-making.

## Project Status

- Dataset acquisition and integrity validation: complete
- Subject-level splitting and leakage validation: complete
- Preprocessing, caching, and patch sampling: complete
- Baseline U-Net and Attention U-Net implementation: complete
- Optuna model selection and MLflow tracking: complete
- Canonical model training and registration: complete
- Full held-out evaluation on 55 subjects: complete
- TTA uncertainty and failure gallery: complete
- FastAPI serving, PostgreSQL telemetry, Celery orchestration: complete
- Backend release tests and documentation: complete
- Frontend dashboard: intentionally removed from scope

## System Architecture

```mermaid
flowchart LR
    Trigger[CLI or API trigger]
    Redis[Redis broker]
    Celery[Celery worker]
    Postgres[(PostgreSQL<br/>runs, stages, lineage)]
    MLflow[MLflow<br/>metrics, artifacts, registry]
    Data[Raw BraTS data]
    Manifest[Verified manifest]
    Cache[Compressed subject cache]
    Model[Registered Attention U-Net]
    API[FastAPI inference/API]
    Artifacts[CLI artifacts<br/>NIfTI, PNG, JSON, reports]

    Trigger --> Redis --> Celery
    Celery --> Data --> Manifest --> Cache
    Celery --> Model
    Celery --> Postgres
    Celery --> MLflow
    Model --> API
    API --> Postgres
    API --> MLflow
    Celery --> Artifacts
```

## Data and Training Flow

```mermaid
flowchart TD
    Download[Kaggle download<br/>progress + same-drive staging]
    Hash[SHA-256 manifest]
    Validate[Exact BraTS validation]
    Geometry[Sampled NIfTI geometry check]
    Split[Subject split<br/>259 / 55 / 55]
    Normalize[Foreground z-score + crop]
    Cache[Content-addressed NPZ cache]
    Patches[Foreground-biased 3D patches]
    Search[Bounded Optuna search]
    Train[Full Attention U-Net training]
    Infer[Sliding-window inference]
    Eval[Dice + HD95 + TTA uncertainty]
    Register[MLflow model registry]

    Download --> Hash --> Validate --> Geometry --> Split
    Split --> Normalize --> Cache --> Patches --> Search --> Train
    Train --> Infer --> Eval --> Register
```

## Canonical Model

The deployed/reference model is the trained high-memory Attention U-Net:

| Property | Value |
|---|---|
| Architecture | Attention U-Net |
| Input channels | T1, T1ce, T2, FLAIR |
| Model channels | `32,64,128,256` |
| Patch size | `112^3` |
| Training subjects | `259` |
| Training epochs | `10` |
| Checkpoint | `artifacts/full-training-high-memory/best-model.pt` |
| MLflow model | `brainseg-segmentation` version `1` |
| Model metadata | `artifacts/primary-model.json` |

The baseline 3D U-Net is retained as a separate comparison model under `artifacts/full-training-baseline/`. It never overwrites the canonical Attention U-Net.

### Baseline 3D U-Net

The baseline is a conventional configurable 3D encoder-decoder with:

- Four input channels: T1, T1ce, T2, FLAIR
- Four output classes mapped to stored BraTS labels `0,1,2,4`
- Instance normalization
- Leaky ReLU activations
- Skip connections
- The same cached data, subject split, loss, patch pipeline, and full-volume evaluation used by the Attention U-Net

The baseline exists to measure the value of attention gates under the same experimental conditions. Its isolated artifacts are:

```text
artifacts/baseline-model.json
artifacts/full-training-baseline/best-model.pt
artifacts/full-training-baseline/training-result.json
artifacts/evaluation-baseline/evaluation.json
```

The baseline is a comparison model, not the registered serving model. The Attention U-Net remains canonical because it produced higher held-out Dice for WT, TC, and ET.

## Results

Evaluation uses full-volume sliding-window inference on the 55-subject held-out split.

| Region | Attention U-Net Dice | Attention U-Net HD95 | Baseline U-Net Dice |
|---|---:|---:|---:|
| Whole Tumor | `0.83091` | `28.5352` | `0.82533` |
| Tumor Core | `0.71299` | `21.1474` | `0.70460` |
| Enhancing Tumor | `0.67402` | `18.4937` | `0.66072` |

The Attention U-Net is better on all three regions in the completed comparison. Two ET subjects have non-finite HD95 because one side of the comparison is empty; they are recorded explicitly and excluded from finite aggregate HD95 statistics.

## Visual Evidence

Generated visual artifacts are external to source control. When restored under `artifacts/`, the following images/reports are available:

### Dataset Charts

![Subject split composition](artifacts/dataset-visualization/split-composition.png)

![Label distribution](artifacts/dataset-visualization/label-distribution.png)

![Modality intensity distributions](artifacts/dataset-visualization/intensity-distributions.png)

Dataset report:

```text
artifacts/dataset-visualization/REPORT.md
artifacts/dataset-visualization/summary.json
```

### Training Charts

Each full-training run writes:

```text
artifacts/<run>/training-curves.png
artifacts/<run>/prediction-previews/train-*.png
artifacts/<run>/prediction-previews/validation-*.png
artifacts/<run>/prediction-previews/test-*.png
```

The training curve contains loss, validation loss, epoch duration, and peak GPU memory. Preview images contain T1, ground truth, and prediction slices for each split.

Canonical model training chart:

![Attention U-Net training curves](artifacts/full-training-high-memory/training-curves.png)

Canonical split previews:

![Training split prediction](artifacts/full-training-high-memory/prediction-previews/train-BraTS20_Training_001.png)

![Validation split prediction](artifacts/full-training-high-memory/prediction-previews/validation-BraTS20_Training_024.png)

![Test split prediction](artifacts/full-training-high-memory/prediction-previews/test-BraTS20_Training_004.png)

Baseline model training chart, when the comparison artifacts are restored:

![Baseline U-Net training curves](artifacts/full-training-baseline/training-curves.png)

The JSON training result is the authoritative source for exact values:

```text
artifacts/full-training-high-memory/training-result.json
artifacts/full-training-baseline/training-result.json
```

It contains per-epoch loss, validation loss, duration, throughput, and peak GPU memory.

### Failure Gallery

![Failure case example](artifacts/failure-gallery/BraTS20_Training_275.png)

Gallery index:

```text
artifacts/failure-gallery/gallery.json
```

### Local Inference Output

The CLI inference workflow writes:

```text
artifacts/inference/<subject>_segmentation.nii.gz
artifacts/inference/<subject>_overlay.png
artifacts/inference/<subject>_summary.json
```

Inference output example:

![Inference segmentation overlay](artifacts/inference-training-001/BraTS20_Training_001_overlay.png)

Inference summaries contain model/version, device, output labels, volume shape, and latency. PostgreSQL stores the same serving events for charting/querying:

```text
scripts/inspect_inference_events.py
```

Inference data sources for charts:

- Latency: PostgreSQL `inference_events.latency_ms`
- Success/failure rate: PostgreSQL `inference_events.status`
- Model version: PostgreSQL `inference_events.model_version`
- Training loss/resources: `training-result.json` and MLflow run metrics
- Held-out Dice/HD95: `evaluation.json`

The backend does not commit binary images or generated charts. Restore the external `artifacts/` directory to render the image links above.

## Environment Setup

Create the Conda environment:

```powershell
conda create -n brainseg python=3.12 -y
conda activate brainseg
```

Install the CUDA-compatible PyTorch build for the local GPU using the official PyTorch selector, then install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Copy local configuration:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`, Kaggle credentials, raw data, caches, checkpoints, or generated artifacts.

The complete chronological command reference is available in:

```text
docs/COMMANDS.md
RUN_COMMANDS.md
```

## Dataset Setup

Download the BraTS dataset through Kaggle:

```powershell
python scripts/download_data.py
```

The download is approximately 42.8 GB and is staged on the configured output drive. It stores data under `data/raw` and shows Kaggle’s live progress bar.

Build and validate the dataset manifest:

```powershell
python scripts/build_manifest.py data/raw --output artifacts/data_manifest.json
python scripts/validate_dataset.py artifacts/data_manifest.json --output artifacts/dataset-validation.json
python scripts/verify_volumes.py data/raw --sample-count 5 --seed 42 --output artifacts/volume-verification.json
```

Create and validate subject splits:

```powershell
python scripts/create_splits.py artifacts/data_manifest.json --seed 42 --output artifacts/split_manifest.json
python scripts/validate_splits.py artifacts/data_manifest.json artifacts/split_manifest.json --output artifacts/split-validation.json
```

Pre-cache training subjects:

```powershell
python scripts/precache_training.py artifacts/data_manifest.json artifacts/split_manifest.json --cache-root data/cache
```

## Backend Services

Start PostgreSQL, Redis, MLflow, and FastAPI:

```powershell
docker compose up -d --build postgres redis mlflow api
```

Host ports:

- PostgreSQL: `5433`
- Redis: `6379`
- MLflow: `5000`
- FastAPI: `8000`

Check health:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health
Invoke-WebRequest -UseBasicParsing http://localhost:5000/version
```

The API requires generated artifacts mounted through `BRAINSEG_ARTIFACT_ROOT` for model analytics, viewer data, and prediction. `/health` remains available in a clean source checkout.

## Training and Evaluation

The canonical trained checkpoint already exists as an external artifact. Do not retrain it unless intentionally producing a new model version.

Train a separate baseline U-Net comparison model:

```powershell
python scripts/train_full.py `
  artifacts/data_manifest.json `
  artifacts/split_manifest.json `
  artifacts/model-selection.json `
  --cache-root data/cache `
  --output-dir artifacts/full-training-baseline `
  --epochs 10 `
  --profile high-memory `
  --architecture unet `
  --num-workers 4 `
  --device cuda `
  --tracking-uri http://localhost:5000
```

Evaluate a checkpoint:

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

Validate evaluation:

```powershell
python scripts/validate_evaluation.py artifacts/evaluation/evaluation.json
```

## Inference

The model requires all four co-registered MRI modalities:

```text
subject-folder/
  subject_t1.nii
  subject_t1ce.nii
  subject_t2.nii
  subject_flair.nii
```

Run inference:

```powershell
python scripts/infer_subject.py subject-folder --output-dir artifacts/inference
```

The command writes a full-size BraTS-label segmentation NIfTI, an overlay PNG, and a JSON summary.

## Orchestration

Initialize orchestration tables:

```powershell
python scripts/init_orchestration_db.py --database-url "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg"
```

Start a Celery worker:

```powershell
python -m celery -A pipeline.tasks.celery_app.celery_app worker --loglevel=INFO --pool=solo
```

Trigger the backend pipeline without retraining:

```powershell
python scripts/trigger_pipeline.py --database-url "postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg"
```

## Release Validation

Run tests:

```powershell
python -m pytest tests -q
```

Validate release metadata:

```powershell
python scripts/validate_release_index.py
```

## Citations

Any public presentation of BraTS results should cite:

- Menze et al. (2015), “The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS),” IEEE Transactions on Medical Imaging.
- Bakas et al. (2017), “Advancing The Cancer Genome Atlas glioma MRI collections,” Scientific Data.
- Bakas et al. (2018), “Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation,” arXiv.

## Limitations and Disclaimer

BrainSeg is a research and portfolio project, not a medical device. It is not validated for diagnosis, treatment, prognosis, or clinical decision-making. Benchmark performance does not establish clinical performance. Uncertainty values represent model disagreement and are not clinical risk estimates. Results may vary under scanner, institution, acquisition, or population shift.
