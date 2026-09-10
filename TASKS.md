# BrainSeg Task Plan

Project: BrainSeg / BRATS  
Source of truth: `BrainSeg-PRD.md`  
Task IDs: `BRATS-XXX`

## Current Status

- Phase 0: `DONE`
- Phase 1: `DONE`
- Phase 2: `DONE` through `BRATS-027`
- Phase 3: `DONE` through `BRATS-034`
- Phase 4: `DONE` through `BRATS-043`; canonical model is already trained and registered
- Current next task: `BRATS-044` - define PostgreSQL orchestration state for the completed primary model pipeline

Inference CLI: `scripts/infer_subject.py` accepts a four-modality subject folder and writes a full-size segmentation NIfTI, overlay PNG, and JSON inference summary.

## Primary Model Status

- Canonical model: high-memory Attention U-Net
- Checkpoint: `artifacts/full-training-high-memory/best-model.pt`
- Model metadata: `artifacts/primary-model.json`
- External reference benchmark: removed from scope and not a blocker for the primary project

Dataset visualization command: `scripts/visualize_dataset.py` generates charts and labeled sample montages under `artifacts/dataset-visualization`.

Secondary model: baseline 3D U-Net configuration is tracked in `artifacts/baseline-model.json` and must use a separate output directory from the canonical Attention U-Net.

## Task Conventions

- A task is complete only when its proof-of-work artifact is produced and linked in the task record or final report.
- Each task should be independently testable where practical.
- Data and model results must include enough metadata to reproduce the run.
- No task may claim clinical validity; this is a research and portfolio project.
- Commit messages should use `feat(BRATS-XXX): ...`, `fix(BRATS-XXX): ...`, or `test(BRATS-XXX): ...`.

## Status Legend

- `TODO`: not started
- `IN PROGRESS`: actively being implemented
- `BLOCKED`: cannot proceed until a dependency or decision is resolved
- `DONE`: implementation and proof-of-work are complete

## Phase 0: Environment and Data Acquisition

| ID | Status | Task | Depends on | Proof of work |
|---|---|---|---|---|
| BRATS-001 | DONE | Create the repository structure for ML code, pipeline stages, API, infrastructure, tests, and documentation. | None | `README.md`, `docs/architecture.md`, and the initial backend/application, pipeline, infrastructure, scripts, and tests tree |
| BRATS-002 | DONE | Define Python dependencies for PyTorch, MONAI, NIfTI I/O, Optuna, Celery, PostgreSQL, MLflow, FastAPI, and testing. | BRATS-001 | `pyproject.toml`, `requirements.txt`, `requirements-gpu.md`, and validated dependency metadata |
| BRATS-003 | DONE | Verify the local RTX 5070, CUDA 12.8 compatibility, PyTorch GPU availability, and device name. | BRATS-002 | `scripts/verify_gpu.py` and `artifacts/hardware/gpu-verification.json`; RTX 5070, PyTorch `2.14.0+cu132`, CUDA available, capability `12.0` |
| BRATS-004 | DONE | Add environment configuration, `.env.example`, secret handling, and ignored local credentials. | BRATS-001 | `.env.example`, `.gitignore`, `app/config/settings.py`, and verified Conda runtime configuration |
| BRATS-005 | DONE | Add Docker Compose services for Redis, PostgreSQL, MLflow, and FastAPI. | BRATS-001 | `docker-compose.yml`, service Dockerfiles, health endpoint, and validated Compose configuration |
| BRATS-006 | DONE | Implement Kaggle API dataset download with resumable/idempotent behavior. | BRATS-004 | `pipeline/stages/download.py`, `scripts/download_data.py`, live Kaggle progress passthrough, same-drive staging, `--force`, and dry-run/no-op behavior |
| BRATS-007 | DONE | Implement file manifest generation with paths, subject IDs, sizes, and checksums. | BRATS-006 | `pipeline/stages/manifest.py`, `scripts/build_manifest.py`, synthetic-fixture verification, and BraTS `*_Segm.nii` filename handling |
| BRATS-008 | DONE | Validate expected BraTS file counts, cohorts, modalities, metadata files, and total dataset structure. | BRATS-007 | `artifacts/dataset-validation.json`: exact BraTS counts passed, including 369 training subjects, 125 validation subjects, 1,845 training NIfTIs, 500 validation NIfTIs, 4 CSV files, and disjoint cohorts |
| BRATS-009 | DONE | Validate random NIfTI samples for readability, expected shape, affine, and spacing. | BRATS-007 | `artifacts/volume-verification.json`: 5 deterministic samples opened successfully with `240x240x155` shape and `1.0mm` isotropic spacing |
| BRATS-010 | DONE | Persist the raw data manifest and acquisition status in PostgreSQL. | BRATS-005, BRATS-007 | PostgreSQL record persisted with manifest ID `2670d6bf9cc8a62600831019704464d1415c1181a289c9ce1013c6d9c8019ffd` and 2,349 file records |

## Phase 1: Splitting and Preprocessing

| ID | Status | Task | Depends on | Proof of work |
|---|---|---|---|---|
| BRATS-011 | DONE | Implement deterministic subject-level train/validation/test splitting for the 369 labeled subjects. | BRATS-008 | `artifacts/split_manifest.json`: seed `42`, source manifest ID recorded, and counts of 259 train, 55 validation, and 55 test subjects |
| BRATS-012 | DONE | Add leakage checks proving that subjects cannot appear in multiple splits. | BRATS-011 | `artifacts/split-validation.json`: all seven checks passed, including full 369-subject coverage, pairwise disjointness, labeled-training membership, and deterministic regeneration |
| BRATS-013 | DONE | Implement multimodal NIfTI loading and BraTS label handling. | BRATS-009 | Real `BraTS20_Training_001` inspection passed: images `(4, 240, 240, 155)`, mask `(240, 240, 155)`, labels `{0,1,2,4}`, and `1.0mm` spacing; 8 unit tests passed |
| BRATS-014 | DONE | Implement intensity normalization and foreground cropping. | BRATS-013 | Real `BraTS20_Training_001` preprocessing passed: crop `240x240x155` to `153x189x147`, with recorded bounds/statistics; 5 preprocessing tests passed |
| BRATS-015 | DONE | Implement compressed preprocessing cache keyed by input manifest and preprocessing configuration hash. | BRATS-014 | Real `BraTS20_Training_001` cache produced a miss then a hit with identical `(4,153,189,147)` data; 4 cache/preprocessing tests passed |
| BRATS-016 | DONE | Implement lazy/cached subject loading with MONAI. | BRATS-015 | Real lazy dataset inspection exposed 259 train subjects and loaded one cache-backed sample as `(4,153,189,147)` tensors with labels `{0,1,2,4}`; unit test passed |
| BRATS-017 | DONE | Implement foreground-biased 3D patch sampling. | BRATS-016 | Real 100-patch report produced 68 foreground-centered and 32 random patches with default `128^3` patches; 4 patch tests passed |
| BRATS-018 | DONE | Persist dataset versions, preprocessing configuration, and lineage in PostgreSQL. | BRATS-010, BRATS-015 | Dataset version `c36885b173a5705fc5c17f1844e45400d5619e7a2a4a661b516671efca90eb10` persisted for manifest `2670d6bf9cc8a62600831019704464d1415c1181a289c9ce1013c6d9c8019ffd`; identical rerun returned `skipped` |

## Phase 2: Modeling and Sanity Gate

| ID | Status | Task | Depends on | Proof of work |
|---|---|---|---|---|
| BRATS-019 | DONE | Implement the baseline 3D U-Net with configurable channels and normalization. | BRATS-016 | Baseline architecture tested and fully trained separately for comparison; run `0f375dd6baf24196881d92f4b2f7b8a2`, checkpoint `artifacts/full-training-baseline/best-model.pt` |
| BRATS-020 | DONE | Implement the Attention U-Net candidate. | BRATS-019 | 2 attention tests passed; CUDA smoke test passed on `(1,4,64,64,64)` with 1,784,703 parameters and matching output shape |
| BRATS-021 | DONE | Implement BraTS region conversion for WT, TC, and ET. | BRATS-013 | 4 region tests passed; real subject counts reproduced: WT `211,979`, TC `43,185`, ET `27,742` |
| BRATS-022 | DONE | Implement Dice, HD95, and composite validation metrics. | BRATS-021 | 3 metric tests passed; real self-comparison returned Dice `1.0` and HD95 `0.0` for WT, TC, and ET |
| BRATS-023 | DONE | Implement combined Dice/Cross-Entropy or Focal loss with configurable weights. | BRATS-021 | 3 loss tests passed; CUDA loss check produced finite loss `1.21698` with gradients |
| BRATS-024 | DONE | Add AMP, gradient accumulation, optional checkpointing, and GPU memory logging. | BRATS-019 | 2 utility tests passed; CUDA AMP/accumulation micro-step produced finite loss `1.50619` and peak memory `6,009,344` bytes |
| BRATS-025 | DONE | Implement sliding-window inference with overlap and Gaussian blending. | BRATS-019 | 2 reconstruction tests passed; CUDA synthetic inference reconstructed `(1,4,80,88,96)` with matching output shape |
| BRATS-026 | DONE | Implement the automated overfit-single-batch sanity gate. | BRATS-017, BRATS-023 | `artifacts/sanity-gate.json`: CUDA gate passed at best epoch 455 with foreground Dice `0.97661` and loss reduced `1.19585` to `0.06890` on a deterministic `32^3` patch |
| BRATS-027 | DONE | Ensure a failed sanity gate blocks all model-selection work. | BRATS-026 | `pipeline/models/gate.py` and 3 failure/acceptance tests; failed or missing sanity artifacts raise before selection |

## Phase 3: Experiment Tracking and Model Selection

| ID | Status | Task | Depends on | Proof of work |
|---|---|---|---|---|
| BRATS-028 | DONE | Configure MLflow tracking and artifact storage. | BRATS-005 | MLflow `2.19.0` health passed; sample run `62da732a868c4f24a2e2ccaf0a9f8424` finished in experiment `1` with metric `health_check=1.0` and `diagnostics/health.json` artifact; UTF-8 console handling added for Windows callers |
| BRATS-029 | DONE | Define the Optuna search space for architecture, patch size, learning rate, and loss weights. | BRATS-019, BRATS-020, BRATS-023 | `pipeline/model_selection/search.py` and 3 search-space tests covering architecture, patch size, learning rate, Dice/CE weights, and variation |
| BRATS-030 | DONE | Enforce a fixed trial count or GPU-hour budget. | BRATS-029 | `SearchConfig` enforces positive trial count/epoch/time budgets; real sweep bounded to 2 trials and `600` seconds |
| BRATS-031 | DONE | Implement short-budget trial training and validation. | BRATS-026, BRATS-029 | Real two-trial short-budget training completed on cached BraTS data with one epoch per trial |
| BRATS-032 | DONE | Log every trial's configuration, metrics, resource use, and artifacts to MLflow. | BRATS-028, BRATS-031 | MLflow runs `01ba926bc14943e7a341ee2811abc751` and `66b147f38e0c407f87ef06f0925da117` completed in experiment `1` |
| BRATS-033 | DONE | Select and persist the winning trial using composite WT/TC/ET validation Dice. | BRATS-032 | `artifacts/model-selection.json` persisted trial 0 as winner with composite Dice `0.13975538810094199` |
| BRATS-034 | DONE | Add a test proving that trial configurations actually vary and are not silently reused. | BRATS-031 | Variation test passed; real trials differed in patch size, learning rate, and Dice weight |

## Phase 4: Full Training and Evaluation

| ID | Status | Task | Depends on | Proof of work |
|---|---|---|---|---|
| BRATS-035 | DONE | Implement full training from the selected configuration. | BRATS-033 | Full training accepts the canonical `primary-model.json` configuration when the historical Optuna artifact is unavailable; future runs store split previews and curves |
| BRATS-036 | DONE | Log per-epoch loss, Dice, HD95, wall-clock time, and peak GPU memory. | BRATS-035 | Training now records train loss, validation loss, duration, throughput, peak memory, `training-curves.png`, and MLflow plot/preview artifacts |
| BRATS-037 | DONE | Register the trained checkpoint in the MLflow model registry. | BRATS-035 | Model `brainseg-segmentation` version `1` registered successfully; registration run `8b8310ae8d744f89b1efe76ea1e8e215` |
| BRATS-038 | DONE | Implement full-volume held-out evaluation using sliding-window inference. | BRATS-022, BRATS-025, BRATS-035 | Evaluation uses checkpoint-adjacent training configuration when available, so the external Optuna selection artifact is not required for restored checkpoints |
| BRATS-039 | DONE | Report per-subject and aggregate WT/TC/ET Dice and HD95. | BRATS-038 | Full aggregate: WT Dice `0.83091`/HD95 `28.5352`, TC Dice `0.71299`/HD95 `21.1474`, ET Dice `0.67402`/HD95 `Infinity` due to empty-region cases |
| BRATS-040 | DONE | Generate representative and worst-case prediction overlays. | BRATS-038 | Generated 5 overlays for `BraTS20_Training_141`, `177`, `259`, `275`, and `280`; gallery manifest at `artifacts/failure-gallery/gallery.json` |
| BRATS-041 | DONE | Add MC dropout and/or test-time augmentation uncertainty estimation. | BRATS-038 | `artifacts/uncertainty-tta.json` completed with 7 flip-TTA passes and nonzero variance for subjects `275` and `310` |
| BRATS-042 | CANCELLED | Run and document an external reference benchmark. | BRATS-011 | Removed from scope; the canonical high-memory Attention U-Net is the sole project model |
| BRATS-043 | DONE | Add checks for leakage, suspiciously high Dice, NaN training, and unchanged checkpoints. | BRATS-038 | Corrected evaluation validated successfully; finite aggregate metrics passed with 2 explicitly recorded non-finite ET HD95 cases |

## Phase 5: Orchestration and State Tracking

| ID | Status | Task | Depends on | Proof of work |
|---|---|---|---|---|
| BRATS-044 | DONE | Define PostgreSQL schema for pipeline runs and stages. | BRATS-005 | PostgreSQL schema initialized successfully; orchestration model test passed |
| BRATS-045 | DONE | Define artifact, dataset, model, and lineage records in PostgreSQL. | BRATS-010, BRATS-018, BRATS-037 | Canonical lineage persisted as model ID `4e46e3058f7746e2d1c968dc16a0706fd798ded38740c9d822b378dbe4194db4`; identical rerun returned `skipped` |
| BRATS-046 | DONE | Wrap download, preprocessing, sanity, selection, training, and evaluation as Celery tasks. | BRATS-006, BRATS-015, BRATS-026, BRATS-031, BRATS-035, BRATS-038 | Celery task registration verified; focused test passed and `scripts/verify_celery.py` reports project tasks |
| BRATS-047 | DONE | Chain all stages with explicit upstream dependency and status checks. | BRATS-046 | `run_ordered_stage` enforces upstream completion; task registration and policy tests passed |
| BRATS-048 | DONE | Add retries with exponential backoff for transient failures. | BRATS-046 | `run_stage` retries `ConnectionError`/`TimeoutError` with exponential backoff and max 3 retries |
| BRATS-049 | DONE | Add hard failures for logic, data, sanity-gate, and resource incompatibility errors. | BRATS-047 | `PipelineLogicError` hard-failure path and upstream blocking are integrated |
| BRATS-050 | DONE | Implement content/config-hash idempotency for each stage. | BRATS-046 | `idempotent_stage` skips outputs with matching content hash and records skip reason |
| BRATS-051 | DONE | Implement a single CLI/API trigger for the complete pipeline. | BRATS-047, BRATS-050 | Worker-backed trigger verified after removing forbidden nested `result.get()`; ordered primary-model stages submit and record through PostgreSQL |

## Phase 6: Serving and Dashboard

| ID | Status | Task | Depends on | Proof of work |
|---|---|---|---|---|
| BRATS-052 | DONE | Implement FastAPI inference endpoint for four-modality input. | BRATS-037, BRATS-025 | `/health` and `/predict` implemented around canonical checkpoint; API/model contract tests passed |
| BRATS-053 | DONE | Load the currently promoted model from MLflow. | BRATS-052 | Real cached-subject inference loaded `brainseg-segmentation` version `1` from MLflow registry on CUDA |
| BRATS-054 | DONE | Return segmentation, optional uncertainty output, model version, and latency metadata. | BRATS-052 | Real inference returned shape `151x181x145`, model/version metadata, CUDA device, and latency `510.65ms` |
| BRATS-055 | DONE | Track inference latency and serving failures. | BRATS-054 | Real registry-backed CUDA inference persisted a completed PostgreSQL event with `503.93ms` latency |
| BRATS-056 | CANCELLED | Build presentation dashboard. | BRATS-044, BRATS-051 | Presentation layer removed from scope; backend analytics endpoints and CLI artifacts remain available |
| BRATS-057 | CANCELLED | Build frontend training/model comparison views. | BRATS-032, BRATS-036, BRATS-039 | Frontend removed from scope; MLflow and JSON artifacts remain available |
| BRATS-058 | CANCELLED | Build frontend slice viewer. | BRATS-052 | Frontend removed from scope; `scripts/infer_subject.py` produces prediction overlays |
| BRATS-059 | DONE | Add uncertainty, citations, limitations, and research-only disclaimer. | BRATS-041 | TTA uncertainty artifact, README citations, CLI outputs, and backend `/about` response retained |

## Phase 7: Release Validation

| ID | Status | Task | Depends on | Proof of work |
|---|---|---|---|---|
| BRATS-060 | DONE | Add unit, integration, data, pipeline, and API test suites. | All implementation phases | Full project suite passed: 66 tests, 0 failures; GitHub Actions CI added for tests, lint, API type checks, release contracts, and Compose builds |
| BRATS-061 | DONE | Validate clean-environment Docker Compose startup. | BRATS-005 | Compose services healthy; CPU API dependencies, artifact/cache mounts, CORS, `/analytics/model` HTTP 200, and `/viewer/subjects` HTTP 200 verified |
| BRATS-062 | DONE | Validate complete pipeline execution from a single trigger. | BRATS-051 | Run `962b85524c6a4ee384ff8fdda21a52c5` submitted 3 ordered stages; all completed in PostgreSQL |
| BRATS-063 | DONE | Validate idempotent rerun behavior and changed-input invalidation. | BRATS-050 | Probe verified first execution accepted, identical payload skipped by content hash, and changed payload re-executed with a new hash |
| BRATS-064 | DONE | Complete README with setup, citations, architecture, usage, results, and limitations. | BRATS-059 | Detailed README includes Mermaid architecture/data-flow diagrams, metrics, artifact image/chart references, commands, serving, orchestration, citations, and limitations |
| BRATS-065 | DONE | Capture final portfolio demonstration and link every phase gate artifact. | BRATS-062, BRATS-064 | `docs/DEMO_RUNBOOK.md` and `artifacts/release-index.json` provide the reproducible backend demo sequence and complete automated artifact index |

## Initial Execution Order

The first implementation sequence should be:

1. `BRATS-001` through `BRATS-005`: establish the environment and local services.
2. `BRATS-006` through `BRATS-010`: make data acquisition verifiable and repeatable.
3. `BRATS-011` through `BRATS-018`: create safe splits and the preprocessing cache.
4. `BRATS-019` through `BRATS-027`: prove the modeling path works before adding automation.
5. Continue through the phases only after each phase's gate artifact exists.
