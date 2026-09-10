# BrainSeg Architecture Decisions

Project: BrainSeg / BRATS  
Source: `BrainSeg-PRD.md`  
Status: Initial decisions for implementation

## Current Status

- Phase 0: complete. Dataset acquisition, validation, GPU verification, manifest generation, and PostgreSQL persistence are verified.
- Phase 1: complete. Subject splits, preprocessing, caching, lazy loading, patch sampling, and dataset lineage are verified.
- Phase 2: complete through `BRATS-027`. U-Net, Attention U-Net, metrics, loss, inference, resource utilities, sanity gate, and gate enforcement are verified.
- Primary model is the verified high-memory Attention U-Net documented in `artifacts/primary-model.json`; next decision boundary is `BRATS-044`, orchestration state for the completed pipeline.

No additional model training is required for the current deliverable. Subsequent phases productize, serve, monitor, and demonstrate the existing registered model. PostgreSQL orchestration, canonical model lineage, and worker-backed ordered triggering have now been verified.

The current implementation is research/portfolio infrastructure only. No clinical or diagnostic claims are supported.

## How To Use This File

This document records decisions that affect architecture, reproducibility, scope, or operational behavior. A decision may be revisited when new evidence appears, but changes should update the original entry rather than silently replacing it.

## Decision Summary

| ID | Decision | Status |
|---|---|---|
| ADR-001 | Use BraTS 2020 from the Kaggle-hosted dataset as the initial data source. | Accepted |
| ADR-002 | Use subject-level splits from the 369 labeled training subjects. | Accepted |
| ADR-003 | Use patch-based 3D training and sliding-window full-volume inference. | Accepted |
| ADR-004 | Use PyTorch and MONAI for the modeling/data layer. | Accepted |
| ADR-005 | Compare 3D U-Net and Attention U-Net during automated model selection. | Accepted |
| ADR-006 | Use Optuna for bounded automated model selection. | Accepted |
| ADR-007 | Use Celery and Redis for initial orchestration. | Accepted |
| ADR-008 | Separate pipeline/lineage state in PostgreSQL from modeling experiments in MLflow. | Accepted |
| ADR-009 | Use Docker Compose as the local deployment boundary. | Accepted |
| ADR-010 | Serve the promoted model through FastAPI and expose results through MLflow, PostgreSQL, and CLI artifacts. | Accepted |
| ADR-011 | Treat uncertainty estimation as a required evaluation output. | Accepted |
| ADR-012 | Keep survival prediction, recurrence classification, distributed training, and clinical deployment out of scope. | Accepted |
| ADR-013 | Organize the repository by runtime boundary and technical responsibility. | Accepted |
| ADR-014 | Use `pyproject.toml` for Python metadata and dependencies, while selecting the PyTorch CUDA wheel during environment verification. | Accepted |
| ADR-015 | Do not mark GPU readiness from driver detection alone; require a passing PyTorch CUDA verification artifact. | Accepted |
| ADR-016 | Use typed environment-backed settings and keep all credentials and generated data outside version control. | Accepted |
| ADR-017 | Use Docker Compose as the local backend service boundary for PostgreSQL, Redis, MLflow, and FastAPI. | Accepted |
| ADR-018 | Require the Kaggle dataset slug from configuration, expose Kaggle's native progress output, and promote downloads through a temporary directory. | Accepted |
| ADR-019 | Generate deterministic streaming SHA-256 manifests and exclude downloader control markers from dataset contents. | Accepted |
| ADR-020 | Make exact BraTS composition validation the production gate, with relaxed expectations only for fixtures/development. | Accepted |
| ADR-021 | Keep NIfTI geometry verification as a separate sampled gate after manifest validation. | Accepted |
| ADR-022 | Persist content-addressed dataset manifests with one manifest record and child file records in PostgreSQL. | Accepted |
| ADR-023 | Map Docker PostgreSQL to host port 5433 because native Windows PostgreSQL owns host port 5432. | Accepted |
| ADR-024 | Build splits only from labeled training subjects using a deterministic 70/15/15 subject-level partition. | Accepted |
| ADR-025 | Validate split coverage, disjointness, training-cohort membership, and deterministic regeneration as separate checks. | Accepted |
| ADR-026 | Load one subject lazily as four modality channels plus a validated label volume, preserving source affine and spacing. | Accepted |
| ADR-027 | Normalize each modality over nonzero foreground and crop the aligned image/mask to a shared foreground bounding box with configurable margin. | Accepted |
| ADR-028 | Cache each preprocessed subject as compressed NPZ plus metadata, keyed by source manifest content and preprocessing configuration. | Accepted |
| ADR-029 | Expose a lazy PyTorch dataset over cached subjects and provide MONAI PersistentDataset as an optional transform-cache adapter. | Accepted |
| ADR-030 | Sample approximately two-thirds tumor-centered 3D patches and one-third random patches, with explicit sample provenance. | Accepted |
| ADR-031 | Persist preprocessing dataset versions by source manifest ID plus canonical configuration hash. | Accepted |
| ADR-032 | Use a four-class configurable 3D U-Net baseline with instance normalization, leaky ReLU, and skip connections. | Accepted |
| ADR-033 | Implement Attention U-Net as a structurally comparable candidate with additive attention gates on skip features. | Accepted |
| ADR-034 | Use explicit WT/TC/ET composite masks for evaluation, with network class channel 3 mapped to BraTS label 4. | Accepted |
| ADR-035 | Report Dice and HD95 independently for WT, TC, and ET, using physical voxel spacing for HD95. | Accepted |
| ADR-036 | Use a configurable normalized combination of foreground soft Dice loss and four-class cross-entropy. | Accepted |
| ADR-037 | Centralize AMP, gradient accumulation, atomic checkpointing, and peak GPU memory measurement in reusable training utilities. | Accepted |
| ADR-038 | Use overlap-based Gaussian-weighted sliding-window inference for full-volume predictions. | Accepted |
| ADR-039 | Require an automated fixed-patch overfit gate before model selection, using loss reduction and foreground Dice thresholds. | Accepted |
| ADR-040 | Enforce the sanity result artifact at model-selection entry and hard-fail on missing, malformed, or failed status. | Accepted |
| ADR-041 | Use MLflow at `localhost:5000` with PostgreSQL backend state and a persistent Docker artifact volume as the experiment source of truth. | Accepted |
| ADR-042 | Use Optuna for a bounded search over architecture, patch size, learning rate, and Dice/Cross-Entropy weighting. | Accepted |
| ADR-043 | Treat every trial as a first-class MLflow run and persist the winning trial/configuration in a JSON selection artifact. | Accepted |
| ADR-044 | Use the selected Phase 3 configuration for checkpointed full training with per-epoch MLflow/resource logging. | Accepted |
| ADR-045 | Evaluate held-out subjects with full-volume sliding-window inference and report per-region Dice/HD95 before model registration. | Accepted |
| ADR-046 | Use nested tqdm progress tracking and MLflow system-metrics logging during full training by default. | Accepted |
| ADR-047 | Prefetch fixed-size cached patches with DataLoader workers/pinned memory, enable cuDNN benchmarking, and expose torch.compile as an opt-in comparison. | Accepted |
| ADR-048 | Generate worst-case prediction overlays and TTA uncertainty summaries as Phase 4 evaluation outputs. | Accepted |
| ADR-049 | Use the verified high-memory Attention U-Net as the sole canonical BrainSeg model. | Accepted |
| ADR-050 | Track pipeline runs, stages, artifacts, resource usage, and MLflow links in PostgreSQL. | Accepted |
| ADR-051 | Use Celery with Redis as the asynchronous task boundary while PostgreSQL remains the orchestration state source of truth. | Accepted |
| ADR-052 | Require explicit upstream stage completion before an ordered Celery stage can execute. | Accepted |
| ADR-053 | Classify transient failures for retry, logic/data failures as hard errors, and stage outputs by content hash for idempotency. | Accepted |
| ADR-054 | Use one CLI trigger to submit ordered post-training primary-model verification/publication stages without retraining. | Accepted |
| ADR-055 | Retry only transient connection/timeout failures, hard-fail logic errors, and skip unchanged stage outputs by content hash. | Accepted |
| ADR-056 | Serve the existing canonical Attention U-Net through a lazy-loaded FastAPI inference service with a four-channel tensor contract. | Accepted |
| ADR-057 | Record inference completion/failure, latency, device, and served model version in PostgreSQL. | Accepted |
| ADR-058 | Prefer the MLflow registered primary model at serving time and fall back to the verified local checkpoint when the registry is unavailable. | Accepted |
| ADR-059 | Keep backend analytics read-only through FastAPI endpoints and PostgreSQL/MLflow artifacts. | Accepted |
| ADR-060 | Normalize model curves, evaluation regions, and MLflow trial comparisons through backend APIs and persisted artifacts. | Accepted |
| ADR-069 | Remove frontend presentation from scope; keep training/analytics controls in backend APIs and CLI workflows. | Accepted |
| ADR-070 | Keep the backend/API and CLI workflows as the sole active product surface. | Accepted |
| ADR-071 | Present training workspace evidence as charts, diagrams, split image samples, MLflow/resource views, and validated editable future-run parameters rather than text-only cards. | Accepted |
| ADR-072 | Generate dataset visualization artifacts deterministically from the verified manifest and subject split rather than embedding raw data in an application. | Accepted |
| ADR-073 | Require all four co-registered MRI modalities for local CLI inference and write segmentation, overlay, and latency summary artifacts. | Accepted |
| ADR-074 | Keep BrainSeg backend-only: remove the frontend dashboard and use FastAPI, MLflow, PostgreSQL, Celery, and CLI artifacts as the supported interfaces. | Accepted |
| ADR-076 | Maintain one complete chronological command runbook at `docs/COMMANDS.md` with a root `RUN_COMMANDS.md` compatibility entry. | Accepted |
| ADR-077 | Use README Mermaid diagrams and generated artifact references as the primary project architecture/results handoff. | Accepted |
| ADR-078 | Use GitHub Actions for backend quality/release/Compose CI while keeping dataset download and GPU training manual. | Accepted |
| ADR-079 | Keep CI CPU/source focused and exclude 42.8 GB data download, GPU training, and generated checkpoint validation from pull-request jobs. | Accepted |
| ADR-075 | Train the baseline 3D U-Net as a separate comparison model with isolated artifacts; never overwrite the canonical Attention U-Net checkpoint. | Accepted |
| ADR-061 | Use a read-only subject-slice API with generated output artifacts, avoiding image-storage duplication. | Accepted |
| ADR-062 | Use the held-out split for backend/CLI inference and generated prediction artifacts. | Accepted |
| ADR-063 | Present TTA entropy summaries and explicit research-only/citation content in backend artifacts and documentation; never frame uncertainty as clinical risk. | Accepted |
| ADR-064 | Treat the full repository test suite as the release gate; generated model binaries are optional external artifacts, not test fixtures. | Accepted |
| ADR-065 | Mount generated model artifacts into the API container read-only through `BRAINSEG_ARTIFACT_ROOT`; source checkouts remain free of model binaries. | Accepted |
| ADR-066 | Keep API container dependencies separate from GPU training dependencies to make clean Compose startup reproducible and lightweight. | Accepted |
| ADR-067 | Make the release README and artifact index the final portfolio handoff surface for the completed canonical model. | Accepted |
| ADR-068 | Use a reproducible demo runbook and release index instead of making a screen recording a technical release blocker. | Accepted |

## ADR-001: Kaggle-Hosted BraTS 2020 Dataset

### Context

The project needs an unattended and reproducible data acquisition path. The working dataset is the Kaggle-hosted copy of BraTS 2020, containing 369 labeled training subjects and 125 unlabeled validation subjects.

### Decision

Use the Kaggle API as the initial automated download source. Credentials must come from environment variables or a local ignored `.env` file. Every download must produce a checksum-verified manifest and validate the expected file composition.

### Consequences

- The pipeline can be run without a manual browser download.
- Dataset corruption and partial downloads are detected before preprocessing.
- The raw working footprint is approximately 42.8 GB and must be planned for.
- The project must include the required BraTS citations in public documentation.

## ADR-002: Subject-Level Dataset Splits

### Context

The official validation and test cohorts do not provide public labels for local scoring. Slice- or patch-level splitting would cause leakage and invalidate metrics.

### Decision

Create deterministic train, validation, and test splits only from the 369 labeled training subjects, approximately 70/15/15. Store the random seed and assignments in the manifest/database and verify no subject appears in more than one split.

### Consequences

- Local metrics are based on labeled data that can actually be evaluated.
- The split is reproducible and auditable.
- The official unlabeled cohorts cannot be used as local scored test data.

## ADR-003: Patch-Based Training With Sliding-Window Inference

### Context

Native volumes are approximately `240x240x155`, and full-volume 3D training is unlikely to fit alongside a useful network on the target 12 GB GPU.

### Decision

Train on random 3D patches, initially targeting `128x128x128` and falling back to a smaller documented size if required. Use foreground-biased sampling, AMP, gradient accumulation, and optional gradient checkpointing. Reconstruct full-volume predictions through overlapping sliding-window inference with Gaussian blending.

### Consequences

- Training fits within constrained GPU memory.
- Full-volume evaluation remains possible.
- Metrics must always be calculated after volume reconstruction, never from isolated patches.
- Patch-size changes are configuration changes that must be tracked and not hidden.

## ADR-004: PyTorch and MONAI

### Context

The project needs flexible 3D model code, medical-imaging transforms, patch sampling, caching, and sliding-window utilities.

### Decision

Use PyTorch as the model/training foundation and MONAI for medical-imaging data utilities and transforms. Use nibabel or SimpleITK for NIfTI I/O where MONAI abstractions are insufficient.

### Consequences

- The stack supports custom architectures and medical-imaging workflows.
- MONAI reduces the amount of custom infrastructure for common volumetric operations.
- CUDA/PyTorch compatibility must be verified early on the RTX 5070.

## ADR-005: Candidate Models and Reference Benchmark

### Context

The portfolio needs both a controlled custom comparison and a credible external reference point.

### Decision

Use a baseline 3D U-Net and Attention U-Net as the automated candidate architectures.

### Consequences

- The automated search remains interpretable and bounded.
- The project can explain why a selected custom model was chosen.
- Candidate comparison uses the same split, preprocessing, metrics, and tracking workflow.

## ADR-006: Bounded Optuna Model Selection

### Context

Manual model selection weakens the automation story, while an unbounded sweep can exhaust local or hosted GPU resources.

### Decision

Use Optuna to search architecture and selected hyperparameters, with a fixed trial count or explicit GPU-hour ceiling. Short-budget trials select the configuration; the winner is automatically promoted to full training. Every trial, including losing trials, is logged.

### Consequences

- Model selection is reproducible and inspectable.
- Compute usage is controlled.
- Trial configurations must be validated to ensure the search actually varies parameters.

## ADR-007: Celery and Redis Orchestration

### Context

The pipeline needs background execution, retries, status transitions, and one-trigger chaining. The project already favors Celery/Redis and the initial DAG is mostly linear.

### Decision

Use Celery with Redis for orchestration. Tasks must explicitly verify upstream success, distinguish transient errors from logic errors, and record their state and outputs.

### Alternatives Considered

Prefect or Airflow would provide more specialized DAG management. They are deferred because the initial workflow is a simple single-machine chain and Celery aligns with the existing project direction.

### Consequences

- The initial stack is relatively lightweight for local deployment.
- More complex branching or scheduling may justify migration to Prefect later.
- Task correctness, failure propagation, and idempotency must be tested explicitly rather than assumed from task ordering.

## ADR-008: PostgreSQL and MLflow Have Separate Responsibilities

### Context

Pipeline health/lineage and model experiment tracking are related but different concerns. Combining them into log files would make runs difficult to query and audit.

### Decision

Use PostgreSQL as the source of truth for pipeline runs, stage state, dataset lineage, artifacts, and resource records. Use MLflow as the source of truth for model hyperparameters, metrics, artifacts, and model registry state.

### Consequences

- Backend analytics can reason about pipeline reliability separately from model quality.
- A model can be traced back to its dataset and preprocessing configuration.
- Integration code must preserve IDs and links between PostgreSQL and MLflow records.

## ADR-009: Docker Compose for the Local Stack

### Context

The project includes Redis, PostgreSQL, MLflow, and FastAPI. Requiring each service to be installed manually makes the backend demo difficult to reproduce.

### Decision

Use Docker Compose as the standard local startup boundary. The GPU training process must remain compatible with the host GPU while the supporting services run in containers.

### Consequences

- The service stack can be started consistently with one command.
- GPU/container compatibility must be documented separately from CPU-only service startup.
- Persistent volumes are required for PostgreSQL, MLflow artifacts, and Redis as appropriate.

## ADR-010: FastAPI Backend Serving

### Context

The project needs a stable inference and orchestration API.

### Decision

Use FastAPI for inference, analytics, and pipeline API endpoints. The API must load the currently promoted model from MLflow rather than a manually copied checkpoint.

### Consequences

- Serving follows the model registry promotion path.
- CLI and API clients can query live pipeline and experiment data.
- Hardcoded metrics and static demo values are not acceptable as final behavior.

## ADR-011: Uncertainty Is a Required Evaluation Output

### Context

The smallest enhancing-tumor region is difficult to segment, and the project should surface model limitations rather than only showing aggregate Dice. BraTS includes uncertainty quantification as an official task.

### Decision

Generate uncertainty maps through MC dropout, test-time augmentation, or a documented equivalent as part of the automated evaluation stage. Store uncertainty overlays and failure examples as artifacts.

### Consequences

- Explainability is part of the pipeline rather than a manual post-processing step.
- Additional inference cost must be measured and tracked.
- Uncertainty outputs must be described as model uncertainty, not clinical certainty or diagnosis.

## ADR-012: Explicit Scope Boundaries

### Context

The dataset contains clinical metadata and the challenge includes additional tasks, but expanding into multiple clinical problems would dilute the segmentation/MLOps objective.

### Decision

Keep the initial project focused on segmentation and uncertainty quantification. Do not implement survival prediction, recurrence classification, distributed/multi-node training, multi-tenant dashboard authentication, or clinical deployment.

### Consequences

- The implementation remains achievable and focused.
- The README must state that excluded tasks are deliberate scope choices, not missing-data problems.
- Any future external-data experiment must include a BraTS-only baseline and report both results.

## ADR-013: Repository Boundaries

### Context

The project combines data engineering, model training, orchestration, APIs, and experiment tracking. A single flat source directory would make those responsibilities difficult to test and evolve independently.

### Decision

Organize the repository into explicit boundaries: `pipeline` for data/model/inference logic, `app` for API/configuration/database/tracking integration, `infrastructure` for service definitions, `scripts` for operational tooling, `tests` for test categories, and `docs` for project documentation.

### Consequences

- Later implementation tasks have clear target locations.
- Model logic can be tested independently from Celery, PostgreSQL, and HTTP integration.
- Supporting services can evolve without mixing API code into the training package.
- The structure is intentionally lightweight at this stage; concrete framework modules will be added only when their corresponding tasks begin.

## ADR-014: Python Dependency and CUDA Management

### Context

The project combines machine-learning, orchestration, API, database, tracking, and development dependencies. PyTorch must also match the target NVIDIA driver and CUDA support, especially for the RTX 5070/Blackwell target.

### Decision

Use `pyproject.toml` as the canonical Python project and dependency definition. Keep the PyTorch package in the declared runtime dependencies, but do not hardcode a CUDA-specific package index or wheel until the hardware verification task confirms the supported installation. Document that selection process in `requirements-gpu.md`.

### Consequences

- Dependency metadata is centralized and can be consumed by modern Python tooling.
- A generated-install-friendly `requirements.txt` is also maintained for the recommended Conda environment workflow.
- Development dependencies are isolated from runtime dependencies.
- GPU installation requires an explicit verification step rather than silently accepting a CPU fallback.
- The exact resolved environment will be recorded after `BRATS-003`.

## ADR-015: GPU Readiness Gate

### Context

The host has an RTX 5070 and a current NVIDIA driver, but the active Python environment does not currently contain PyTorch. Driver-level detection cannot prove that the training framework can access CUDA or use the target device.

### Decision

GPU readiness requires all of the following: a detected NVIDIA GPU, an installed PyTorch build, `torch.cuda.is_available()` returning true, and a recorded device name/capability in `artifacts/hardware/gpu-verification.json`. Until those checks pass, `BRATS-003` remains blocked.

### Consequences

- The project will not silently fall back to CPU training.
- The PyTorch/CUDA installation must be resolved and verified before model implementation is treated as runnable.
- The verified environment uses Python 3.12.14, PyTorch `2.14.0+cu132`, CUDA `13.2`, and RTX 5070 capability `12.0`.

## ADR-016: Environment Configuration and Secret Handling

### Context

The pipeline needs dataset credentials, service URLs, filesystem roots, and runtime options. Credentials must be available locally without being embedded in source code or committed to the repository.

### Decision

Use `pydantic-settings` for typed environment-backed configuration with an optional local `.env` file. Commit `.env.example` with names and safe defaults, ignore `.env` and credential files, and validate Kaggle settings only when the download stage requests them.

### Consequences

- Application code reads one consistent typed settings object.
- Importing the application remains possible without Kaggle credentials.
- Data acquisition fails with a clear configuration error when credentials are missing.
- Local datasets, caches, artifacts, and service state are explicitly excluded from version control.
- Runtime settings behavior will be verified after the declared Python environment is installed; the current host interpreter has no project dependencies yet.

## ADR-017: Local Service Boundary

### Context

The project depends on PostgreSQL, Redis, MLflow, and FastAPI. These services should be reproducible locally before the full backend pipeline is used.

### Decision

Use Docker Compose for the local backend service stack. PostgreSQL, Redis, and MLflow are configured with persistent volumes and health checks. FastAPI provides inference and analytics endpoints.

### Consequences

- The service topology and ports are stable from the beginning.
- Supporting services can be brought up independently of model training.
- API health is not evidence that inference or analytics are complete.
- Persistent local service state is excluded from version control.

## ADR-018: Safe Dataset Acquisition

### Context

The PRD identifies the Kaggle-hosted BraTS copy but does not specify a dataset slug. Downloading into the final data directory can also leave partial files after a network or CLI failure.

### Decision

Require `KAGGLE_DATASET` to be supplied through configuration rather than guessing a slug. Run the Kaggle download into a temporary directory, promote its contents only after a successful subprocess, and write a dataset-specific completion marker for idempotent reruns. Checksum and expected-composition validation remain the responsibility of the manifest stage.

### Consequences

- A missing or incorrect dataset slug fails explicitly instead of downloading an unintended dataset.
- Failed downloads do not receive a completion marker and do not masquerade as valid raw data.
- The downloader can be tested without downloading the full dataset.
- A successful download is not considered fully usable until `BRATS-007` and `BRATS-008` validate it.
- Kaggle's subprocess stdout/stderr is inherited by the terminal so the user can monitor download progress without a second progress implementation that could disagree with Kaggle's transfer state.
- The project uses Kaggle's supported `datasets download -d <owner/dataset>` command form.
- Staging is created beside the configured output directory so the compressed archive and uncompressed extraction use the same selected drive rather than the Windows system temp drive.

## ADR-019: Dataset Manifest Format

### Context

The download stage needs a durable inventory for integrity checks, subject-level splitting, and downstream lineage. Loading full volumes is unnecessary for this inventory step and would waste memory.

### Decision

Generate a JSON manifest by recursively scanning supported `.nii`, `.nii.gz`, and `.csv` files in deterministic path order. Hash files with streaming SHA-256, record relative paths, sizes, subject IDs, modality/type, aggregate counts, and total bytes. Exclude `.download-complete.json` because it is pipeline control state rather than dataset content.

### Consequences

- The manifest can detect changed, missing, or unexpected files before preprocessing.
- Hashing is bounded by the configured chunk size rather than file size.
- Subject and modality metadata can be reused by split and validation stages.
- NIfTI semantic validation remains a separate task; a checksum alone does not prove that a volume is readable.
- The manifest parser recognizes both standard `<subject>_seg.nii` files and the BraTS `*_Segm.nii` naming variant observed in the downloaded corpus.

## ADR-020: Dataset Composition Gate

### Context

The BraTS copy has a known composition, and a download that completes with missing, extra, or misclassified files must not proceed silently into preprocessing. Small synthetic fixtures are still needed for fast development tests.

### Decision

Validate exact production expectations by default: 369 training subjects, 125 validation subjects, 1,845 training NIfTIs, 500 validation NIfTIs, four CSV files, and disjoint cohorts. Expose relaxed expectations only through explicit test/development configuration; never use relaxed validation for a real pipeline run.

### Consequences

- Dataset acquisition fails early when the downloaded corpus is not the expected BraTS copy.
- Cohort classification and subject counts are auditable in a validation artifact.
- Unit tests can use small fixtures without weakening the production gate.
- Volume readability, shape, affine, and spacing remain separate validation concerns for `BRATS-009`.
- The downloaded corpus passed the strict production composition gate with all expected counts and disjoint cohorts.

## ADR-021: Sampled NIfTI Geometry Verification

### Context

Manifest checks establish file inventory and integrity, but they do not prove that NIfTI files can be decoded or that their geometry is usable for medical-imaging transforms. Opening every 42.8 GB volume during every validation run would be unnecessarily expensive.

### Decision

Add a deterministic, seed-controlled sampled verification stage using nibabel. For each sampled NIfTI, verify readability, positive 3D shape, positive three-axis voxel spacing, and record the affine matrix. Persist the selected paths and geometry in an artifact.

### Consequences

- Common corruption and geometry problems are detected before preprocessing.
- Verification cost is bounded by the sample count.
- The sample seed and artifact make the check reproducible.
- Full-corpus semantic validation is not claimed by this sampled gate.
- The downloaded corpus passed the sampled NIfTI gate: all five samples opened with `240x240x155` shape and `1.0mm` isotropic spacing. The observed negative X/Y affine orientation is consistent with the BraTS files and is retained rather than rewritten at acquisition time.

## ADR-022: Dataset Manifest Persistence

### Context

The raw manifest must be queryable after acquisition and linked to every file checksum. Re-running persistence should not create duplicate records for the same dataset contents.

### Decision

Use PostgreSQL tables `dataset_manifests` and `dataset_manifest_files`. The parent manifest ID is a SHA-256 hash of stable content fields and file records, excluding generation timestamp and local root path. Persist the full manifest JSON plus normalized file metadata, and skip an already-known manifest ID.

### Consequences

- Dataset lineage is queryable without reading the filesystem.
- Identical content is idempotent across regenerated manifests and machines.
- File-level checksums and modality metadata are available for later stages.
- PostgreSQL must be running before `BRATS-010` can be verified.
- Byte counts use PostgreSQL `BIGINT` because the raw corpus exceeds 32-bit integer capacity.
- The verified manifest was persisted with ID `2670d6bf9cc8a62600831019704464d1415c1181a289c9ce1013c6d9c8019ffd` and 2,349 child file records.

## ADR-023: Docker PostgreSQL Host Port

### Context

The host already has a native Windows PostgreSQL process listening on port `5432`. Docker cannot reliably expose its PostgreSQL container on the same host port, and host-side persistence commands were reaching the native service instead of the container.

### Decision

Expose the Docker PostgreSQL service on host port `5433`, while retaining container port `5432`. Host clients use `127.0.0.1:5433`; services inside the Compose network continue to use `postgres:5432`.

### Consequences

- The native Windows PostgreSQL installation remains undisturbed.
- Docker PostgreSQL is reachable unambiguously from host tools.
- `.env` should use `postgresql+psycopg://brainseg:brainseg@127.0.0.1:5433/brainseg` for host-side scripts.
- Container-to-container URLs must continue using the service hostname and port `5432`.

## ADR-024: Subject-Level Split Strategy

### Context

Only the 369 training subjects have local segmentation labels. The 125 official validation subjects must not enter the project's scored train/validation/test split, and slice- or patch-level splitting would leak subject-specific information.

### Decision

Derive the split universe from training subjects with a segmentation mask, sort IDs before shuffling with a recorded seed, and assign approximately 70%/15%/15% to train/validation/test. With the implemented round-to-nearest allocation, 369 subjects produce 259 train, 55 validation, and 55 test subjects.

### Consequences

- Every local evaluation subject has a ground-truth mask.
- Split membership is reproducible from the manifest and seed.
- No subject can appear in multiple splits.
- The unlabeled official validation cohort remains available for future inference-only use but is excluded from local scoring.

## ADR-025: Split Leakage Validation

### Context

A persisted split file can be edited or corrupted after generation. Checking only the generation code is insufficient to prove that the artifact used downstream has no overlap or missing subjects.

### Decision

Validate the saved split artifact independently against the source manifest. Check that every ID is a labeled training subject, all 369 labeled subjects are covered exactly once, each pair of splits is disjoint, recorded counts match actual lists, and regenerating with the recorded seed produces the same assignments.

### Consequences

- Split artifacts are independently auditable before preprocessing.
- Accidental subject leakage fails early.
- Changes to the seed or fractions produce an explicit artifact difference rather than an implicit behavior change.
- The real seed-42 split passed all seven validation checks with 259 train, 55 validation, and 55 test subjects.

## ADR-026: Multimodal Subject Loading

### Context

Training requires aligned T1, T1ce, T2, and FLAIR volumes plus the segmentation mask, but materializing all subjects at once would exceed practical memory and undermine patch-based training.

### Decision

Index subjects from the verified manifest and load one subject on demand. Stack the four modalities in fixed channel order `(T1, T1ce, T2, FLAIR)`, validate matching shape/affine between modalities and mask, preserve source affine/spacing, and accept only BraTS labels `0, 1, 2, 4`. Expose composite Whole Tumor, Tumor Core, and Enhancing Tumor masks using the PRD definitions.

### Consequences

- Data loading is lazy at subject granularity and ready for patch sampling.
- Geometry mismatches fail before training rather than producing invalid labels.
- Label `3` is rejected because it is not part of the BraTS 2020 annotation scheme.
- The source negative X/Y affine orientation is preserved until a documented preprocessing transform handles it.
- Real subject verification passed on `BraTS20_Training_001` with channel order `(T1, T1ce, T2, FLAIR)`, expected shapes, labels `{0,1,2,4}`, and `1.0mm` spacing.

## ADR-027: Intensity Normalization and Foreground Crop

### Context

Native BraTS volumes contain large zero-valued background regions and modality-specific intensity scales. Training on the full native field wastes memory and makes optimization less stable.

### Decision

Build a combined nonzero foreground mask across the four modalities. For each modality, compute mean and standard deviation over that mask, z-score the modality, set background back to zero, and crop images and labels to the shared foreground bounding box plus a configurable voxel margin. Return crop bounds and statistics for reproducibility and later reconstruction.

### Consequences

- Background storage and computation are reduced before patch sampling.
- Modality normalization is robust to different intensity scales.
- Image/mask alignment is preserved because one shared crop is applied.
- Crop bounds and normalization statistics must be stored with cached subjects.
- Real `BraTS20_Training_001` verification produced crop shape `153x189x147`, bounds `(41,34,0)` to `(194,223,147)`, and valid aligned labels.

## ADR-028: Preprocessing Cache

### Context

Repeated training and model-selection runs should not reread and recompute preprocessing for 369 subjects. Reusing a stale cache is unsafe when the raw manifest or preprocessing configuration changes.

### Decision

Store each preprocessed subject as compressed NumPy arrays (`.npz`) plus JSON metadata. Derive the cache key from the source manifest content identity, subject ID, crop margin, normalization version, and cache schema version. Write arrays through a temporary file and atomic replace; treat a cache as a hit only when both data and metadata files exist.

### Consequences

- Repeated subject access avoids NIfTI decoding and preprocessing work.
- Cache invalidation occurs automatically when source content or preprocessing settings change.
- Crop bounds, affine, spacing, and normalization statistics remain available for reconstruction and audit.
- The compressed cache is an intermediate representation and is excluded from version control.
- Real `BraTS20_Training_001` verification produced a cache miss followed by a hit with identical image/mask shapes.

## ADR-029: Lazy Dataset Access

### Context

The training cohort contains 369 four-channel 3D subjects and cannot be materialized in RAM. The cache should be consumed one subject at a time while remaining compatible with MONAI transforms.

### Decision

Implement a PyTorch `Dataset` that stores only subject IDs and manifest references at initialization, loads or creates one cached subject during `__getitem__`, and returns tensors plus geometry/cache metadata. Provide a small MONAI `PersistentDataset` factory for transform pipelines that need MONAI's own persistent caching.

### Consequences

- Dataset construction is lightweight and memory-safe.
- DataLoader workers can request subjects independently.
- Training receives fixed channel order and integer labels as tensors.
- Transform-level caching and subject preprocessing caching remain separate, explicit layers.
- Real dataset inspection confirmed 259 train subjects are indexed lazily and one cache-backed sample loads with expected tensor shapes and labels.

## ADR-030: Foreground-Biased Patch Sampling

### Context

Tumor voxels occupy a small fraction of each 3D volume. Uniform patch sampling would frequently produce all-background patches and make the class imbalance harder to address through optimization alone.

### Decision

Sample patch centers from tumor voxels with probability `2/3` and sample uniformly random centers with probability `1/3`. Pad volumes when a configured patch is larger than the cropped subject. Return center type and tumor voxel fraction with every sample so the distribution can be measured.

### Consequences

- Training sees tumor structures more frequently without discarding background context.
- Patch-size and sampling-probability changes remain explicit configuration changes.
- Small cropped subjects remain valid inputs through zero padding.
- Sampling behavior can be audited from counts rather than inferred from code.
- Real `BraTS20_Training_001` sampling produced a `0.68` foreground-center fraction over 100 patches, consistent with the configured `2/3` probability.

## ADR-031: Preprocessing Dataset Lineage

### Context

The same raw manifest can produce different caches when normalization, crop margin, modality order, or preprocessing schema changes. PostgreSQL needs to distinguish those versions and support idempotent reruns.

### Decision

Persist a `dataset_versions` record keyed by a hash of the source manifest ID and canonical preprocessing configuration. Store the config, cache root, subject count, status, and source manifest foreign key. Reusing the same source/config pair returns the existing version instead of creating a duplicate.

### Consequences

- Training runs can later reference an exact preprocessing version.
- Changing preprocessing settings creates an explicit new lineage node.
- Cache paths and configuration remain queryable independently of local filesystem state.
- The verified preprocessing version is `c36885b173a5705fc5c17f1844e45400d5619e7a2a4a661b516671efca90eb10`, with config hash `d150e921e4d8fa808a27b11cb9b9a629708475c89163abf153ec1466ec8484c7`; an identical rerun was skipped.

## ADR-032: Baseline 3D U-Net

### Context

BraTS segmentation requires four MRI input modalities and labels encoded as background plus three tumor classes. The baseline must be understandable, configurable, and compatible with memory-constrained patch training.

### Decision

Implement a 3D U-Net with four input channels, four output logits corresponding to background and labels `{1,2,4}`, configurable channel widths and convolution depth, instance normalization, leaky ReLU, and interpolation-safe skip connections.

### Consequences

- The output is directly compatible with multiclass cross-entropy and later region conversion.
- Channel widths can be reduced for GPU experiments without changing the model interface.
- Instance normalization avoids dependence on large batch statistics.
- The baseline is intentionally separate from attention and automated model-selection variants.
- The baseline passed CPU/unit checks and a CUDA forward/backward smoke test with 1,779,156 parameters at `64^3` patch size.

## ADR-033: Attention U-Net Candidate

### Context

Enhancing tumor is often the smallest and most difficult region. Attention gates provide a controlled architectural candidate that can suppress irrelevant skip features using decoder context.

### Decision

Reuse the baseline encoder-decoder interface and add one additive attention gate per skip connection. The candidate keeps four input channels, four output logits, normalization, activation, and patch-shape behavior consistent with the baseline.

### Consequences

- Architecture comparison can focus on the attention mechanism rather than unrelated input/output changes.
- Attention adds parameters and compute that must be tracked during model selection.
- The baseline remains available as a simpler reference candidate.
- The attention candidate passed unit and CUDA smoke tests with 1,784,703 parameters at `64^3` patch size.

## ADR-034: BraTS Composite Regions

### Context

BraTS labels use values `0, 1, 2, 4`, while segmentation metrics are conventionally reported for Whole Tumor, Tumor Core, and Enhancing Tumor. Model logits use contiguous channels for cross-entropy, so the enhancing class must be mapped explicitly.

### Decision

Define regions as WT=`1+2+4`, TC=`1+4`, and ET=`4`. Treat model channels as `(background, NCR/NET, edema, enhancing)` and map argmax channel `3` to label `4`. Provide NumPy and Torch implementations with identical semantics.

### Consequences

- Ground truth and predictions use the same region definitions.
- Metrics do not depend on ad hoc label logic in individual evaluators.
- The non-contiguous source label `4` remains correct in saved predictions and visualizations.
- Real `BraTS20_Training_001` conversion produced WT `211,979`, TC `43,185`, and ET `27,742` voxels.

## ADR-035: Segmentation Metrics

### Context

Dice measures overlap but can hide boundary errors, while HD95 captures robust boundary distance. The project must report both separately for the three BraTS composite regions.

### Decision

Implement Dice and 95th-percentile Hausdorff distance for WT, TC, and ET. HD95 uses voxel spacing so distances are expressed in physical units; both-empty regions score HD95 `0`, while one-empty regions return `inf` and remain visible to downstream aggregation.

### Consequences

- Region-specific overlap and boundary quality are both available.
- Metrics remain comparable across subjects with spacing differences.
- Missing-region behavior cannot silently appear as a good score.
- Aggregate evaluation will report mean and standard deviation per region/metric.
- Real self-comparison verified Dice `1.0` and HD95 `0.0` for all three regions; aggregate output keeps separate mean/std values for each metric.

## ADR-036: Segmentation Loss

### Context

Dice directly optimizes overlap but can be unstable under class imbalance, while cross-entropy supplies voxel-level classification gradients. The training objective must expose the balance between them for model selection.

### Decision

Use a normalized weighted combination of foreground-only soft multiclass Dice loss and four-class cross-entropy. Expose independent Dice and cross-entropy weights plus optional class weights, and reject configurations where both component weights are zero.

### Consequences

- The loss addresses overlap and voxel classification together.
- Weight changes are explicit hyperparameters for Optuna.
- The background is excluded from the Dice term but remains part of cross-entropy.
- The loss is differentiable and compatible with mixed precision training.
- The loss passed three tests and a CUDA gradient check with finite output and non-null gradients.

## ADR-037: Resource-Aware Training Utilities

### Context

The target GPU has 12 GB of VRAM and physical batch sizes may be 1-2 for 3D patches. Resource behavior must be explicit, reusable, and observable rather than scattered across model code.

### Decision

Provide shared utilities for CUDA-only AMP contexts and grad scalers, gradient accumulation with optional clipping, atomic checkpoint save/load, and peak GPU memory reset/read operations.

### Consequences

- Training code can reach a larger effective batch size without requiring larger physical batches.
- Checkpoints cannot be observed in a partially written state.
- Memory usage becomes a metric that can be logged per step/epoch.
- CPU execution remains available for tests, while CUDA is required for the performance target.
- The CUDA micro-step verified finite AMP training loss `1.50619` and peak-memory reporting at `6,009,344` bytes.

## ADR-038: Sliding-Window Inference

### Context

The model trains on patches but evaluation and serving must produce predictions for complete cropped volumes. Direct full-volume inference may exceed GPU memory and patch seams can create inconsistent predictions.

### Decision

Run overlapping windows, apply a Gaussian importance map that emphasizes window centers, accumulate weighted logits and weights, normalize, and crop any temporary padding back to the original volume shape. Support window batching and optional CUDA AMP while accepting batch size one.

### Consequences

- Full-volume outputs are reconstructed without requiring full-volume model memory.
- Overlap and Gaussian weighting reduce border artifacts.
- Padding allows windows larger than small test volumes.
- Window size, overlap, and batch size are explicit inference configuration.
- Sliding-window tests passed constant-field reconstruction and CUDA synthetic inference preserved an `80x88x96` output shape.

## ADR-039: Overfit Sanity Gate

### Context

Data alignment, labels, loss, and model wiring can fail while a normal training run still produces plausible logs. A fixed-batch overfit test provides an early, automated proof that the complete modeling path can learn a small sample.

### Decision

Before model selection, train a small baseline U-Net on one deterministic foreground-centered patch. Record the full loss/foreground-Dice curve and pass only when final Dice exceeds the configured threshold and final loss falls below the configured initial-loss ratio.

### Consequences

- Failed data/model/loss paths halt downstream model selection.
- Gate thresholds and seed are explicit and reproducible.
- The gate is diagnostic, not a performance estimate for the full dataset.
- The result artifact can be used by orchestration to block later stages.

### Label Contract Correction

The first real CUDA gate run exposed that stored BraTS label `4` cannot be passed directly to four-class cross-entropy, which expects class indices `0..3`. The shared region utilities now explicitly map stored labels `0/1/2/4` to model classes `0/1/2/3` and map predictions back before metric/reporting use. This is a required correctness boundary, not a dataset modification.

The gate also retains and evaluates the best-Dice model state rather than the final epoch, preventing late optimization regression from masking a successful overfit. The final-loss ratio default is `0.30`, requiring at least a 70% loss reduction while accommodating stable best-epoch overfit behavior observed on real patches. The real gate passed on CUDA with a `32^3` patch, best foreground Dice `0.97661`, and best loss `0.06890` from initial `1.19585`.

## ADR-040: Sanity Gate Enforcement

### Context

A Celery or CLI workflow can accidentally continue after a failed task, missing artifact, or swallowed exception. Model selection must not run on an unverified modeling path.

### Decision

Require the sanity artifact immediately before model selection. Reject missing or invalid JSON and accept only an explicit `status: passed`; any other state raises a hard error.

### Consequences

- A failed sanity test cannot silently produce trial records.
- Orchestration has a reusable gate function rather than duplicating status checks.
- The sanity artifact becomes a required dependency of all selection workflows.

## ADR-041: MLflow Tracking and Artifacts

### Context

Model-selection and training runs need queryable parameters, metrics, artifacts, and later model-registry state. The project already runs MLflow in Docker alongside PostgreSQL.

### Decision

Use the MLflow server at `http://localhost:5000` for host-side scripts. Store MLflow backend metadata in the Docker PostgreSQL database and artifacts in the persistent `mlflow-artifacts` Docker volume. Standardize experiment creation and run metadata through `app/tracking/mlflow.py`.

### Consequences

- Host training code does not write directly to the MLflow backend database.
- Artifacts survive container recreation while the volume is retained.
- Container-to-container MLflow clients should use `http://mlflow:5000`; host clients use `http://localhost:5000`.
- A live health/sample run is required before model-selection tasks depend on tracking.
- The stock MLflow image lacked `psycopg2`, so the service uses a project image that adds `psycopg2-binary` before connecting to PostgreSQL.
- MLflow health returned `2.19.0`; sample run `62da732a868c4f24a2e2ccaf0a9f8424` finished successfully in experiment `1` with metric `health_check=1.0` and a diagnostic artifact.

## ADR-042: Bounded Optuna Search

### Context

Automated model selection must vary meaningful modeling choices without consuming unbounded GPU time. The search also must use the verified sanity gate before training trials.

### Decision

Use Optuna with a deterministic TPE sampler to search architecture, patch size, learning rate, and Dice/Cross-Entropy weighting. Enforce explicit trial count, epoch, and elapsed GPU-time budgets through `SearchConfig`; reject invalid budgets before starting.

### Consequences

- Search behavior is reproducible from the seed and configuration.
- Compute exposure is bounded and visible in the selection artifact.
- Short trials use the real cached data, model, label mapping, and loss path.
- The selected configuration remains a candidate for full training, not a final performance claim.

## ADR-043: Trial and Winner Lineage

### Context

Logging only the winner would hide failed or inferior configurations and weaken reproducibility. The selected trial must be traceable to its experiment run and parameters.

### Decision

Create one MLflow run per Optuna trial, logging all suggested parameters and trial metrics. Persist every trial plus the best trial number, value, parameters, and MLflow run ID in `artifacts/model-selection.json`.

### Consequences

- Losing trials remain queryable for comparison.
- The winner can be promoted to full training without manual parameter copying.
- Trial variation is testable and visible rather than implied.

## ADR-044: Full Training Workflow

### Context

The winning short-budget trial must be promoted automatically into a reproducible full-training run without manually copying parameters. The target GPU requires patch-based training and explicit resource tracking.

### Decision

`scripts/train_full.py` consumes `artifacts/model-selection.json`, trains the selected architecture/configuration using cached subjects, AMP-capable utilities, checkpointing, and MLflow per-epoch logging, and writes a training result/checkpoint artifact under `artifacts/full-training`.

### Consequences

- Full training is traceable to the selected trial parameters.
- Checkpoint and training history become inputs to registration and evaluation.
- The initial implementation logs epoch loss, duration, and peak GPU memory; evaluation metrics are produced by the following held-out evaluation stage.
- Full training normalizes persisted selection parameters by deriving `cross_entropy_weight = 1 - dice_weight` when the Optuna artifact contains only the independently sampled Dice weight.

## ADR-046: Training Progress and System Metrics

### Context

Full 3D training can take several minutes per epoch and silent terminal output makes a healthy process look stalled. Resource usage also needs to be queryable alongside model metrics.

### Decision

Use nested `tqdm` progress bars for epochs and subjects, with live loss and peak GPU-memory postfix values. Start MLflow runs with system-metrics logging enabled by default, capturing supported host/GPU metrics through MLflow; provide `--no-system-metrics` only for environments where collection is undesirable or unsupported.

### Consequences

- Training visibly reports forward/backward progress instead of waiting for epoch completion.
- MLflow stores system metrics alongside training parameters and metrics.
- The terminal remains usable in non-interactive logs because tqdm dynamically adapts its output.
- System-metrics collection adds a small monitoring overhead and can be disabled explicitly.

## ADR-047: Training Throughput

### Context

The initial loop loaded and preprocessed one subject synchronously before every GPU step, making the GPU wait on disk/CPU work. Training speed should improve without reducing subjects, epochs, patch size, architecture, or loss.

### Decision

Pre-cache all training subjects before full training, consume fixed-size cached patches through a PyTorch `DataLoader`, use configurable worker processes, pinned memory, and nonblocking device transfers. Enable cuDNN benchmarking for fixed patch shapes and expose `torch.compile(mode="reduce-overhead")` as an opt-in flag rather than making compilation a required dependency.

### Consequences

- CPU/cache work can overlap with GPU training.
- The amount and semantics of training data remain unchanged.
- Worker count and batch size become explicit runtime settings.
- Compilation must be benchmarked separately because startup cost and compatibility vary by model/environment.
- On Windows with the current PyTorch environment, TorchInductor reported `TritonMissing`; `--compile` now records the request and falls back to eager CUDA execution rather than failing the training run.
- The initial GPU observation was explained by the selected `4,8,16` channel model, `32^3` patches, and batch size `1`; full training now accepts explicit channel/patch overrides and records them in MLflow rather than silently changing the selected configuration.
- The optimized 8-subject smoke run passed with Attention U-Net channels `(16,32,64,128)`, `64^3` patches, batch size `2`, peak memory `791,608,832` bytes, and throughput `0.37786` subjects/sec. This configuration remains a throughput smoke configuration until full validation confirms its quality.
- Added an explicit `high-memory` profile targeting channels `(32,64,128,256)`, `112^3` patches, and batch size `4` to target roughly 9-10 GB of VRAM. Actual usage must be verified from the resulting peak-memory metric; the profile does not silently alter ordinary runs.
- The full high-memory run completed all 10 epochs over 259 subjects in `23:08` at `138.87s/epoch`, with final loss `0.25310` and peak reported training memory `5351 MB`. MLflow system monitoring started and terminated normally.
- Full training now persists visual evidence for every future run: train/validation loss curves, epoch duration, peak memory, and one prediction/ground-truth preview for each train/validation/test split. Test previews are observational only and do not influence optimization.

## ADR-048: Explainability Artifacts

### Context

Aggregate metrics do not show where the model fails or how uncertain it is.

### Decision

Generate a worst-case overlay gallery from held-out predictions and estimate uncertainty through deterministic TTA with voxel variance and entropy summaries.

### Consequences

- Failure cases become inspectable as image artifacts rather than only metric rows.
- Uncertainty is measured and stored for selected subjects.
- External reference comparisons are outside the current project scope.
- Gallery and uncertainty generation reuse the exact trained checkpoint configuration.
- The trained checkpoint was registered as model `brainseg-segmentation`, version `1`, through registration run `8b8310ae8d744f89b1efe76ea1e8e215`. MLflow warned that the local CUDA suffix was normalized in the generated pip requirement; the registry operation itself succeeded.
- Real serving smoke loaded registry model `brainseg-segmentation:1` on CUDA and produced a `151x181x145` segmentation in `510.65ms`.
- Inference telemetry is verified in PostgreSQL with a completed CUDA event at `503.93ms`; the API uses the project database default when no database environment variable is present.
- The post-training single trigger was verified with run `962b85524c6a4ee384ff8fdda21a52c5`; `verify_primary_model`, `evaluate_primary_model`, and `publish_primary_results` completed in order without retraining.
- Idempotency probe verified unchanged payload skipping and changed payload invalidation through a PostgreSQL-backed Celery stage.
- Dashboard analytics endpoint verified with live PostgreSQL-backed counts of 3 pipeline runs, 7 stages, and 1 inference event.
- Clean Compose validation verified CPU API startup, artifact/cache mounts, analytics HTTP 200, viewer subject-list HTTP 200, MLflow health, PostgreSQL, and Redis startup.
- Model analytics endpoint verified with 10 training epochs, WT/TC/ET evaluation data, 14 MLflow runs, and the canonical primary-model metadata.
- Evaluation must use the configuration persisted beside the checkpoint in `training-result.json` when available; the Phase 3 selection artifact may describe the smaller search model while the actual full-training checkpoint uses an explicit high-memory override.
- One-subject evaluation smoke passed with the high-memory checkpoint: WT Dice `0.93687`, TC Dice `0.86951`, ET Dice `0.83441`; corresponding HD95 values were `2.2361`, `10.8167`, and `2.8284`. These are smoke results, not final held-out performance.
- Full held-out evaluation completed for 55 subjects: WT Dice `0.83091`/HD95 `28.5352`, TC Dice `0.71299`/HD95 `21.1474`, ET Dice `0.67402`/HD95 `Infinity`. Infinite ET HD95 reflects empty prediction/target cases and must be handled explicitly in validation/reporting.
- HD95 aggregation now excludes non-finite empty-region cases from mean/std while recording `valid_count`, `non_finite_count`, and per-subject `non_finite_hd95_cases`; such cases remain visible rather than contaminating aggregate statistics.
- Final evaluation validation passed. Aggregate metrics are WT Dice `0.83091`/HD95 `28.5352`, TC Dice `0.71299`/HD95 `21.1474`, and ET Dice `0.67402`/HD95 `18.4937` over 53 finite HD95 cases. ET HD95 cases for subjects `BraTS20_Training_275` and `BraTS20_Training_310` remain explicitly recorded as non-finite.
- The failure gallery was generated for five worst subjects. The trained Attention U-Net uses deterministic flip TTA because it has no dropout layers.
- TTA uncertainty completed for subjects `275` and `310` with 7 transforms: variance means `0.000767` and `0.000557`, maximum variances `0.106529` and `0.114900`, and nonzero entropy maxima. This replaces the invalid zero-variance MC-dropout artifact.

### System-Metrics Dependency Correction

MLflow system metrics require monitoring dependencies in the training environment. `psutil` is required for host CPU/RAM metrics, and `nvidia-ml-py` provides NVML access for GPU utilization/memory metrics. These are now explicit project dependencies, and system metrics must be checked through the run's stored metric keys rather than inferred from the `log_system_metrics` flag alone. The project uses a 1-second sampling interval and logs after the first sample so short smoke runs are observable; longer production runs still retain frequent samples.

## ADR-045: Full-Volume Evaluation Gate

### Context

Patch-level metrics are not valid final segmentation results. The model must be evaluated on reconstructed complete subjects, with regional overlap and boundary metrics retained separately.

### Decision

Evaluate test subjects using the existing Gaussian-blended sliding-window predictor and report per-subject and aggregate WT/TC/ET Dice and HD95. Generate failure-case records and reject invalid/suspicious metric artifacts before model promotion.

### Consequences

- Final metrics correspond to full reconstructed volumes.
- Small-region behavior remains visible instead of being hidden by one average.
- Model registration and later serving depend on a completed training checkpoint and evaluation artifact.

## Cross-Cutting Rules

- Record all configuration values that affect data, training, evaluation, or serving.
- Use content/configuration hashes to prevent stale cache reuse.
- Treat a failed stage as a real failure; never swallow exceptions and report success.
- Report WT, TC, and ET separately instead of hiding performance behind one average Dice.
- Investigate implausibly high first-pass results, especially Dice above approximately `0.98`.
- Cite Menze et al. 2015, Bakas et al. 2017, and Bakas et al. 2018 wherever BraTS results are presented.
- State that the project is for research/portfolio use and is not a medical device or diagnostic system.
