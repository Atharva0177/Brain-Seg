# BrainSeg: 3D Brain Tumor Segmentation PRD

**Project codename:** BRATS
**Owner:** Atharva
**Status:** Draft, ready for AI coding agent handoff
**Target use:** ML/CV portfolio project for job applications

---

## 1. Overview

Build a fully automated, end-to-end pipeline that takes multi-modal brain MRI volumes (T1, T1ce, T2, FLAIR) from the BraTS dataset through to a served, tracked segmentation model, with no manual intervention required between stages. A single trigger (CLI command or API call) runs the entire sequence: download, preprocess, select a model configuration, train, evaluate, and expose the result for inference, with every stage's inputs, outputs, metrics, and resource usage tracked and queryable afterward.

This is a step up in difficulty on two axes at once: the modeling problem itself (volumetric, multi-modal, memory-constrained, class-imbalanced segmentation), and the systems problem of orchestrating and tracking a multi-stage ML pipeline reliably, which is closer to what an ML engineering role actually looks like day to day than a one-off training notebook.

## 2. Problem Statement & Motivation

Most CV portfolio projects stop at "here's a notebook that trains a model." This project is explicitly framed around four things that read as hard to an ML hiring manager:

- **Volumetric data at scale.** A single subject is four 240x240x155 volumes. Loading and training on this forces real engineering decisions instead of `ImageFolder` and a pretrained backbone.
- **Multi-modal fusion.** Four MRI sequences carry different, partially redundant information the model has to learn to weigh.
- **Severe class imbalance and small-structure evaluation.** Enhancing tumor, the most clinically important region, is often the smallest and hardest to segment accurately.
- **Memory-constrained training.** Full 3D volumes at native resolution will not fit in GPU memory alongside a reasonably deep 3D CNN. Solving this properly, not by silently downsampling, is a core engineering narrative.
- **End-to-end automation and tracking.** Download, preprocessing, model selection, training, and inference are wired into one orchestrated, idempotent, observable pipeline rather than a sequence of manually-run scripts. This is the difference between "I can train a model" and "I can build a system that trains and maintains a model," which is the actual job description for most ML engineering roles.

## 3. Goals and Non-Goals

**Goals**
- A single trigger runs the full pipeline end to end: download, preprocess, select, train, evaluate, serve, with no manual step in between.
- Every pipeline stage is idempotent (safe to re-run without redoing completed work) and tracked (status, timing, resource usage, outputs all queryable after the fact).
- Model selection is automated: a defined search over architectures and key hyperparameters, with the winning configuration promoted to a full training run automatically, not hand-picked.
- Hit or approach published strong segmentation baseline performance on our own held-out split of the labeled BraTS training cohort (the official BraTS validation/test cohorts have no public labels, see Section 4).
- Produce an uncertainty-aware, explainable output, generated automatically as part of the pipeline, not a manual one-off analysis step.
- Ship a demo centered on the inference API, MLflow, PostgreSQL, CLI visualizations, and reproducible artifacts.

**Non-goals**
- No clinical validation or diagnostic claims of any kind. Every artifact and README states this is a research project on a public benchmark, not a medical device.
- No attempt to beat the current BraTS leaderboard. The bar is a credible, benchmarked, fully automated pipeline, not SOTA accuracy.
- No distributed/multi-node training infrastructure. Automation here means reliable single-machine orchestration, not a cluster scheduler.
- No production-grade secrets management or multi-tenant auth. Environment-variable-based credentials are sufficient for a portfolio deployment.
- No attempt at BraTS's overall survival prediction or pseudoprogression-vs-recurrence tasks. `survival_info.csv` (age, survival days, extent of resection) does ship with the training cohort, so this exclusion is a deliberate scope choice, not a data-availability gap; wiring a second, tabular/clinical modeling task into the pipeline is left out to keep the project focused on segmentation and uncertainty. Segmentation (Task 1) and uncertainty quantification (Task 4) are the only official challenge tasks in scope.

## 4. Dataset

**Source:** BraTS 2020 (Multimodal Brain Tumor Segmentation Challenge), accessed via the Kaggle-hosted copy of the official challenge data.

**Access:** Kaggle dataset, downloaded via the Kaggle API. The official challenge also has a Synapse/CBICA IPP route, but since the working copy is the Kaggle mirror, that is the one the automated download stage targets.

**Automated download requirements:**
- Download runs unattended via the Kaggle API (`kaggle datasets download`), not a manual browser download.
- Kaggle API credentials read from environment variables or a `.env` file excluded from version control, never hardcoded.
- Every downloaded file checksum-verified against an expected manifest before being marked usable; corrupted or partial downloads must fail the pipeline stage loudly rather than being silently passed on to preprocessing.
- Download is idempotent: re-running the pipeline against an already-complete local dataset must detect this and skip re-downloading, not re-pull everything from scratch.

**Composition (BraTS 2020, confirmed from the Kaggle Data Explorer):**
- **Training cohort:** 369 diffuse glioma subjects (a mix of high-grade and low-grade glioma) with pixel-level ground truth, the only cohort with public labels. 5 files per subject (T1, T1ce, T2, FLAIR, segmentation mask) = 1,845 `.nii` files.
- **Validation cohort (this Kaggle copy, distinct from the withheld official test cohort):** 125 cases, same four modalities, **no segmentation file**, since ground truth is not provided. 4 files per subject = 500 `.nii` files.
- Total: 2,345 `.nii` files plus 4 `.csv` files (name mapping and survival/clinical metadata, covered below) = 2,349 files, 42.8 GB total, uncompressed NIfTI (`.nii`, not `.nii.gz`).
- Data contributed by 19 institutions using different scanners and acquisition protocols, which is a genuine multi-institutional heterogeneity signal worth calling out directly in the domain-shift/portfolio narrative (Section 14), not something invented for effect.
- Four modalities per subject: native T1, post-contrast T1 (T1Gd/T1ce), T2, and T2-FLAIR, each a 240x240x155 volume.
- All volumes co-registered to the SRI24 anatomical atlas, resampled to 1mm^3 isotropic resolution, and skull-stripped, prior to distribution.
- Labels: 0 = background, 1 = necrotic and non-enhancing tumor core (NCR/NET), 2 = peritumoral edema (ED), 4 = GD-enhancing tumor (ET). Label 3 is not used (retired from the annotation scheme after BraTS 2016). Conventionally evaluated as three composite regions: Whole Tumor (1+2+4), Tumor Core (1+4), Enhancing Tumor (4).
- **The 4 CSV files** (`name_mapping.csv`, `survival_info.csv` for the training cohort; `name_mapping_validation_data.csv`, `survival_evaluation.csv` for the validation cohort) carry non-imaging clinical/lineage metadata: subject ID cross-references across BraTS challenge years, plus patient age, survival days, and extent of resection for the survival-prediction task. This means clinical data for the survival task **is actually present** in this dataset, correcting the earlier assumption that it wasn't; survival prediction remains a non-goal for this project by scope choice, not data availability (see Section 3).

**Storage/format consequence:** since the raw files are uncompressed `.nii` (42.8 GB total across both cohorts), the Phase 1 preprocessing/caching stage should write its normalized, cropped output as compressed `.nii.gz` (or an equivalent compressed array format), both to cut disk footprint and to speed up the repeated I/O that patch-based training does against the cache. Treat the 42.8 GB raw download as a working/scratch footprint, not the steady-state size the pipeline should be reading from during training.

**Important consequence for the split strategy:** because only the 369-subject training cohort has public ground truth, the project's own train/val/test split must be carved out of those 369 subjects. The official 125-case validation and 166-case test cohorts cannot be used as this project's held-out evaluation set, since they have no labels to score against locally. Subject-level split within the 369 (roughly 70/15/15, around 258/55/56 subjects), enforced and recorded at the manifest stage, never a slice- or patch-level split.

**Optional stretch goal, not a required deliverable:** BraTS 2020 has an official evaluation server (CBICA IPP) that scores submissions against the withheld validation/test labels. Since this is a legacy 2020 challenge, verify separately whether that server is still accepting submissions before planning around it; treat any submission to it as a bonus, not something the pipeline depends on.

**Data use agreement and citation requirements (binding, not optional):**
- Use of this dataset requires citing three specific manuscripts (Menze et al. 2015, IEEE TMI; Bakas et al. 2017, Nature Scientific Data; Bakas et al. 2018, arXiv) in any writeup, README, or portfolio page that presents results from it. Two further citations covering the TCGA-GBM and TCGA-LGG collection labels are requested where the venue allows. These citations belong in the project README and public documentation, not just a code comment.
- Combining this data with additional public or private data for augmentation is only permitted if results using BraTS data alone are also reported, with any difference discussed explicitly. If an external-data augmentation variant is ever added to the model-selection search space, the pipeline must produce a BraTS-only baseline result alongside it, not replace it.
- The challenge dataset also covers three further official tasks beyond segmentation (overall survival prediction, pseudoprogression vs. true recurrence classification, and uncertainty quantification of predicted segmentations). Survival prediction and pseudoprogression classification are explicit non-goals for this project (Section 3), a scope choice rather than a data limitation, since `survival_info.csv` is part of this dataset. The uncertainty quantification task, however, is exactly what Phase 5's uncertainty-map work already targets, and is worth stating as such: this project's explainability layer isn't an ad hoc addition, it's tackling one of the challenge's own official evaluation tasks.

## 5. Success Metrics

**Modeling metrics:** per-region Dice similarity coefficient (Whole Tumor, Tumor Core, Enhancing Tumor) reported separately, plus Hausdorff Distance 95th percentile (HD95) per region, since Dice alone can look acceptable while boundary quality is poor. Published strong segmentation results on BraTS cohorts have historically landed roughly in the low-to-high 0.90s Dice for Whole Tumor, mid-0.80s for Tumor Core, and high-0.70s to mid-0.80s for Enhancing Tumor; treat this as an approximate sanity band, not a target to chase, since exact figures vary by cohort year and split.

**Pipeline/systems metrics (new, given the automation goal):**
- A full end-to-end run (download through trained, evaluated, registered model) completes unattended, with every stage's status and duration recorded.
- Re-running the pipeline with unchanged inputs and configuration completes in a fraction of the original time via idempotent stage-skipping, proving the caching logic is real and not decorative.
- Every training run, including losing configurations from the model-selection sweep, is queryable afterward with its full hyperparameters, metrics, and artifacts.

## 6. System Architecture

```
Trigger (CLI command or API call)
  |
  v
Orchestrator (Celery + Redis)
  |
  |--> Stage: Download        --> checksum-verified raw data, manifest -> Postgres
  |--> Stage: Preprocess       --> normalized/cropped/cached volumes, dataset version -> Postgres
  |--> Stage: Sanity Gate      --> overfit-single-batch check; pipeline halts if this fails
  |--> Stage: Model Selection  --> Optuna sweep across architectures/hyperparams, every trial -> MLflow
  |--> Stage: Full Training    --> best config promoted, trains to convergence, logs -> MLflow, checkpoint registered
  |--> Stage: Evaluation       --> per-region Dice/HD95, uncertainty maps, failure gallery -> Postgres + MLflow
  |--> Stage: Inference/Serve  --> FastAPI endpoint, model pulled from MLflow registry
  |
  v
Postgres (pipeline/run/lineage state)  <-->  MLflow (hyperparameters, metrics, artifacts, model registry)
  |
  v
Backend analytics and CLI artifacts (pipeline status, training curves, model comparison, inference demo)
```

## 7. Memory Management Strategy (core modeling difficulty)

This remains the central modeling constraint and needs to be treated as a first-class design concern, not an afterthought discovered via an out-of-memory crash.

- **Patch-based training, not full-volume training.** Random 3D patches (128x128x128 as a starting point, 96x96x96 if memory forces it) rather than full 240x240x155 volumes.
- **Foreground-biased patch sampling.** Roughly two-thirds of sampled patches centered on tumor voxels, the remainder fully random, addressing class imbalance at the data level rather than relying entirely on loss weighting.
- **Sliding-window inference** at evaluation and serving time, reconstructing full-volume predictions from overlapping patches (roughly 50% overlap, Gaussian-weighted blending) since the model only ever trains on patches.
- **Mixed precision training** (`torch.cuda.amp`) on by default, roughly halving memory use.
- **Gradient checkpointing** in the encoder if the chosen depth still doesn't fit after patch size and AMP.
- **Gradient accumulation** to reach a reasonable effective batch size given a likely physical batch size of 1-2.
- **Lazy, cached data loading**, never the full dataset materialized in RAM at once; preprocessed volumes cached to disk and lazily loaded per subject (MONAI's `PersistentDataset` or a tuned `CacheDataset`).
- **Memory usage logged automatically, not reviewed manually.** `torch.cuda.max_memory_allocated()` captured every epoch and written to the run's tracking record in Postgres/MLflow as part of the automation layer, so memory problems show up as a queryable trend, not a surprise crash.

## 8. Model Architecture & Automated Model Selection

**Candidate architectures in the search space:**
- Baseline 3D U-Net: encoder-decoder with skip connections, instance normalization, and leaky ReLU.
- Attention U-Net variant: attention gates on the skip connections, intended to help with the small enhancing-tumor region specifically.

**Automated search process:**
- Optuna-driven sweep over architecture choice plus key hyperparameters: patch size (within a bounded range), initial learning rate, and loss weighting between Dice and Cross-Entropy/Focal terms.
- Each trial trained for a short, fixed epoch budget (enough to differentiate configurations, not enough for full convergence) to keep total sweep cost bounded.
- Explicit sweep budget cap (a fixed trial count, e.g. 15-20, or a fixed total GPU-hour ceiling) stated up front, since unbounded automated sweeps are an easy way to silently burn a free-tier weekly GPU quota.
- Selection criterion: a composite validation Dice score across the three regions after the short trial budget; the winning trial's configuration is promoted automatically to the full training stage.
- Every trial, including losing configurations, logged to MLflow with its full hyperparameters and metrics, so the sweep itself is inspectable afterward: the portfolio story is "the pipeline selected configuration X because of Y," not "I manually tried a few things."

**Full training:** the winning configuration retrained to convergence with the complete loss/regularization setup (deep supervision, TTA at inference), logged to MLflow, checkpoint registered in the MLflow model registry as the candidate for serving.

## 9. Automation & Orchestration Architecture

- **Orchestrator:** Celery with Redis as the broker, chaining the pipeline stages (download, preprocess, sanity gate, model selection, full training, evaluation, serving handoff) as a task chain, consistent with the stack already used on the ANPR project.
- **Idempotency:** every stage checks for existing valid output before running: the download stage checks the manifest and checksums, the preprocessing stage checks for a matching cached dataset version keyed by a content hash of the preprocessing config, training checks whether a run with an identical resolved configuration already completed successfully. Re-running the pipeline against unchanged inputs should skip completed stages rather than redo them.
- **Retry semantics:** transient failures (a dropped network connection during download, a temporary GPU allocation failure) retried with exponential backoff. Logic failures (the sanity-gate overfit test failing, a validation Dice that's suspiciously and impossibly high, an architecture that doesn't fit in memory even at the minimum patch size) must halt the pipeline and surface an explicit failure, never retry silently into a false success.
- **Dependency integrity:** a stage must not proceed if its declared upstream dependency did not complete successfully. This is stated explicitly because chained task calls that don't check upstream status are a common way pipelines silently run on stale or missing data.
- **State tracking (Postgres):** a `pipeline_runs` table (run id, trigger time, overall status) and a `pipeline_stages` table (run id, stage name, status, start/end timestamp, duration, resource usage, and a foreign key to whatever artifact or MLflow run id the stage produced). This is what makes "tracking of everything" queryable rather than just implied by log files.
- **Alternative considered:** Prefect or Airflow would be more purpose-built for ML pipeline orchestration (native DAG visibility, built-in caching semantics) than Celery/Redis, which is really a general task queue. Celery/Redis is the recommended default here for consistency with the existing project stack and because the pipeline's DAG is simple and linear enough not to need a dedicated orchestrator's scheduling complexity; if the pipeline later grows branching/conditional stages, Prefect is the natural upgrade path.

## 10. Tracking & Analytics Architecture

- **MLflow (self-hosted via Docker)** is the source of truth for anything modeling-related: hyperparameters per run, per-epoch metrics (loss, Dice per region), artifacts (checkpoints, sample prediction overlays, uncertainty maps), and a model registry marking which version is currently promoted to serving. Backend store can live in a separate database/schema on the same Postgres instance used for pipeline state; artifact store is local disk for portfolio scope, with S3/MinIO noted as a later upgrade if cloud-storage integration is worth demonstrating.
- **Postgres** is the source of truth for orchestration state and data lineage: which raw data manifest and preprocessing configuration produced which cached dataset version, and which dataset version fed which training run. This is deliberately kept separate from MLflow's modeling metrics so pipeline health and model quality can be reasoned about independently.
- **Backend analytics and CLI artifacts**, showing pipeline run history, training curves, model comparisons, GPU memory, wall-clock time, inference latency, and prediction overlays.
- This tracking layer is itself a portfolio artifact. It demonstrates MLOps competence (pipeline reliability, experiment reproducibility, lineage) layered on top of the segmentation modeling work, which is a distinct and valuable signal from pure research/notebook skills for ML engineering roles specifically.

## 11. Tech Stack

The automation requirement changes the earlier "lighter stack" call: this is now genuinely a multi-stage background pipeline with state to track, not a single training script, so the heavier default stack is warranted.

- **ML:** PyTorch, MONAI (medical-imaging transforms, sliding-window inference, caching utilities), nibabel/SimpleITK for NIfTI I/O, Optuna for the model-selection sweep.
- **Orchestration:** Celery + Redis.
- **Tracking DB:** PostgreSQL for pipeline/run/lineage state (plain Postgres, not TimescaleDB, since this is low-volume structured state rather than a genuine time-series workload).
- **Experiment tracking and model registry:** MLflow, self-hosted via Docker.
- **Serving:** FastAPI for the inference endpoint, pulling the currently promoted model from the MLflow registry.
- **Frontend:** none in the current scope; backend/API and CLI artifacts are the project surface.
- **Environment:** Docker Compose for Redis, Postgres, MLflow, and FastAPI, so the backend stack can be brought up with one command.

## 12. Phased Delivery Plan

Each phase still has a hard proof-of-work gate, and each phase's validated logic gets wrapped as an automated, idempotent, tracked pipeline stage before the project moves on, rather than remaining a manually-run script.

### Phase 0: Environment & Automated Data Acquisition
- Automated download task via the Kaggle API, checksum verification, idempotent re-run detection, manifest written to Postgres. Manifest validation checks exact expected counts, not just checksums: 1,845 `.nii` files under the training cohort (369 subjects x 5: T1/T1ce/T2/FLAIR/seg), 500 `.nii` files under the validation cohort (125 subjects x 4, no seg), and the 4 metadata `.csv` files, for 2,349 files and roughly 42.8 GB total. A download that completes but produces a different file count should fail this gate, not pass silently.
- **Gate artifact:** `data_manifest.json` plus the corresponding Postgres record, a verification script confirming a random sample of volumes opens correctly with correct shape/affine/spacing, and a demonstrated no-op re-run (second trigger completes near-instantly because the manifest and checksums already validate).

### Phase 1: Automated Preprocessing Pipeline
- Preprocessing (normalization, foreground cropping, caching) wrapped as an idempotent Celery task keyed by a content hash of the preprocessing configuration.
- **Gate artifact:** before/after intensity histograms, patch-sampling visualization, foreground/background voxel ratio table, and the dataset version record in Postgres linking this cache to the manifest and config hash that produced it.

### Phase 2: Pipeline Sanity Gate
- The overfit-single-batch check, codified as an automated gate that must pass before the model-selection stage is allowed to run, not just a one-time manual check during development.
- **Gate artifact:** the overfit-test loss/Dice curve, and confirmation that the pipeline halts (rather than proceeding) if this check is re-run and fails.

### Phase 3: Automated Model Selection
- Optuna sweep across the architecture/hyperparameter search space defined in Section 8, bounded sweep budget, every trial logged to MLflow, winning configuration selected automatically.
- **Gate artifact:** the full sweep results table (every trial's config and resulting composite Dice) pulled from MLflow, and a record of which trial was auto-promoted and why.

### Phase 4: Automated Full Training
- The winning configuration retrained to convergence as its own tracked Celery task, checkpoint registered in the MLflow model registry.
- **Gate artifact:** training/validation loss and Dice curves, per-epoch memory logs, the registered model version in MLflow, and the first honestly reported per-region validation Dice. Any Dice above roughly 0.98 on first pass is treated as a red flag requiring investigation, not a win.

### Phase 5: Automated Evaluation & Explainability
- Evaluation (per-region Dice/HD95), MC dropout/TTA uncertainty maps, and failure-case gallery generation chained automatically after training completes, with results written to Postgres and MLflow.
- **Gate artifact:** uncertainty map visualizations and the failure-case gallery with written analysis, all produced without a manual trigger separate from the training stage completing.

### Phase 6: Orchestration Wiring, Serving & Dashboard
- Full Celery/Redis chain wiring every prior stage into one triggerable pipeline, FastAPI inference endpoint serving the MLflow-registry-promoted model, and reproducible CLI/artifact reporting.
- **Gate artifact:** a single command or API call executing the backend pipeline end to end on a clean environment, a reproducible demo runbook, and a demonstrated idempotent re-run.

## 13. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| GPU memory exhaustion during training | Blocks all training | Patch-based training, AMP, gradient checkpointing, gradient accumulation (Section 7) |
| Raw dataset disk footprint (42.8 GB uncompressed `.nii`) exceeds free-tier disk quota | Blocks download or preprocessing on constrained environments | Preprocess-then-compress to `.nii.gz` in Phase 1, avoid keeping both raw and cached copies simultaneously once preprocessing is verified |
| Enhancing tumor Dice much lower than Whole Tumor Dice | Expected, but easy to misreport | Report per-region, never a single averaged Dice; discuss the gap explicitly |
| Train/val leakage via patch- or slice-level splitting | Silently inflated, meaningless metrics | Subject-level split enforced at the manifest stage, verified before training |
| Automated sweep silently exceeding compute budget | Burns free-tier GPU quota, stalls the project | Explicit bounded trial count/GPU-hour cap on the Optuna sweep, enforced in code, not just documented |
| Pipeline stage silently proceeding on a failed upstream stage | Corrupted or meaningless downstream results | Explicit dependency/status checks between chained Celery tasks, not just call-order assumptions |
| Idempotency bugs (stale cache reused, or unnecessary rework) | Wasted compute or incorrect results reused | Content-hash-keyed caching for preprocessing and training configs, tested by deliberately re-running with unchanged and changed inputs |
| Missing required BraTS citations in README/portfolio pages | Violates the dataset's binding data use agreement | The three mandatory citations (Menze 2015, Bakas 2017, Bakas 2018) included in the README and dashboard about page as a checked step in Phase 6, not left to memory |
| External data mixed into training without a BraTS-only comparison | Violates the data use agreement's reporting condition | Any external-data augmentation variant added to the model-selection search space must ship with a BraTS-only baseline result alongside it |
| Framing as a diagnostic tool | Ethical and credibility risk | Every artifact, README, and dashboard states research/portfolio use only |

## 14. Explainability and Portfolio Narrative

The interview narrative now has two layers, not one: the modeling story (multi-modal volumetric data forced real memory-management decisions, subject-level splitting avoided a common leakage trap, per-region metrics reported honestly including where the model struggles, uncertainty maps surface failure cases instead of hiding them), and the systems story (a single trigger runs the entire pipeline unattended, every run is idempotent and re-runnable, model selection is an automated search rather than manual guesswork, and everything is tracked and queryable afterward). That second layer is what distinguishes this from a strong Kaggle notebook and starts to look like production ML engineering. Worth stating explicitly in any writeup: the uncertainty-map work in Phase 5 isn't a bolted-on interpretability gesture, it's an attempt at BraTS's own official Task 4 (uncertainty quantification), which is a stronger claim than most portfolio segmentation projects can make.

## 15. Evaluation Protocol

- Always evaluate on full reconstructed volumes via sliding-window inference, never on isolated patches.
- Report Dice and HD95 per region (WT/TC/ET), per subject, then aggregated as mean plus standard deviation.
- Evaluation is triggered automatically as the pipeline stage following training completion, not run manually after the fact, and its results are written to both Postgres and MLflow.
- Include a small qualitative section: side-by-side slice visualizations of ground truth vs. prediction for representative and worst-case subjects, generated as part of the automated evaluation stage.

## 16. Agent Handoff Conventions

- Task IDs formatted `BRATS-XXX`, tracked in `TASKS.md` with phase, description, and the specific gate artifact each task must produce.
- Git commit convention: `feat(BRATS-012): implement sliding-window inference`, `fix(BRATS-019): correct patch sampling foreground bias`.
- No task is marked complete without linking to the artifact path or Postgres/MLflow record it produced, not just a claim that it was done.

**Known failure modes to watch for specifically on this project**, extended for the automation layer:
- Writing the full training loop against data paths before Phase 0's manifest actually exists.
- Reporting validation Dice computed on patches instead of full sliding-window-reconstructed volumes.
- Splitting the dataset at the patch or slice level instead of the subject level.
- Silently reducing patch size or volume resolution to dodge a memory error without surfacing it as a documented tradeoff.
- Claiming a model was "trained for N epochs" when the loss curve is flat or NaN, or the saved checkpoint is unchanged from initialization.
- **New, automation-specific:** a Celery task marked "success" when it actually caught an exception internally and swallowed it; an idempotency check that always returns "already done" regardless of whether the config actually matches (silently reusing stale results); or a model-selection sweep that logs trials but doesn't actually vary the hyperparameters between them.

## 17. Appendix: Compute Budget and Hardware Options

- **Primary target: local RTX 5070, 12 GB.** Patch-based training at roughly 128^3 with batch size 1 and AMP should fit for the custom 3D models; drop to 96-112^3 if it doesn't. This is Blackwell architecture (compute capability sm_120), which needs CUDA 12.8 and an sm_120-compatible PyTorch build; verify with `torch.cuda.is_available()` and `torch.cuda.get_device_name()` before running anything else, since early Blackwell driver/framework mismatches silently fell back to CPU rather than erroring loudly. Running locally also means Phase 0's download and Phase 1's preprocessing aren't bound by a remote notebook's disk quota, which matters given the 42.8 GB raw dataset (Section 4).
- **Fallback / burst capacity:** Kaggle notebooks offer a P100 or dual T4 with roughly 30 GPU-hours per week, 16 GB of VRAM per GPU. Useful if the local card is tied up or for running the Optuna sweep (Phase 3) in parallel with local full training.
- **Low-cost tier:** Colab Pro for more consistent access to a faster GPU if iteration speed becomes a bottleneck.
- **Timing:** actual training and sweep time depends heavily on final patch size, batch size, and hardware; do not commit to a full pipeline run's duration until Phase 3's sweep and Phase 4's first training epoch have been timed and extrapolated on the actual local card. The Optuna search space's patch-size upper bound (Section 8) should be set at what actually fits on 12 GB, not the 128^3 default assumed before the target hardware was confirmed.
