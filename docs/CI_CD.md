# BrainSeg CI/CD

The GitHub Actions workflow is backend-only and intentionally separates source validation from heavyweight ML operations.

## Pull Requests and Pushes

The `BrainSeg CI` workflow runs:

- Ruff lint and format checks
- Mypy type checking
- The complete 66-test Python suite
- Release-index and Python syntax validation
- Backend-only source checks
- Docker Compose syntax validation
- API and MLflow image builds

The workflow does not download BraTS or run GPU training. Those operations require approximately 42.8 GB of data, external Kaggle credentials, a compatible CUDA GPU, and generated model artifacts that are not committed to source control.

## Required Repository Secrets

The automatic CI workflow requires no secrets. Kaggle and MLflow credentials remain local/manual because dataset and training operations are not part of pull-request CI.

## Local CI Equivalent

Run from an activated `brainseg` environment:

```powershell
python -m pip install -r requirements.txt
python -m pip install optuna celery[redis] redis kaggle SimpleITK scikit-image nibabel
ruff check app pipeline scripts tests
ruff format --check app pipeline scripts tests
mypy app/api --ignore-missing-imports --follow-imports=skip
python -m pytest tests -q
python scripts/validate_release_index.py
python -m compileall -q app pipeline scripts
docker compose config --quiet
docker compose build api
docker compose build mlflow
```

## Release Flow

1. Open a pull request and wait for all CI jobs to pass.
2. Run or verify the generated model artifacts locally.
3. Run the backend stack with Docker Compose.
4. Run the commands in `docs/COMMANDS.md` and attach release artifacts.
5. Use `docs/DEMO_RUNBOOK.md` for the final portfolio demonstration.

## Failure Policy

- Source, test, type, release-contract, and Compose failures block merge.
- Missing external model/data artifacts are reported by API routes as explicit artifact-unavailable responses.
- GPU training is never silently substituted with CPU execution for the target training workflow.
