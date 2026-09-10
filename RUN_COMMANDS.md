# BrainSeg Run Commands

The complete chronological command runbook is maintained at:

```text
docs/COMMANDS.md
```

Open it from the repository root:

```powershell
Get-Content docs/COMMANDS.md
```

It includes setup, Docker backend services, Kaggle acquisition, manifest validation, splitting, preprocessing, caching, model verification, training, MLflow, evaluation, inference, Celery orchestration, telemetry, and release validation commands.

The project is backend-only. There is no frontend dashboard to start.
