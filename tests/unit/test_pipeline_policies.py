from pipeline.tasks.pipeline import celery_app


def test_idempotent_stage_task_is_registered() -> None:
    assert "brainseg.idempotent_stage" in celery_app.tasks
