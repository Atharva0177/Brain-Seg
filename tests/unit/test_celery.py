from pipeline.tasks.celery_app import celery_app


def test_celery_tasks_are_registered() -> None:
    assert "brainseg.health_check" in celery_app.tasks
    assert "brainseg.run_stage" in celery_app.tasks
    assert "brainseg.run_ordered_stage" in celery_app.tasks
