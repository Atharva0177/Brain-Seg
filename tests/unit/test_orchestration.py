from app.db.orchestration import PipelineArtifact, PipelineRun, PipelineStage


def test_orchestration_models_define_expected_tables() -> None:
    assert PipelineRun.__tablename__ == "pipeline_runs"
    assert PipelineStage.__tablename__ == "pipeline_stages"
    assert PipelineArtifact.__tablename__ == "pipeline_artifacts"
