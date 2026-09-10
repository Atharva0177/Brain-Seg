from app.tracking.mlflow import TrackingConfig


def test_tracking_config_defaults_to_local_mlflow() -> None:
    config = TrackingConfig()
    assert config.tracking_uri == "http://localhost:5000"
    assert config.experiment_name == "brainseg"
