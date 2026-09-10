from app.db.dataset_versions import dataset_version_id, preprocessing_config_hash


def test_config_hash_is_order_independent() -> None:
    assert preprocessing_config_hash({"margin": 8, "mode": "zscore"}) == preprocessing_config_hash(
        {"mode": "zscore", "margin": 8}
    )


def test_dataset_version_changes_when_config_changes() -> None:
    assert dataset_version_id("manifest", "config-a") != dataset_version_id("manifest", "config-b")
