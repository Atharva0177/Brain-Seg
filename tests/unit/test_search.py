import optuna
import pytest

from pipeline.model_selection.search import SearchConfig, suggest_trial_config


def test_search_space_contains_required_parameters() -> None:
    trial = optuna.trial.FixedTrial(
        {"architecture": "unet", "patch_size": 32, "learning_rate": 0.001, "dice_weight": 0.5}
    )
    params = suggest_trial_config(trial, SearchConfig())
    assert set(
        ("architecture", "patch_size", "learning_rate", "dice_weight", "cross_entropy_weight")
    ) <= set(params)
    assert params["cross_entropy_weight"] == pytest.approx(0.5)


def test_search_budget_validation() -> None:
    with pytest.raises(ValueError):
        SearchConfig(trial_count=0).validate()


def test_search_space_can_produce_variation() -> None:
    sampler = optuna.samplers.RandomSampler(seed=7)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(
        lambda trial: float(trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)), n_trials=3
    )
    assert len({trial.params["learning_rate"] for trial in study.trials}) > 1
