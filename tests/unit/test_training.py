from pathlib import Path

import torch

from pipeline.models.training import GradientAccumulator, load_checkpoint, save_checkpoint


def test_gradient_accumulator_steps_after_configured_count() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    accumulator = GradientAccumulator(optimizer, scaler, accumulation_steps=2)
    optimizer.zero_grad()
    first = accumulator.backward(model(torch.ones(1, 2)).sum())
    second = accumulator.backward(model(torch.ones(1, 2)).sum())
    assert first is False
    assert second is True


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=3, metrics={"loss": 0.5})
    restored = torch.nn.Linear(2, 1)
    checkpoint = load_checkpoint(path, restored, map_location="cpu")
    assert checkpoint["epoch"] == 3
    assert checkpoint["metrics"]["loss"] == 0.5
    assert all(
        torch.equal(first, second)
        for first, second in zip(model.parameters(), restored.parameters())
    )
