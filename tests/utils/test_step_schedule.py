import copy

import pytest
import torch

from acorn.utils.ml_utils import get_optimizers
from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage
from types import SimpleNamespace


def make_schedule(**overrides):
    config = dict(
        lr=1.0,
        min_lr=0.2,
        scheduler="StepLR",
        scheduler_interval="step",
        warmup=4,
        patience=6,
        factor=0.5,
    )
    config.update(overrides)
    optimizers, schedulers = get_optimizers([torch.nn.Parameter(torch.ones(1))], config)
    return optimizers[0], schedulers[0]


def advance(optimizer, scheduler, count, min_lr=0.2):
    values = []
    for _ in range(count):
        EdgeClassifierStage.on_before_optimizer_step(
            SimpleNamespace(hparams=dict(warmup=None, min_lr=min_lr)), optimizer
        )
        values.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return values


def test_step_warmup_decay_and_floor():
    optimizer, config = make_schedule()
    assert config["interval"] == "step"
    values = advance(optimizer, config["scheduler"], 20)
    assert values == pytest.approx(
        [0.25, 0.5, 0.75, 1] + [1] * 6 + [0.5] * 6 + [0.25] * 4
    )


def test_step_schedule_without_warmup():
    optimizer, config = make_schedule(warmup=0)
    assert advance(optimizer, config["scheduler"], 8) == pytest.approx(
        [1] * 6 + [0.5] * 2
    )


@pytest.mark.parametrize("interval", ["epoch", "step"])
@pytest.mark.parametrize("split", [2, 3, 4, 10, 23])
def test_schedule_resume_matches_uninterrupted_run(split, interval):
    overrides = dict(scheduler_interval=interval)
    optimizer, config = make_schedule(**overrides)
    scheduler = config["scheduler"]
    advance(optimizer, scheduler, split)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    expected = advance(optimizer, scheduler, 8)

    restored_optimizer, restored_config = make_schedule(**overrides)
    restored_config["scheduler"].load_state_dict(scheduler_state)
    restored_optimizer.load_state_dict(optimizer_state)
    actual = advance(restored_optimizer, restored_config["scheduler"], 8)
    assert actual == pytest.approx(expected)


def test_default_epoch_schedule_is_unchanged():
    optimizers, configs = get_optimizers(
        [torch.nn.Parameter(torch.ones(1))],
        dict(lr=1.0, patience=2, factor=0.5, warmup=None),
    )
    assert configs[0]["interval"] == "epoch"
    assert isinstance(configs[0]["scheduler"], torch.optim.lr_scheduler.StepLR)
    assert advance(optimizers[0], configs[0]["scheduler"], 5) == [1, 1, 0.5, 0.5, 0.25]


@pytest.mark.parametrize("interval", ["epoch", "step"])
@pytest.mark.parametrize("warmup", [0, 1, 4])
def test_sequential_schedule_boundaries(interval, warmup):
    optimizer, config = make_schedule(
        scheduler_interval=interval,
        patience=2,
        warmup=warmup,
    )
    expected = [0.25, 0.5, 0.75, 1] if warmup == 4 else [1] * warmup
    expected += [1, 1, 0.5, 0.5, 0.25, 0.25, 0.2, 0.2]
    assert advance(optimizer, config["scheduler"], len(expected)) == pytest.approx(
        expected
    )
    kind = (
        torch.optim.lr_scheduler.SequentialLR
        if warmup
        else torch.optim.lr_scheduler.StepLR
    )
    assert isinstance(config["scheduler"], kind)


@pytest.mark.parametrize(
    "overrides",
    [
        {"scheduler_interval": "batch"},
        {"scheduler_interval": "steps"},
        {"warmup": -1},
        {"warmup": 1.5},
        {"patience": 0},
        {"patience": 1.5},
        {"lr": 0},
        {"lr": -1},
        {"min_lr": -1},
        {"min_lr": 2},
        {"warmup": "4"},
        {"warmup_steps": 4},
        {"scheduler_interval": "epoch", "warmup": -1},
    ],
)
def test_invalid_step_schedule_configuration(overrides):
    with pytest.raises(AssertionError):
        make_schedule(**overrides)


@pytest.mark.parametrize("interval", ["epoch", "step"])
def test_cosine_uses_shared_warmup(interval):
    optimizers, configs = get_optimizers(
        [torch.nn.Parameter(torch.ones(1))],
        dict(
            lr=1.0,
            scheduler="CosineAnnealingWarmRestarts",
            scheduler_interval=interval,
            warmup=2,
            patience=2,
            min_lr=0.2,
        ),
    )
    optimizer, scheduler = optimizers[0], configs[0]["scheduler"]
    values = []
    for _ in range(5):
        values.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert values == pytest.approx([0.5, 1, 1, 0.6, 1])
