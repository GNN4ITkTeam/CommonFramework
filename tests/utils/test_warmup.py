from types import SimpleNamespace

import pytest
import torch

from acorn.utils.ml_utils import get_warmup_scale
from acorn.utils.ml_utils import PlateauWarmupCallback
from acorn.core.core_utils import get_trainer
from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage
from acorn.stages.graph_construction.models.metric_learning import MetricLearning
from acorn.stages.graph_construction.models.double_metric_learning import (
    DoubleMetricLearning,
)
from acorn.stages.track_building.models.ml_track_building import MLTrackBuildingStage


@pytest.mark.parametrize(
    "duration, expected",
    [(None, [1] * 6), (0, [1] * 6), (1, [1] * 6), (4, [0.25, 0.5, 0.75, 1, 1, 1])],
)
def test_warmup_scale(duration, expected):
    assert [get_warmup_scale(i, duration) for i in range(6)] == expected


@pytest.mark.parametrize(
    "stage",
    [EdgeClassifierStage, MetricLearning, DoubleMetricLearning, MLTrackBuildingStage],
)
@pytest.mark.parametrize("warmup", [None, 0, 4])
@pytest.mark.parametrize(
    "scheduler", [None, "StepLR", "ReduceLROnPlateau", "CosineAnnealingWarmRestarts"]
)
def test_stage_hook_preserves_only_plateau_warmup(stage, warmup, scheduler):
    model = SimpleNamespace(
        hparams=dict(warmup=warmup, lr=1.0, min_lr=0.2, scheduler=scheduler),
        trainer=SimpleNamespace(current_epoch=0),
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    actual = []
    for epoch in range(6):
        model.trainer.current_epoch = epoch
        optimizer.param_groups[0]["lr"] = 0.1
        PlateauWarmupCallback().on_before_optimizer_step(
            model.trainer, model, optimizer
        )
        stage.on_before_optimizer_step(model, optimizer)
        actual.append(optimizer.param_groups[0]["lr"])
    expected = (
        [0.25, 0.5, 0.75, 1, 0.2, 0.2]
        if scheduler == "ReduceLROnPlateau" and warmup
        else [0.2] * 6
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "scheduler", [None, "StepLR", "CosineAnnealingWarmRestarts", "ReduceLROnPlateau"]
)
@pytest.mark.parametrize("extra_callbacks", [False, True])
def test_trainer_installs_plateau_warmup_only_for_plateau(
    tmp_path, scheduler, extra_callbacks
):
    trainer = get_trainer(
        dict(
            accelerator="cpu",
            devices=1,
            nodes=1,
            max_epochs=1,
            stage_dir=str(tmp_path),
            log_wandb=False,
            scheduler=scheduler,
            export={"jit": True} if extra_callbacks else None,
            time_inference=extra_callbacks,
        ),
        str(tmp_path),
    )
    assert any(
        isinstance(callback, PlateauWarmupCallback) for callback in trainer.callbacks
    ) == (scheduler == "ReduceLROnPlateau")


@pytest.mark.parametrize("scheduler", [None, "StepLR", "CosineAnnealingWarmRestarts"])
def test_callback_leaves_other_schedulers_untouched(scheduler):
    model = SimpleNamespace(hparams=dict(scheduler=scheduler, warmup=4, lr=1.0))
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.37)
    PlateauWarmupCallback().on_before_optimizer_step(
        SimpleNamespace(current_epoch=0), model, optimizer
    )
    assert optimizer.param_groups[0]["lr"] == 0.37
