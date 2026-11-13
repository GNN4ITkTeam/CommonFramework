import csv
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from acorn.core import core_utils, train_stage
from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage
from acorn.utils.ml_utils import get_optimizers


class TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters(
            dict(
                lr=0.01,
                scheduler="StepLR",
                scheduler_interval="step",
                warmup=4,
                patience=6,
                factor=0.5,
            )
        )
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.used_lrs = []
        self.validation_steps = []

    def configure_optimizers(self):
        return get_optimizers(self.parameters(), self.hparams)

    def training_step(self, batch, batch_idx):
        return (self.weight - batch[0]).square().mean()

    def on_before_optimizer_step(self, optimizer):
        EdgeClassifierStage.on_before_optimizer_step(self, optimizer)
        self.used_lrs.append(optimizer.param_groups[0]["lr"])

    def validation_step(self, batch, batch_idx):
        self.validation_steps.append(self.global_step)


@pytest.mark.parametrize("override", [{}, {"scheduler": "StepLR"}, {"scheduler": None}])
@pytest.mark.parametrize("load_only_model_parameters", [False, True])
def test_training_inherits_only_scheduler_for_trainer(
    monkeypatch, tmp_path, override, load_only_model_parameters
):
    model = TinyModel()
    model.hparams.update(
        scheduler="ReduceLROnPlateau", max_epochs=999, val_check_interval=123
    )
    config = dict(variable_with_prefix=True, max_epochs=2, **override)
    original_config = config.copy()
    trainer = Mock()
    get_trainer = Mock(return_value=trainer)
    monkeypatch.setattr(train_stage, "get_trainer", get_trainer)
    monkeypatch.setattr(train_stage, "wandb", None)
    monkeypatch.setattr(
        train_stage,
        "get_stage_module",
        Mock(return_value=(model, {}, str(tmp_path), "restored.ckpt")),
    )

    train_stage.lightning_train(
        config, TinyModel, load_only_model_parameters=load_only_model_parameters
    )

    expected = {"scheduler": "ReduceLROnPlateau", **original_config}
    get_trainer.assert_called_once_with(expected, str(tmp_path))
    assert config == original_config
    if load_only_model_parameters:
        trainer.fit.assert_called_once_with(model)
    else:
        trainer.fit.assert_called_once_with(model, ckpt_path="restored.ckpt")


@pytest.mark.parametrize("interval", [1.0, 4])
def test_trainer_forwards_validation_interval(tmp_path, interval):
    config = dict(
        accelerator="cpu",
        devices=1,
        nodes=1,
        max_epochs=2,
        stage_dir=str(tmp_path),
        log_wandb=False,
    )
    if interval != 1.0:
        config["val_check_interval"] = interval
    trainer = core_utils.get_trainer(config, str(tmp_path))
    assert trainer.val_check_interval == interval


@pytest.mark.parametrize(
    "interval, expected_lrs",
    [("step", [0.005, 0.01, 0.01, 0.01]), ("epoch", [0.01, 0.005])],
)
def test_lr_logging_follows_scheduler_interval(tmp_path, interval, expected_lrs):
    configured = core_utils.get_trainer(
        dict(
            accelerator="cpu",
            devices=1,
            nodes=1,
            max_epochs=2,
            stage_dir=str(tmp_path),
            log_wandb=False,
        ),
        str(tmp_path),
    )
    monitor = next(
        c for c in configured.callbacks if isinstance(c, LearningRateMonitor)
    )
    model = TinyModel()
    if interval == "epoch":
        model.hparams.update(scheduler_interval="epoch", warmup=0, patience=1)
    logger = CSVLogger(str(tmp_path), name=interval)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=logger,
        callbacks=[monitor],
        log_every_n_steps=2,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    train = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1)
    trainer.fit(model, train)
    with (Path(logger.log_dir) / "metrics.csv").open() as stream:
        values = [
            float(row["lr-AdamW"]) for row in csv.DictReader(stream) if row["lr-AdamW"]
        ]
    assert values == pytest.approx(expected_lrs)


def make_trainer(tmp_path, **kwargs):
    return Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
        max_epochs=3,
        accumulate_grad_batches=2,
        val_check_interval=4,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


@pytest.mark.parametrize("interval", ["step", "epoch"])
def test_warmup_with_floor_does_not_overshoot_base_lr(tmp_path, interval):
    model = TinyModel()
    model.hparams.update(min_lr=0.006, patience=2, scheduler_interval=interval)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=10,
        logger=False,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    train = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    trainer.fit(model, train)
    assert min(model.used_lrs) >= 0.006
    assert max(model.used_lrs) <= 0.01 + 1e-12
    assert model.used_lrs[:4] == pytest.approx(
        [0.006, 0.006 + 0.004 / 3, 0.006 + 0.008 / 3, 0.01]
    )
    assert model.used_lrs[4:6] == pytest.approx([0.01, 0.01])
    assert model.used_lrs[6:] == pytest.approx([0.006] * 4)


@pytest.mark.parametrize("interval", ["step", "epoch"])
def test_lr_logging_records_scheduler_value_before_floor_hook(tmp_path, interval):
    model = TinyModel()
    model.hparams.update(
        warmup=0, patience=1, min_lr=0.006, scheduler_interval=interval
    )
    logger = CSVLogger(str(tmp_path), name=interval)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval=None)],
        log_every_n_steps=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1))
    with (Path(logger.log_dir) / "metrics.csv").open() as stream:
        logged = [
            float(row["lr-AdamW"]) for row in csv.DictReader(stream) if row["lr-AdamW"]
        ]
    assert logged == pytest.approx([0.01, 0.005, 0.003])
    assert model.used_lrs == pytest.approx([0.01, 0.006, 0.006])


@pytest.mark.parametrize("resume_step", [2, 4, 10])
def test_optimizer_steps_with_accumulation_validation_and_resume(tmp_path, resume_step):
    train = DataLoader(TensorDataset(torch.ones(8, 1)), batch_size=1)
    val = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    expected = TinyModel()
    full = make_trainer(tmp_path / "full")
    full.fit(expected, train, val)
    assert expected.used_lrs == pytest.approx(
        [0.0025, 0.005, 0.0075, 0.01] + [0.01] * 6 + [0.005] * 2
    )
    assert expected.validation_steps == [2, 4, 6, 8, 10, 12]

    initial = TinyModel()
    checkpoint = tmp_path / "resume.ckpt"
    if resume_step in (2, 10):

        class InterruptAfterCheckpoint(Callback):
            def on_validation_end(self, trainer, pl_module):
                if trainer.global_step == resume_step:
                    trainer.save_checkpoint(checkpoint)
                    raise RuntimeError("Simulated mid-epoch interruption")

        partial = make_trainer(
            tmp_path / "partial", callbacks=[InterruptAfterCheckpoint()]
        )
        with pytest.raises(RuntimeError, match="Simulated mid-epoch interruption"):
            partial.fit(initial, train, val)
    else:
        partial = make_trainer(tmp_path / "partial", max_steps=resume_step)
        partial.fit(initial, train, val)
        partial.save_checkpoint(checkpoint)

    restored = TinyModel()
    resumed = make_trainer(tmp_path / "resumed")
    resumed.fit(restored, train, val, ckpt_path=str(checkpoint))
    assert initial.used_lrs + restored.used_lrs == pytest.approx(expected.used_lrs)
    torch.testing.assert_close(restored.weight, expected.weight)


@pytest.mark.parametrize("interval", ["step", "epoch"])
def test_scheduler_yaml_example(tmp_path, interval):
    example = (
        Path(__file__).resolve().parents[2]
        / "examples/Example_scheduler/scheduler_step.yaml"
    )
    with example.open() as stream:
        config = yaml.safe_load(stream)
    assert config["scheduler_interval"] == "step"
    if interval == "epoch":
        config.pop("scheduler_interval")
    model = TinyModel()
    model.hparams.clear()
    model.hparams.update(config)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
        max_epochs=3 if interval == "step" else 12,
        accumulate_grad_batches=2,
        logger=False,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    batches = 8 if interval == "step" else 2
    train = DataLoader(TensorDataset(torch.ones(batches, 1)), batch_size=1)
    trainer.fit(model, train)
    assert trainer.global_step == 12
    assert model.used_lrs == pytest.approx(
        [0.0025, 0.005, 0.0075, 0.01] + [0.01] * 6 + [0.005] * 2
    )
