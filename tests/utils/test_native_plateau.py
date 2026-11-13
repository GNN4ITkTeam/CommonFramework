import copy

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage
from acorn.utils.ml_utils import get_optimizers
from acorn.utils.ml_utils import PlateauWarmupCallback


class PlateauModel(LightningModule):
    def __init__(self, warmup=None, monitor="val_loss"):
        super().__init__()
        self.save_hyperparameters(
            dict(
                lr=1.0,
                scheduler="ReduceLROnPlateau",
                warmup=warmup,
                patience=0,
                factor=0.5,
                min_lr=0.2,
                metric_to_monitor=monitor,
            )
        )
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.used_lrs = []
        self.validations = 0

    def configure_optimizers(self):
        return get_optimizers(self.parameters(), self.hparams)

    def training_step(self, batch, batch_idx):
        return self.weight.square().sum()

    def on_before_optimizer_step(self, optimizer):
        EdgeClassifierStage.on_before_optimizer_step(self, optimizer)
        self.used_lrs.append(optimizer.param_groups[0]["lr"])

    def validation_step(self, batch, batch_idx):
        self.validations += 1
        self.log(self.hparams.metric_to_monitor, torch.tensor(1.0), on_epoch=True)


def make_trainer(tmp_path, **kwargs):
    callbacks = [PlateauWarmupCallback(), *kwargs.pop("callbacks", [])]
    return Trainer(
        callbacks=callbacks,
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        default_root_dir=tmp_path,
        accumulate_grad_batches=2,
        val_check_interval=4,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs
    )


class UpstreamPlateauModel(PlateauModel):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.0, amsgrad=True)
        return [optimizer], [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=0, factor=0.5
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.hparams.metric_to_monitor,
            }
        ]


@pytest.mark.parametrize("warmup", [None, 0, 1, 2])
def test_native_plateau_configuration_and_resume(warmup):
    model = PlateauModel(warmup)
    optimizers, configs = model.configure_optimizers()
    config = configs[0]
    assert config["interval"] == "epoch"
    assert config["monitor"] == "val_loss"
    scheduler = config["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    scheduler.step(1)
    scheduler.step(2)
    saved = copy.deepcopy((optimizers[0].state_dict(), scheduler.state_dict()))
    restored_opts, restored_configs = model.configure_optimizers()
    restored = restored_configs[0]["scheduler"]
    restored.load_state_dict(saved[1])
    restored_opts[0].load_state_dict(saved[0])
    for metric in [0.5, 1, 2]:
        scheduler.step(metric)
        restored.step(metric)
        assert restored.get_last_lr() == scheduler.get_last_lr()


@pytest.mark.parametrize("warmup", [None, 2])
@pytest.mark.parametrize("monitor", ["val_loss", "score"])
def test_plateau_uses_epoch_cadence_and_legacy_warmup(tmp_path, warmup, monitor):
    model = PlateauModel(warmup, monitor)
    runner = make_trainer(tmp_path)
    train = DataLoader(TensorDataset(torch.ones(8, 1)), batch_size=1)
    val = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    runner.fit(model, train, val)
    assert model.validations == 6
    expected = (
        [1] * 8 + [0.5] * 4 if warmup is None else [0.5] * 4 + [1] * 4 + [0.5] * 4
    )
    assert model.used_lrs == pytest.approx(expected)
    scheduler = runner.lr_scheduler_configs[0].scheduler
    assert scheduler.last_epoch == 3  # Six validations, but three scheduler updates.


@pytest.mark.parametrize("resume_step", [2, 6])
def test_native_plateau_mid_epoch_resume_matches_upstream(tmp_path, resume_step):
    train = DataLoader(TensorDataset(torch.ones(8, 1)), batch_size=1)
    val = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    full = PlateauModel(warmup=2)
    make_trainer(tmp_path / "full").fit(full, train, val)
    checkpoint = tmp_path / "resume.ckpt"

    class Interrupt(Callback):
        def on_validation_end(self, trainer, pl_module):
            if trainer.global_step == resume_step:
                trainer.save_checkpoint(checkpoint)
                raise RuntimeError("Interrupt")

    initial = PlateauModel(warmup=2)
    with pytest.raises(RuntimeError, match="Interrupt"):
        make_trainer(tmp_path / "partial", callbacks=[Interrupt()]).fit(
            initial, train, val
        )
    restored = PlateauModel(warmup=2)
    make_trainer(tmp_path / "resumed").fit(restored, train, val, ckpt_path=checkpoint)
    reference_full = UpstreamPlateauModel(warmup=2)
    make_trainer(tmp_path / "reference_full").fit(reference_full, train, val)
    assert full.used_lrs == pytest.approx(reference_full.used_lrs)
    reference_initial = UpstreamPlateauModel(warmup=2)
    with pytest.raises(RuntimeError, match="Interrupt"):
        make_trainer(tmp_path / "reference_partial", callbacks=[Interrupt()]).fit(
            reference_initial, train, val
        )
    reference_restored = UpstreamPlateauModel(warmup=2)
    make_trainer(tmp_path / "reference_resumed").fit(
        reference_restored, train, val, ckpt_path=checkpoint
    )
    # Native Lightning may defer a plateau update after a mid-epoch resume.
    assert initial.used_lrs + restored.used_lrs == pytest.approx(
        reference_initial.used_lrs + reference_restored.used_lrs
    )
    torch.testing.assert_close(restored.weight, reference_restored.weight)


def test_plateau_rejects_step_interval():
    model = PlateauModel()
    model.hparams.update(scheduler_interval="step", warmup=2)
    with pytest.raises(AssertionError, match="epoch"):
        model.configure_optimizers()


def test_stage_floor_is_applied_after_warmup_callback(tmp_path):
    model = PlateauModel(warmup=4)
    model.hparams.min_lr = 0.6
    train = DataLoader(TensorDataset(torch.ones(8, 1)), batch_size=1)
    val = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    make_trainer(tmp_path).fit(model, train, val)
    assert model.used_lrs == pytest.approx([0.6] * 8 + [0.75] * 4)
