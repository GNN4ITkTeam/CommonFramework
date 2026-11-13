# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch
from pytorch_lightning import Callback


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation=None,
    layer_norm=False,  # TODO : change name to hidden_layer_norm while ensuring backward compatibility
    output_layer_norm=False,
    batch_norm=False,  # TODO : change name to hidden_batch_norm while ensuring backward compatibility
    output_batch_norm=False,
    input_dropout=0,
    hidden_dropout=0,
    track_running_stats=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        if i == 0 and input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:  # hidden_layer_norm
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:  # hidden_batch_norm
            layers.append(
                nn.BatchNorm1d(
                    sizes[i + 1],
                    eps=6e-05,
                    track_running_stats=track_running_stats,
                    affine=True,
                )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
            )
        layers.append(hidden_activation())
        if hidden_dropout > 0:
            layers.append(nn.Dropout(hidden_dropout))
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if output_layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if output_batch_norm:
            layers.append(
                nn.BatchNorm1d(
                    sizes[-1],
                    eps=6e-05,
                    track_running_stats=track_running_stats,
                    affine=True,
                )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
            )
        layers.append(output_activation())
    return nn.Sequential(*layers)


def get_warmup_scale(index, warmup):
    """Linear LR multiplier for a zero-based epoch or optimizer-update index."""
    return min(1.0, (index + 1) / warmup) if warmup else 1.0


class PlateauWarmupCallback(Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        config = pl_module.hparams
        warmup = config.get("warmup")
        if (
            config.get("scheduler") == "ReduceLROnPlateau"
            and warmup
            and trainer.current_epoch < warmup
        ):
            lr = config["lr"] * get_warmup_scale(trainer.current_epoch, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr


def _validate_scheduler_config(hparams):
    """Validate configuration and return the scheduler name, interval and warmup."""
    interval = hparams.get("scheduler_interval", "epoch")
    assert interval in {"epoch", "step"}, "scheduler_interval must be 'epoch' or 'step'"
    assert (
        "warmup_steps" not in hparams
    ), "Use warmup with scheduler_interval: step instead of warmup_steps"
    if interval == "step":
        assert hparams["lr"] > 0, "Step scheduling requires lr > 0"
    warmup = hparams.get("warmup") or 0
    assert (
        isinstance(warmup, int) and warmup >= 0
    ), "warmup must be a nonnegative integer"
    name = hparams.get("scheduler") or "StepLR"
    assert (
        name != "ReduceLROnPlateau" or interval == "epoch"
    ), "ReduceLROnPlateau requires scheduler_interval: epoch"
    assert isinstance(hparams["patience"], int) and hparams["patience"] >= (
        0 if name == "ReduceLROnPlateau" else 1
    ), "Invalid scheduler patience"
    if name not in {"StepLR", "ReduceLROnPlateau", "CosineAnnealingWarmRestarts"}:
        raise ValueError(f"Unknown scheduler: {name}")
    if warmup:
        assert hparams["lr"] > 0, "Warmup requires lr > 0"
        assert (
            0 <= hparams.get("min_lr", 0) <= hparams["lr"]
        ), "Warmup requires 0 <= min_lr <= lr"
    return name, interval, warmup


def get_optimizers(parameters, hparams):
    """Get the optimizer and scheduler."""
    name, interval, warmup = _validate_scheduler_config(hparams)
    weight_decay = hparams.get("lr_weight_decay", 0.01)
    optimizer = [
        torch.optim.AdamW(
            parameters,
            lr=(hparams["lr"]),
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True,
            weight_decay=weight_decay,
        )
    ]

    if name == "ReduceLROnPlateau":
        return optimizer, [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer[0],
                    mode=hparams.get("metric_mode", "min"),
                    factor=hparams["factor"],
                    patience=hparams["patience"],
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": hparams.get("metric_to_monitor", "val_loss"),
            }
        ]

    phases = []
    if warmup:
        phases.append(
            torch.optim.lr_scheduler.LinearLR(
                optimizer[0],
                start_factor=max(
                    1.0 / warmup, hparams.get("min_lr", 0) / hparams["lr"]
                ),
                end_factor=1.0,
                total_iters=max(1, warmup - 1),
            )
        )
    if name == "StepLR":
        phases.append(
            torch.optim.lr_scheduler.StepLR(
                optimizer[0], step_size=hparams["patience"], gamma=hparams["factor"]
            )
        )
    elif name == "CosineAnnealingWarmRestarts":
        phases.append(
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer[0],
                T_0=hparams["patience"],
                T_mult=2,
                eta_min=hparams.get("min_lr", 1e-8),
            )
        )
    schedule = (
        torch.optim.lr_scheduler.SequentialLR(optimizer[0], phases, milestones=[warmup])
        if len(phases) > 1
        else phases[0]
    )
    config = {"scheduler": schedule, "interval": interval, "frequency": 1}
    scheduler = [config]

    return optimizer, scheduler
