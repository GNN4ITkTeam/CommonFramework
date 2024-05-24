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


def get_optimizers(parameters, hparams):
    """Get the optimizer and scheduler."""
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

    if (
        "scheduler" not in hparams
        or hparams["scheduler"] is None
        or hparams["scheduler"] == "StepLR"
    ):
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=hparams["patience"],
                    gamma=hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    elif hparams["scheduler"] == "ReduceLROnPlateau":
        metric_mode = hparams.get("metric_mode", "min")
        metric_to_monitor = hparams.get("metric_to_monitor", "val_loss")
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer[0],
                    mode=metric_mode,
                    factor=hparams["factor"],
                    patience=hparams["patience"],
                    verbose=True,
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": metric_to_monitor,
            }
        ]
    elif hparams["scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer[0],
                    T_0=hparams["patience"],
                    T_mult=2,
                    eta_min=1e-8,
                    last_epoch=-1,
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    else:
        raise ValueError(f"Unknown scheduler: {hparams['scheduler']}")

    return optimizer, scheduler
