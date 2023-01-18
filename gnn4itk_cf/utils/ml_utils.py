import torch.nn as nn
import torch

def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation=None,
    layer_norm=False,
    batch_norm=False,
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
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1], track_running_stats=False, affine=False))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1], track_running_stats=False, affine=False))
        layers.append(output_activation())
    return nn.Sequential(*layers)

def get_optimizers(parameters, hparams):
    optimizer = [
        torch.optim.AdamW(
            parameters,
            lr=(hparams["lr"]),
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True,
        )
    ]

    if "scheduler" not in hparams or hparams["scheduler"] is None or hparams["scheduler"] == "StepLR":
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
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer[0],
                    mode="min",
                    factor=hparams["factor"],
                    patience=hparams["patience"],
                    verbose=True,
                ),
                "interval": "epoch",
                "frequency": 1,
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