import torch.nn as nn

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