import os, sys
import logging
import random

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
try:
    import cupy as cp
except:
    pass

from tqdm import tqdm

from torch_geometric.data import Dataset
from torch_geometric.nn import radius

# ---------------------------- Dataset Processing -------------------------


# ---------------------------- Edge Building ------------------------------

def build_edges(
    query, database, indices=None, r_max=1.0, k_max=10, return_indices=False, backend="FRNN"
):

    if backend == "FRNN":
        dists, idxs, nn, grid = frnn.frnn_grid_points(
            points1=query.unsqueeze(0),
            points2=database.unsqueeze(0),
            lengths1=None,
            lengths2=None,
            K=k_max,
            r=r_max,
            grid=None,
            return_nn=False,
            return_sorted=True,
        )      
        
        idxs = idxs.squeeze().int()
        ind = torch.Tensor.repeat(
        torch.arange(idxs.shape[0], device=device), (idxs.shape[1], 1), 1
        ).T.int()
        positive_idxs = idxs >= 0
        edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()

    elif backend == "PYG":
        edge_list = radius(database, query, r=r_max, max_num_neighbors=k_max)
        
    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    if return_indices:
        return edge_list, dists, idxs, ind
    else:
        return edge_list



# ------------------------- Convenience Utilities ---------------------------


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
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
