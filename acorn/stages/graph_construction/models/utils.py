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

import logging
from typing import Optional, Tuple, Union


import torch.nn as nn
import torch
from torch_geometric.nn import radius

try:
    import frnn

    FRNN_AVAILABLE = True
    logging.warning("FRNN is available")
except ImportError:
    FRNN_AVAILABLE = False
    logging.warning(
        "FRNN is not available, install it at https://github.com/murnanedaniel/FRNN."
        " Using PyG radius instead."
    )
if not torch.cuda.is_available():
    FRNN_AVAILABLE = False
    logging.warning("FRNN is not available, as no GPU is available")


# ---------------------------- Dataset Processing -------------------------

# ---------------------------- Edge Building ------------------------------


def build_edges(
    query: torch.Tensor,
    database: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    r_max: float = 1.0,
    k_max: int = 10,
    return_indices: bool = False,
    backend: str = "FRNN",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Type hint
    if backend == "FRNN" and FRNN_AVAILABLE:
        # Compute edges
        dists, idxs, _, _ = frnn.frnn_grid_points(
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

        idxs: torch.Tensor = idxs.squeeze().int()
        ind = (
            torch.arange(idxs.shape[0], device=query.device)
            .repeat(idxs.shape[1], 1)
            .T.int()
        )
        positive_idxs = idxs >= 0
        edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()
    else:
        edge_list = radius(database, query, r=r_max, max_num_neighbors=k_max)

    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return (
        (edge_list, dists, idxs, ind)
        if (return_indices and backend == "FRNN")
        else edge_list
    )


def graph_intersection(
    input_pred_graph,
    input_truth_graph,
    return_y_pred=True,
    return_y_truth=False,
    return_pred_to_truth=False,
    return_truth_to_pred=False,
    unique_pred=True,
    unique_truth=True,
):
    """
    An updated version of the graph intersection function, which is around 25x faster than the
    Scipy implementation (on GPU). Takes a prediction graph and a truth graph, assumed to have unique entries.
    If unique_pred or unique_truth is False, the function will first find the unique entries in the input graphs, and return the updated edge lists.
    """

    if not unique_pred:
        input_pred_graph = torch.unique(input_pred_graph, dim=1)
    if not unique_truth:
        input_truth_graph = torch.unique(input_truth_graph, dim=1)

    unique_edges, inverse = torch.unique(
        torch.cat([input_pred_graph, input_truth_graph], dim=1),
        dim=1,
        sorted=False,
        return_inverse=True,
        return_counts=False,
    )

    inverse_pred_map = torch.ones_like(unique_edges[1]) * -1
    inverse_pred_map[inverse[: input_pred_graph.shape[1]]] = torch.arange(
        input_pred_graph.shape[1], device=input_pred_graph.device
    )

    inverse_truth_map = torch.ones_like(unique_edges[1]) * -1
    inverse_truth_map[inverse[input_pred_graph.shape[1] :]] = torch.arange(
        input_truth_graph.shape[1], device=input_truth_graph.device
    )

    pred_to_truth = inverse_truth_map[inverse][: input_pred_graph.shape[1]]
    truth_to_pred = inverse_pred_map[inverse][input_pred_graph.shape[1] :]

    return_tensors = []

    if not unique_pred:
        return_tensors.append(input_pred_graph)
    if not unique_truth:
        return_tensors.append(input_truth_graph)
    if return_y_pred:
        y_pred = pred_to_truth >= 0
        return_tensors.append(y_pred)
    if return_y_truth:
        y_truth = truth_to_pred >= 0
        return_tensors.append(y_truth)
    if return_pred_to_truth:
        return_tensors.append(pred_to_truth)
    if return_truth_to_pred:
        return_tensors.append(truth_to_pred)

    return return_tensors if len(return_tensors) > 1 else return_tensors[0]


# ------------------------- Convenience Utilities ---------------------------


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
            layers.append(
                nn.BatchNorm1d(sizes[i + 1], track_running_stats=False, affine=False)
            )
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if batch_norm:
            layers.append(
                nn.BatchNorm1d(sizes[-1], track_running_stats=False, affine=False)
            )
        layers.append(output_activation())
    return nn.Sequential(*layers)
