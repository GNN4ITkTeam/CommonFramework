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

import torch
from acorn.utils.version_utils import get_pyg_data_keys


def build_signal_edges(event, weighting_config, true_edges):
    signal_mask = torch.zeros_like(true_edges[0], dtype=torch.bool)
    for weight_spec in weighting_config:
        if not weight_spec["conditions"]["y"] or weight_spec["weight"] <= 0.0:
            continue
        # Copy weight_spec but remove the y condition
        yless_weight_spec = weight_spec.copy()
        yless_weight_spec["conditions"] = weight_spec["conditions"].copy()
        yless_weight_spec["conditions"].pop("y")
        signal_mask |= get_weight_mask(
            event, true_edges, yless_weight_spec["conditions"]
        )
    return true_edges[:, signal_mask]


def get_weight_mask(event, edges, weight_conditions, true_edges=None, truth_map=None):
    graph_mask = torch.ones_like(edges[0], dtype=torch.bool)

    for condition_key, condition_val in weight_conditions.items():
        assert condition_key in get_pyg_data_keys(
            event
        ), f"Condition key {condition_key} not found in event keys {get_pyg_data_keys(event)}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        graph_mask &= map_value_to_edges(
            event, value_mask, edges, true_edges, truth_map
        )

    return graph_mask


def handle_weighting(
    event,
    weighting_config,
    pred_edges=None,
    truth=None,
    true_edges=None,
    truth_map=None,
):
    """
    Take the specification of the weighting and convert this into float values. The default is:
    - True edges have weight 1.0
    - Negative edges have weight 1.0

    The weighting_config can be used to change this behaviour. For example, we might up-weight target particles - that is edges that pass:
    - y == 1
    - primary == True
    - pt > 1 GeV
    - etc. As desired.

    We can also down-weight (i.e. mask) edges that are true, but not of interest. For example, we might mask:
    - y == 1
    - primary == False
    - pt < 1 GeV
    - etc. As desired.
    """

    if pred_edges is None:
        assert "edge_index" in get_pyg_data_keys(
            event
        ), "If pred_edges is not provided, it must be in the event"
        pred_edges = event.edge_index

    if truth is None:
        assert "y" in get_pyg_data_keys(
            event
        ), "If truth is not provided, it must be in the event"
        truth = event.y

    if true_edges is None:
        assert "track_edges" in get_pyg_data_keys(
            event
        ), "If true_edges is not provided, it must be in the event"
        true_edges = event.track_edges

    if truth_map is None:
        assert "truth_map" in get_pyg_data_keys(
            event
        ), "If truth_map is not provided, it must be in the event"
        truth_map = event.truth_map

    # Set the default value of weights, which is to mask (weight=0), which will be overwritten if specified in the config
    weights = torch.zeros_like(pred_edges[0], dtype=torch.float)
    weights[
        truth == 0
    ] = 1.0  # Default to 1.0 for negative edges - can overwritten in config

    for weight_spec in weighting_config:
        weight_val = weight_spec["weight"]
        weights[
            get_weight_mask(
                event, pred_edges, weight_spec["conditions"], true_edges, truth_map
            )
        ] = weight_val

    return weights


def handle_hard_cuts(event, hard_cuts_config):
    true_track_mask = torch.ones_like(event.truth_map, dtype=torch.bool)

    for condition_key, condition_val in hard_cuts_config.items():
        assert condition_key in get_pyg_data_keys(
            event
        ), f"Condition key {condition_key} not found in event keys {get_pyg_data_keys(event)}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        true_track_mask = true_track_mask * value_mask

    graph_mask = torch.isin(
        event.edge_index, event.track_edges[:, true_track_mask]
    ).all(0)
    remap_from_mask(event, graph_mask)

    for edge_key in ["edge_index", "y", "weight", "scores"]:
        if edge_key in get_pyg_data_keys(event):
            event[edge_key] = event[edge_key][..., graph_mask]

    num_track_edges = event.track_edges.shape[1]
    for track_feature in get_pyg_data_keys(event):
        if isinstance(event[track_feature], torch.Tensor) and (
            event[track_feature].shape[-1] == num_track_edges
        ):
            event[track_feature] = event[track_feature][..., true_track_mask]


def map_value_to_edges(event, value_mask, edges, true_edges=None, truth_map=None):
    """
    Map the value mask to the graph. This is done by testing which dimension the value fits.
    - If it is already equal to the graph size, nothing needs to be done
    - If it is equal to the track edges, it needs to be mapped to the graph edges
    - If it is equal to node list size, it needs to be mapped to the incoming/outgoing graph edges
    """

    if edges is not None and edges.shape[1] in [
        value_mask.shape[0],
        2 * value_mask.shape[0],
    ]:
        # Value mask is already of shape (E) or (E/2)
        # print("Value mask is already of shape (E) or (E/2)")
        return map_edges_to_edges(value_mask, edges)
    elif true_edges is not None and true_edges.shape[1] in [
        value_mask.shape[0],
        2 * value_mask.shape[0],
    ]:
        # Value mask is of shape (T) or (T/2), that is directed or undirected respectively
        # print("Value mask is of shape (T) or (T/2), that is directed or undirected respectively")
        return map_tracks_to_edges(value_mask, edges, truth_map)
    elif value_mask.shape[0] == event.num_nodes:
        # Value mask is of shape (N)
        # print("Value mask is of shape (N)")
        return map_nodes_to_edges(value_mask, edges)
    else:
        # Unsure what shape value mask is, simply "ignore" it by returning a mask of all True
        # print("Unsure what shape value mask is, simply ignore it by returning a mask of all True")
        return torch.ones_like(edges[0], dtype=torch.bool)


def map_edges_to_edges(edges_mask, edges):
    if edges.shape[1] == 2 * edges_mask.shape[0]:
        edges_mask = torch.cat([edges_mask, edges_mask], dim=0)

    return edges_mask


def map_tracks_to_edges(track_mask, edges, truth_map):
    edges_mask = torch.zeros_like(edges[0], dtype=torch.bool, device=edges.device)

    if truth_map.shape[0] == 2 * track_mask.shape[0]:  # Handle undirected case
        track_mask = torch.cat([track_mask, track_mask], dim=0)

    edges_mask[truth_map[truth_map >= 0]] = track_mask[truth_map >= 0]

    return edges_mask


def map_nodes_to_edges(node_mask, edges):
    return node_mask[edges].any(0)


def get_condition_lambda(condition_key, condition_val):
    # Refactor the above switch case into a dictionary
    condition_dict = {
        "is": lambda event: event[condition_key] == condition_val,
        "is_not": lambda event: event[condition_key] != condition_val,
        "in": lambda event: torch.isin(
            event[condition_key],
            torch.tensor(condition_val[1], device=event[condition_key].device),
        ),
        "not_in": lambda event: ~torch.isin(
            event[condition_key],
            torch.tensor(condition_val[1], device=event[condition_key].device),
        ),
        "within": lambda event: (condition_val[0] <= event[condition_key].float())
        & (event[condition_key].float() <= condition_val[1]),
        "not_within": lambda event: not (
            (condition_val[0] <= event[condition_key].float())
            & (event[condition_key].float() <= condition_val[1])
        ),
    }

    if isinstance(condition_val, bool):
        return lambda event: event[condition_key] == condition_val
    elif isinstance(condition_val, list) and not isinstance(condition_val[0], str):
        return lambda event: (condition_val[0] <= event[condition_key].float()) & (
            event[condition_key].float() <= condition_val[1]
        )
    elif isinstance(condition_val, list):
        return condition_dict[condition_val[0]]
    else:
        raise ValueError(f"Condition {condition_val} not recognised")


def remap_from_mask(event, edge_mask):
    """
    Takes a mask applied to the edge_index tensor, and remaps the truth_map tensor indices to match.
    """

    truth_map_to_edges = torch.ones(event.edge_index.shape[1], dtype=torch.long) * -1
    truth_map_to_edges[event.truth_map[event.truth_map >= 0]] = torch.arange(
        event.truth_map.shape[0]
    )[event.truth_map >= 0]
    truth_map_to_edges = truth_map_to_edges[edge_mask]

    new_map = torch.ones(event.truth_map.shape[0], dtype=torch.long) * -1
    new_map[truth_map_to_edges[truth_map_to_edges >= 0]] = torch.arange(
        truth_map_to_edges.shape[0]
    )[truth_map_to_edges >= 0]
    event.truth_map = new_map.to(event.truth_map.device)
