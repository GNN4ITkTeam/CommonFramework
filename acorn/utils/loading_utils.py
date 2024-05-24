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

import os
from typing import List
import warnings
import torch
import logging
from torch_geometric.data import Data
from pathlib import Path

from .mapping_utils import (
    get_condition_lambda,
    map_tensor_handler,
    remap_from_mask,
    get_variable_type,
)


def load_datafiles_in_dir(input_dir, data_name=None, data_num=None):
    if data_name is not None:
        input_dir = os.path.join(input_dir, data_name)

    data_files = [str(path) for path in Path(input_dir).rglob("*.pyg")][:data_num]
    if len(data_files) == 0:
        warnings.warn(f"No data files found in {input_dir}")
    if data_num is not None:
        assert len(data_files) == data_num, (
            f"Number of data files found ({len(data_files)}) is less than the number"
            f" requested ({data_num})"
        )

    return data_files


def load_dataset_from_dir(input_dir, data_name, data_num):
    """
    Load in the PyG Data dataset from the data directory.
    """
    data_files = load_datafiles_in_dir(input_dir, data_name, data_num)

    return [torch.load(f, map_location="cpu") for f in data_files]


def run_data_tests(datasets: List, required_features, optional_features):
    for dataset in datasets:
        sample_event = dataset[0]
        assert sample_event is not None, "No data loaded"
        # Check that the event is the latest PyG format
        try:
            _ = len(sample_event)
        except RuntimeError:
            warnings.warn(
                "Data is not in the latest PyG format, so will be converted on-the-fly."
                " Consider re-saving the data in latest PyG Data type."
            )
            if dataset is not None:
                for i, event in enumerate(dataset):
                    dataset[i] = convert_to_latest_pyg_format(event)

        for feature in required_features:
            assert feature in sample_event or f"x_{feature}" in sample_event, (
                f"Feature [{feature}] not found in data, this is REQUIRED. Features"
                f" found: {sample_event.keys}"
            )

        missing_optional_features = [
            feature
            for feature in optional_features
            if feature not in sample_event or f"x_{feature}" not in sample_event
        ]
        for feature in missing_optional_features:
            warnings.warn(f"OPTIONAL feature [{feature}] not found in data")

        # Check that the number of nodes is compatible with the edge indexing
        if "edge_index" in sample_event.keys and "x" in sample_event.keys:
            assert (
                sample_event.x.shape[0] >= sample_event.edge_index.max().item() + 1
            ), (
                "Number of nodes is not compatible with the edge indexing. Possibly an"
                " earlier stage has removed nodes, but not updated the edge indexing."
            )


def convert_to_latest_pyg_format(event):
    """
    Convert the data to the latest PyG format.
    """
    return Data.from_dict(event.__dict__)


def handle_weighting(event, weighting_config):
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

    # Set the default values, which will be overwritten if specified in the config
    weights = torch.zeros_like(event.edge_y, dtype=torch.float)
    weights[event.edge_y == 0] = 1.0

    for weight_spec in weighting_config:
        weight_val = weight_spec["weight"]
        weights[get_weight_mask(event, weight_spec["conditions"])] = weight_val

    return weights


def handle_hard_cuts(event, hard_cuts_config):
    true_track_mask = torch.ones_like(event.track_to_edge_map, dtype=torch.bool)

    for condition_key, condition_val in hard_cuts_config.items():
        assert (
            condition_key in event.keys
        ), f"Condition key {condition_key} not found in event keys {event.keys}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        true_track_mask = true_track_mask * value_mask

    graph_mask = torch.isin(
        event.edge_index, event.track_edges[:, true_track_mask]
    ).all(0)
    remap_from_mask(event, graph_mask)

    for edge_key in event.keys:
        if (
            isinstance(event[edge_key], torch.Tensor)
            and get_variable_type(edge_key) == "edge-like"
        ):
            event[edge_key] = event[edge_key][..., graph_mask]

    for track_feature in event.keys:
        if (
            isinstance(event[track_feature], torch.Tensor)
            and get_variable_type(edge_key) == "track-like"
        ):
            event[track_feature] = event[track_feature][..., true_track_mask]


def handle_hard_node_cuts(
    event, hard_cuts_config, min_nodes=0, max_nodes=None, edges=False, tracks=False
):
    """
    Given set of cut config, remove nodes that do not pass the cuts.
    Remap the track_edges to the new node list.
    """
    node_like_feature = [
        event[feature]
        for feature in event.keys
        if get_variable_type(feature) == "node-like"
    ][0]
    node_mask = torch.ones_like(node_like_feature, dtype=torch.bool)

    # TODO: Refactor this to simply trim the true tracks and check which nodes are in the true tracks
    for condition_key, condition_val in hard_cuts_config.items():
        assert (
            condition_key in event.keys
        ), f"Condition key {condition_key} not found in event keys {event.keys}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        node_val_mask = map_tensor_handler(
            value_mask,
            output_type="node-like",
            input_type=get_variable_type(condition_key),
            track_edges=event.track_edges,
            num_nodes=node_like_feature.shape[0],
            num_track_edges=event.track_edges.shape[1],
        )
        node_mask = node_mask * node_val_mask

    if node_mask.sum() < min_nodes:
        return False
    if max_nodes and node_mask.sum() > max_nodes:
        return False

    logging.info(
        f"Masking the following number of nodes with the HARD CUT: {node_mask.sum()} /"
        f" {node_mask.shape[0]}"
    )

    # TODO: Refactor the below to use the remap_from_mask function
    num_nodes = event.num_nodes
    for feature in event.keys:
        if (
            isinstance(event[feature], torch.Tensor)
            and get_variable_type(feature) == "node-like"
        ):
            event[feature] = event[feature][node_mask]

    num_tracks = event.track_edges.shape[1]
    track_mask = node_mask[event.track_edges].all(0)
    node_lookup = torch.cumsum(node_mask, dim=0) - 1
    for feature in event.keys:
        if (
            isinstance(event[feature], torch.Tensor)
            and get_variable_type(feature) == "track-like"
        ):
            event[feature] = event[feature][..., track_mask]

    event.track_edges = node_lookup[event.track_edges]
    event.num_nodes = node_mask.sum()

    if edges:
        edge_mask = node_mask[event.edge_index]
        edge_mask = edge_mask[0] & edge_mask[1]
        for feature in event.keys:
            if feature == "edge_index":
                event.edge_index = event.edge_index.T[edge_mask].T
                event.edge_index = node_lookup[event.edge_index]
            elif (
                isinstance(event[feature], torch.Tensor)
                and get_variable_type(feature) == "edge-like"
            ):
                event[feature] = event[feature][edge_mask]

    return True


def reset_angle(angles):
    angles[angles > torch.pi] = angles[angles > torch.pi] - 2 * torch.pi
    angles[angles < -torch.pi] = angles[angles < -torch.pi] + 2 * torch.pi
    return angles


def handle_edge_features(event, edge_features):
    src, dst = event.edge_index

    if "edge_dr" in edge_features and not ("edge_dr" in event.keys):
        event.edge_dr = event.hit_r[dst] - event.hit_r[src]
    if "edge_dphi" in edge_features and not ("edge_dphi" in event.keys):
        event.edge_dphi = (
            reset_angle((event.hit_phi[dst] - event.hit_phi[src]) * torch.pi) / torch.pi
        )
    if "edge_dz" in edge_features and not ("edge_dz" in event.keys):
        event.edge_dz = event.hit_z[dst] - event.hit_z[src]
    if "edge_deta" in edge_features and not ("edge_deta" in event.keys):
        event.edge_deta = event.hit_eta[dst] - event.hit_eta[src]
    if "edge_phislope" in edge_features and not ("edge_phislope" in event.keys):
        dr = event.hit_r[dst] - event.hit_r[src]
        dphi = (
            reset_angle((event.hit_phi[dst] - event.hit_phi[src]) * torch.pi) / torch.pi
        )
        phislope = dphi / dr
        event.edge_phislope = phislope
    if "edge_phislope" in edge_features:
        event.edge_phislope = torch.nan_to_num(
            event.edge_phislope, nan=0.0, posinf=100, neginf=-100
        )
        event.edge_phislope = torch.clamp(event.edge_phislope, -100, 100)
    if "edge_rphislope" in edge_features and not ("edge_rphislope" in event.keys):
        r_ = (event.hit_r[dst] + event.hit_r[src]) / 2.0
        dr = event.hit_r[dst] - event.hit_r[src]
        dphi = (
            reset_angle((event.hit_phi[dst] - event.hit_phi[src]) * torch.pi) / torch.pi
        )
        phislope = dphi / dr
        phislope = torch.nan_to_num(phislope, nan=0.0, posinf=100, neginf=-100)
        phislope = torch.clamp(phislope, -100, 100)
        rphislope = torch.multiply(r_, phislope)
        event.edge_rphislope = rphislope  # features / norm / pre_proc once
    if "edge_rphislope" in edge_features:
        event.edge_rphislope = torch.nan_to_num(event.edge_rphislope, nan=0.0)


def get_weight_mask(event, weight_conditions):
    graph_mask = torch.ones_like(event.edge_y)

    for condition_key, condition_val in weight_conditions.items():
        assert (
            condition_key in event.keys
        ), f"Condition key {condition_key} not found in event keys {event.keys} {event.event_id}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        graph_mask = graph_mask * map_tensor_handler(
            value_mask,
            output_type="edge-like",
            input_type=get_variable_type(condition_key),
            num_nodes=event.num_nodes,
            edge_index=event.edge_index,
            truth_map=event.track_to_edge_map,
        )

    return graph_mask
