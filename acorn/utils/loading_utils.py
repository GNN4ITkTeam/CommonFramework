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
from typing import List, Union
import warnings
import torch
import logging
from torch_geometric.data import Data as PygData
from pathlib import Path
import pandas as pd

from .mapping_utils import (
    get_condition_lambda,
    map_tensor_handler,
    remap_from_mask,
    get_variable_type,
    VariableType,
)
from .version_utils import get_pyg_data_keys


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
        if dataset is None or len(dataset) == 0:
            warnings.warn(
                "Found an empty dataset. Please check if this is not intended."
            )
            continue
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
                f" found: {get_pyg_data_keys(sample_event)}"
            )

        missing_optional_features = [
            feature for feature in optional_features if feature not in sample_event
        ]
        for feature in missing_optional_features:
            warnings.warn(f"OPTIONAL feature [{feature}] not found in data")

        # Check that the number of nodes is compatible with the edge indexing
        if "edge_index" in get_pyg_data_keys(sample_event) and "x" in get_pyg_data_keys(
            sample_event
        ):
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
    return PygData.from_dict(event.__dict__)


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


def handle_hard_node_cuts(
    event: Union[PygData, pd.DataFrame], hard_cuts_config: dict, passing_hit_ids=None
):
    """
    Given set of cut config, remove nodes that do not pass the cuts.
    Remap the track_edges to the new node list.
    """
    if isinstance(event, PygData):
        return handle_hard_node_cuts_pyg(event, hard_cuts_config)
    elif isinstance(event, pd.DataFrame):
        return handle_hard_node_cuts_pandas(event, hard_cuts_config, passing_hit_ids)
    else:
        raise ValueError(f"Data type {type(event)} not recognised.")


def handle_hard_node_cuts_pyg(event: PygData, hard_cuts_config: dict):
    """
    Given set of cut config, remove nodes that do not pass the cuts.
    Remap the track_edges to the new node list.
    """
    node_like_feature = [
        event[feature]
        for feature in get_pyg_data_keys(event)
        if get_variable_type(feature) == VariableType.NODE_LIKE
    ][0]
    node_mask = torch.ones_like(node_like_feature, dtype=torch.bool)

    # TODO: Refactor this to simply trim the true tracks and check which nodes are in the true tracks
    for condition_key, condition_val in hard_cuts_config.items():
        assert condition_key in get_pyg_data_keys(
            event
        ), f"Condition key {condition_key} not found in event keys {get_pyg_data_keys(event)}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        node_val_mask = map_tensor_handler(
            value_mask,
            output_type=VariableType.NODE_LIKE,
            input_type=get_variable_type(condition_key),
            track_edges=event.track_edges,
            num_nodes=node_like_feature.shape[0],
            num_track_edges=event.track_edges.shape[1],
        )
        node_mask = node_mask * node_val_mask

    logging.info(
        f"Masking the following number of nodes with the HARD CUT: {node_mask.sum()} /"
        f" {node_mask.shape[0]}"
    )

    num_nodes = event.num_nodes
    for feature in get_pyg_data_keys(event):
        if (
            isinstance(event[feature], torch.Tensor)
            and get_variable_type(feature) == VariableType.NODE_LIKE
        ):
            event[feature] = event[feature][node_mask]

    num_tracks = event.track_edges.shape[1]
    track_mask = node_mask[event.track_edges].all(0)
    node_lookup = torch.cumsum(node_mask, dim=0) - 1
    for feature in get_pyg_data_keys(event):
        if (
            isinstance(event[feature], torch.Tensor)
            and get_variable_type(feature) == VariableType.TRACK_LIKE
        ):
            event[feature] = event[feature][..., track_mask]

    event.track_edges = node_lookup[event.track_edges]
    event.num_nodes = node_mask.sum()

    return event


def handle_hard_node_cuts_pandas(
    event: pd.DataFrame, hard_cuts_config: dict, passing_hit_ids=None
):
    """
    Given set of cut config, remove nodes that do not pass the cuts.
    If passing_hit_ids is provided, only the hits with the given ids will be kept.
    """

    if passing_hit_ids is not None:
        event = event[event.hit_id.isin(passing_hit_ids)]
        return event

    # Create a temporary DataFrame with the same structure as the PyTorch event
    temp_event = {
        col: torch.tensor(event[col].values)
        for col in event.columns
        if pd.api.types.is_numeric_dtype(event[col])
        or pd.api.types.is_bool_dtype(event[col])
    }

    # Initialize the mask with all True values
    value_mask = torch.ones(len(event), dtype=torch.bool)

    for condition_key, condition_val in hard_cuts_config.items():
        assert (
            condition_key in event.columns
        ), f"Condition key {condition_key} not found in event keys {event.columns}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)

        # Apply the condition lambda to the temporary event and update the mask
        value_mask &= condition_lambda(temp_event)

    # Convert the mask back to a Pandas-compatible format
    value_mask = value_mask.numpy().astype(bool)

    event = event[value_mask]

    return event


def handle_hard_cuts(event: PygData, hard_cuts_config: dict):
    """
    Given set of cut config, remove edges that do not pass the cuts.
    """
    true_track_mask = torch.ones_like(event.track_to_edge_map, dtype=torch.bool)

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

    for edge_key in get_pyg_data_keys(event):
        if (
            isinstance(event[edge_key], torch.Tensor)
            and get_variable_type(edge_key) == VariableType.EDGE_LIKE
        ):
            event[edge_key] = event[edge_key][..., graph_mask]

    for track_feature in get_pyg_data_keys(event):
        if (
            isinstance(event[track_feature], torch.Tensor)
            and get_variable_type(track_feature) == VariableType.TRACK_LIKE
        ):
            event[track_feature] = event[track_feature][..., true_track_mask]


def reset_angle(angles):
    angles[angles > torch.pi] = angles[angles > torch.pi] - 2 * torch.pi
    angles[angles < -torch.pi] = angles[angles < -torch.pi] + 2 * torch.pi
    return angles


def handle_edge_features(event, edge_features):
    src, dst = event.edge_index

    if "edge_dr" in edge_features and not ("edge_dr" in get_pyg_data_keys(event)):
        event.edge_dr = event.hit_r[dst] - event.hit_r[src]
    if "edge_dphi" in edge_features and not ("edge_dphi" in get_pyg_data_keys(event)):
        event.edge_dphi = (
            reset_angle((event.hit_phi[dst] - event.hit_phi[src]) * torch.pi) / torch.pi
        )
    if "edge_dz" in edge_features and not ("edge_dz" in get_pyg_data_keys(event)):
        event.edge_dz = event.hit_z[dst] - event.hit_z[src]
    if "edge_deta" in edge_features and not ("edge_deta" in get_pyg_data_keys(event)):
        event.edge_deta = event.hit_eta[dst] - event.hit_eta[src]
    if "edge_phislope" in edge_features and not (
        "edge_phislope" in get_pyg_data_keys(event)
    ):
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
    if "edge_rphislope" in edge_features and not (
        "edge_rphislope" in get_pyg_data_keys(event)
    ):
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
        assert condition_key in get_pyg_data_keys(
            event
        ), f"Condition key {condition_key} not found in event keys {get_pyg_data_keys(event)}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        graph_mask = graph_mask * map_tensor_handler(
            value_mask,
            output_type=VariableType.EDGE_LIKE,
            input_type=get_variable_type(condition_key),
            num_nodes=event.num_nodes,
            edge_index=event.edge_index,
            truth_map=event.track_to_edge_map,
        )

    return graph_mask


variable_name_prefix_map = {
    "x": "hit_x",
    "z": "hit_z",
    "r": "hit_r",
    "phi": "hit_phi",
    "eta": "hit_eta",
    "region": "hit_region",
    "module_id": "hit_module_id",
    "cluster_x_1": "hit_cluster_x_1",
    "cluster_y_1": "hit_cluster_y_1",
    "cluster_z_1": "hit_cluster_z_1",
    "cluster_x_2": "hit_cluster_x_2",
    "cluster_y_2": "hit_cluster_y_2",
    "cluster_z_2": "hit_cluster_z_2",
    "cluster_r_1": "hit_cluster_r_1",
    "cluster_phi_1": "hit_cluster_phi_1",
    "cluster_eta_1": "hit_cluster_eta_1",
    "cluster_r_2": "hit_cluster_r_2",
    "cluster_phi_2": "hit_cluster_phi_2",
    "cluster_eta_2": "hit_cluster_eta_2",
    "norm_x_1": "hit_norm_x_1",
    "norm_y_1": "hit_norm_y_1",
    "norm_x_2": "hit_norm_x_2",
    "norm_y_2": "hit_norm_y_2",
    "norm_z_1": "hit_norm_z_1",
    "eta_angle_1": "hit_eta_angle_1",
    "phi_angle_1": "hit_phi_angle_1",
    "eta_angle_2": "hit_eta_angle_2",
    "phi_angle_2": "hit_phi_angle_2",
    "norm_z_2": "hit_norm_z_2",
    "count_1": "hit_count_1",
    "charge_count_1": "hit_charge_count_1",
    "loc_eta_1": "hit_loc_eta_1",
    "loc_phi_1": "hit_loc_phi_1",
    "localDir0_1": "hit_localDir0_1",
    "localDir1_1": "hit_localDir1_1",
    "localDir2_1": "hit_localDir2_1",
    "lengthDir0_1": "hit_lengthDir0_1",
    "lengthDir1_1": "hit_lengthDir1_1",
    "lengthDir2_1": "hit_lengthDir2_1",
    "glob_eta_1": "hit_glob_eta_1",
    "glob_phi_1": "hit_glob_phi_1",
    "count_2": "hit_count_2",
    "charge_count_2": "hit_charge_count_2",
    "loc_eta_2": "hit_loc_eta_2",
    "loc_phi_2": "hit_loc_phi_2",
    "localDir0_2": "hit_localDir0_2",
    "localDir1_2": "hit_localDir1_2",
    "localDir2_2": "hit_localDir2_2",
    "lengthDir0_2": "hit_lengthDir0_2",
    "lengthDir1_2": "hit_lengthDir1_2",
    "lengthDir2_2": "hit_lengthDir2_2",
    "glob_eta_2": "hit_glob_eta_2",
    "glob_phi_2": "hit_glob_phi_2",
    "volume_id": "hit_volume_id",
    "layer_id": "hit_layer_id",
    "module_id": "hit_module_id",
    "module_index": "hit_module_index",
    "weight": "hit_weight",
    "cell_count": "hit_cell_count",
    "cell_val": "hit_cell_val",
    "leta": "hit_leta",
    "lphi": "hit_lphi",
    "lx": "hit_lx",
    "ly": "hit_ly",
    "lz": "hit_lz",
    "geta": "hit_geta",
    "gphi": "hit_gphi",
    "particle_id": "track_particle_id",
    "pt": "track_particle_pt",
    "radius": "track_particle_radius",
    "primary": "track_particle_primary",
    "nhits": "track_particle_nhits",
    "pdgId": "track_particle_pdgId",
    "eta_particle": "track_particle_eta",
    "redundant_split_edges": "track_redundant_split_edges",
    "y": "edge_y",
    "truth_map": "track_to_edge_map",
    "dr": "edge_dr",
    "dphi": "edge_dphi",
    "dz": "edge_dz",
    "deta": "edge_deta",
    "phislope": "edge_phislope",
    "rphislope": "edge_rphislope",
    "scores": "edge_scores",
    "labels": "edge_track_labels",
}


def add_variable_name_prefix_in_pyg(graph):
    for key in get_pyg_data_keys(graph):
        if key in variable_name_prefix_map:
            graph[variable_name_prefix_map[key]] = graph.pop(key)
    return graph


def remove_variable_name_prefix_in_pyg(graph):
    reverse_variable_name_prefix_map = {
        v: k for k, v in variable_name_prefix_map.items()
    }
    for key in get_pyg_data_keys(graph):
        if key in reverse_variable_name_prefix_map:
            graph[reverse_variable_name_prefix_map[key]] = graph.pop(key)
    return graph


def add_variable_name_prefix_in_config(config):
    if config.get("feature_sets"):
        if "hit_features" in config["feature_sets"]:
            for i in range(len(config["feature_sets"]["hit_features"])):
                if (
                    config["feature_sets"]["hit_features"][i]
                    in variable_name_prefix_map
                ):
                    config["feature_sets"]["hit_features"][
                        i
                    ] = variable_name_prefix_map[
                        config["feature_sets"]["hit_features"][i]
                    ]
        if "track_features" in config["feature_sets"]:
            for i in range(len(config["feature_sets"]["track_features"])):
                if (
                    config["feature_sets"]["track_features"][i]
                    in variable_name_prefix_map
                ):
                    config["feature_sets"]["track_features"][
                        i
                    ] = variable_name_prefix_map[
                        config["feature_sets"]["track_features"][i]
                    ]
    if config.get("node_features"):
        for i in range(len(config["node_features"])):
            if config["node_features"][i] in variable_name_prefix_map:
                config["node_features"][i] = variable_name_prefix_map[
                    config["node_features"][i]
                ]
    if config.get("edge_features"):
        for i in range(len(config["edge_features"])):
            if config["edge_features"][i] in variable_name_prefix_map:
                config["edge_features"][i] = variable_name_prefix_map[
                    config["edge_features"][i]
                ]
    if config.get("weighting"):
        for i in range(len(config["weighting"])):
            for key in list(config["weighting"][i]["conditions"].keys()):
                if key in variable_name_prefix_map:
                    config["weighting"][i]["conditions"][
                        variable_name_prefix_map[key]
                    ] = config["weighting"][i]["conditions"].pop(key)
    if config.get("target_tracks"):
        for key in list(config["target_tracks"].keys()):
            if key in variable_name_prefix_map:
                config["target_tracks"][variable_name_prefix_map[key]] = config[
                    "target_tracks"
                ].pop(key)
    if config.get("hard_cuts"):
        for key in list(config["hard_cuts"].keys()):
            if key in variable_name_prefix_map:
                config["hard_cuts"][variable_name_prefix_map[key]] = config[
                    "hard_cuts"
                ].pop(key)
    return config


def infer_num_nodes(graph):
    """
    Ensure the num_nodes is set properly
    """

    if "num_nodes" not in get_pyg_data_keys(graph) or graph.num_nodes is None:
        assert "hit_id" in get_pyg_data_keys(graph), "No node features found in graph"
        graph.num_nodes = graph.hit_id.shape[0]
