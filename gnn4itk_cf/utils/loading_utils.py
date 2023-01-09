import os
from typing import List
import warnings
import torch
from torch_geometric.data import Data
from pathlib import Path

def load_datafiles_in_dir(input_dir, data_name = None, data_num = None):

    if data_name is not None:
        input_dir = os.path.join(input_dir, data_name)

    data_files = [str(path) for path in Path(input_dir).rglob("*.pyg")][:data_num]
    assert len(data_files) > 0, f"No data files found in {input_dir}"
    if data_num is not None:
        assert len(data_files) == data_num, f"Number of data files found ({len(data_files)}) is less than the number requested ({data_num})"

    return data_files

def load_dataset_from_dir(input_dir, data_name, data_num):
    """
    Load in the PyG Data dataset from the data directory.
    """
    data_files = load_datafiles_in_dir(input_dir, data_name, data_num)
    
    return [ torch.load(f, map_location="cpu") for f in data_files ]

def run_data_tests(datasets: List, required_features, optional_features):
    
    for dataset in datasets:
        sample_event = dataset[0]
        assert sample_event is not None, "No data loaded"
        # Check that the event is the latest PyG format
        try:
            _ = len(sample_event)
        except RuntimeError:
            warnings.warn("Data is not in the latest PyG format, so will be converted on-the-fly. Consider re-saving the data in latest PyG Data type.")
            if dataset is not None:
                for i, event in enumerate(dataset):
                    dataset[i] = convert_to_latest_pyg_format(event)

        for feature in required_features:
            assert feature in sample_event or f"x_{feature}" in sample_event, f"Feature [{feature}] not found in data, this is REQUIRED. Features found: {sample_event.keys}"

        missing_optional_features = [ feature for feature in optional_features if feature not in sample_event or f"x_{feature}" not in sample_event ]
        for feature in missing_optional_features:
            warnings.warn(f"OPTIONAL feature [{feature}] not found in data")

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
    weights = torch.zeros_like(event.y, dtype=torch.float)
    weights[event.y == 0] = 1.0

    for weight_spec in weighting_config:
        weight_val = weight_spec["weight"]
        weights[get_weight_mask(event, weight_spec["conditions"])] = weight_val

    event.weights = weights

def handle_hard_cuts(event, hard_cuts_config):

    true_track_mask = torch.ones_like(event.truth_map, dtype=torch.bool)

    for condition_key, condition_val in hard_cuts_config.items():
        assert condition_key in event.keys, f"Condition key {condition_key} not found in event keys {event.keys}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        true_track_mask = true_track_mask * value_mask

    graph_mask = torch.isin(event.edge_index, event.track_edges[:, true_track_mask]).all(0)
    remap_from_mask(event, graph_mask)

    for edge_key in ["edge_index", "y", "weight", "scores"]:
        if edge_key in event.keys:
            event[edge_key] = event[edge_key][..., graph_mask]

    for track_feature in event.keys:
        if isinstance(event[track_feature], torch.Tensor) and (event[track_feature].ndim == 1) and (event[track_feature].shape[0] == event.track_edges.shape[1]) and track_feature != "track_edges":
            event[track_feature] = event[track_feature][true_track_mask]
    
    event.track_edges = event.track_edges[:, true_track_mask]

def get_weight_mask(event, weight_conditions):

    graph_mask = torch.ones_like(event.y)

    for condition_key, condition_val in weight_conditions.items():
        assert condition_key in event.keys, f"Condition key {condition_key} not found in event keys {event.keys}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        graph_mask = graph_mask * map_value_to_graph(event, value_mask)

    return graph_mask

def map_value_to_graph(event, value_mask):
    """
    Map the value mask to the graph. This is done by testing which dimension the value fits. 
    - If it is already equal to the graph size, nothing needs to be done
    - If it is equal to the track edges, it needs to be mapped to the graph edges
    - If it is equal to node list size, it needs to be mapped to the incoming/outgoing graph edges 
    """

    if value_mask.shape[0] == event.y.shape[0]:
        return value_mask
    elif value_mask.shape[0] == event.track_edges.shape[1]:
        return map_tracks_to_graph(event, value_mask)
    elif value_mask.shape[0] == event.x.shape[0]:
        return map_nodes_to_graph(event, value_mask)
    else:
        raise ValueError(f"Value mask has shape {value_mask.shape}, which is not compatible with the graph")

def map_tracks_to_graph(event, track_mask):

    graph_mask = torch.zeros_like(event.y)
    graph_mask[event.truth_map[event.truth_map >= 0]] = track_mask[event.truth_map >= 0]

    return graph_mask

def map_nodes_to_graph(event, node_mask):

    return node_mask[event.edge_index[0]]

def get_condition_lambda(condition_key, condition_val):

    # Refactor the above switch case into a dictionary
    condition_dict = {
        "is": lambda event: event[condition_key] == condition_val,
        "is_not": lambda event: event[condition_key] != condition_val,
        "in": lambda event: torch.isin(event[condition_key], torch.tensor(condition_val[1])),
        "not_in": lambda event: ~torch.isin(event[condition_key], torch.tensor(condition_val[1])),
        "within": lambda event: (condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1]),
        "not_within": lambda event: not ((condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1])),
    }

    if isinstance(condition_val, bool):
        return lambda event: event[condition_key] == condition_val
    elif isinstance(condition_val, list) and not isinstance(condition_val[0], str):
        return lambda event: (condition_val[0] <= event[condition_key].float()) & (event[condition_key].float() <= condition_val[1])
    elif isinstance(condition_val, list):
        return condition_dict[condition_val[0]]
    else:
        raise ValueError(f"Condition {condition_val} not recognised")

def remap_from_mask(event, edge_mask):
    """ 
    Takes a mask applied to the edge_index tensor, and remaps the truth_map tensor indices to match.
    """

    truth_map_to_edges = torch.ones(event.edge_index.shape[1], dtype=torch.long) * -1
    truth_map_to_edges[event.truth_map[event.truth_map >= 0]] = torch.arange(event.truth_map.shape[0])[event.truth_map >= 0]
    truth_map_to_edges = truth_map_to_edges[edge_mask]

    new_map = torch.ones(event.truth_map.shape[0], dtype=torch.long) * -1
    new_map[truth_map_to_edges[truth_map_to_edges >= 0]] = torch.arange(truth_map_to_edges.shape[0])[truth_map_to_edges >= 0]
    event.truth_map = new_map.to(event.truth_map.device)
