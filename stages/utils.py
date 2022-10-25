import sys
import os
from typing import List
import warnings

import numpy as np
import scipy as sp
import torch
from torch_geometric.data import Data

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def load_datafiles_in_dir(input_dir, data_name, data_num):

    data_dir = os.path.join(input_dir, "data", data_name)
    if not os.path.exists(data_dir):
        warnings.warn(f"Data directory {data_dir} does not exist. Looking in {input_dir}.")
        data_dir = os.path.join(input_dir, data_name)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")][:data_num]
    assert len(data_files) > 0, f"No data files found in {data_dir}"
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
            sample_event = dataset[0]

        for feature in required_features:
            assert feature in sample_event, f"Feature {feature} not found in data, this is REQUIRED. Features found: {sample_event.keys}"
        
        missing_optional_features = [ feature for feature in optional_features if feature not in sample_event ]
        for feature in missing_optional_features:
            warnings.warn(f"OPTIONAL feature [{feature}] not found in data")

def convert_to_latest_pyg_format(event):
    """
    Convert the data to the latest PyG format.
    """
    return Data.from_dict(event.__dict__)

def construct_event_truth(event, config):
    # event.edge_index, event.y = graph_intersection(event.edge_index, event.truth_graph)
    assert event.y.shape[0] == event.edge_index.shape[1], f"Input graph has {event.edge_index.shape[1]} edges, but {event.y.shape[0]} truth labels"

    if "weighting_config" in config:
        assert isinstance(config["weighting_config"], dict), "Weighting config must be a dictionary"
        event.weights = handle_weighting(event, config["weighting_config"])

    return event

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

    """
    TODO: This now needs to handle the new edge feature system
    """


    weights = torch.zeros_like(event.y)

    for weight_spec in weighting_config:
        weight_val = weight_spec["weight"]
        weights = weights + weight_val * get_weight_mask(event, weight_spec)

    return weights

def get_weight_mask(event, weight_spec):

    graph_mask = torch.ones_like(event.y)

    for condition_key, condition_val in weight_spec["conditions"].items():
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        graph_mask = graph_mask * map_value_to_graph(event, value_mask)

    return mask

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
    graph_mask[event.truth_map] = track_mask

    return graph_mask

def map_nodes_to_graph(event, node_mask):

    return node_mask[event.edge_index[0]]

def get_condition_lambda(condition_key, condition_val):

    # Refactor the above switch case into a dictionary
    condition_dict = {
        "is": lambda event: event[condition_key] == condition_val,
        "is_not": lambda event: event[condition_key] != condition_val,
        "in": lambda event: event[condition_key] in condition_val,
        "not_in": lambda event: event[condition_key] not in condition_val,
        "within": lambda event: condition_val[0] <= event[condition_key] <= condition_val[1],
        "not_within": lambda event: not (condition_val[0] <= event[condition_key] <= condition_val[1]),
    }

    if isinstance(condition_val, bool):
        return lambda event: event[condition_key] == condition_val
    elif isinstance(condition_val, list):
        return lambda event: condition_val <= event[condition_key] <= condition_val
    elif isinstance(condition_val, tuple):
        return condition_dict[condition_val[0]]
    else:
        raise ValueError(f"Condition {condition_val} not recognised")

