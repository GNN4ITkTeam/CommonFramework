import sys
import os
import warnings

import numpy as np
import scipy as sp
import torch
from torch_geometric.data import Data

from .models import *   

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

def run_data_tests(trainset, valset, testset, required_features, optional_features):
    
    sample_event = trainset[0]
    assert sample_event is not None, "No data loaded"
    # Check that the event is the latest PyG format
    try:
        _ = len(sample_event)
    except RuntimeError:
        warnings.warn("Data is not in the latest PyG format, so will be converted on-the-fly. Consider re-saving the data in latest PyG Data type.")
        for dataset in [trainset, valset, testset]:
            if dataset is not None:
                for i, event in enumerate(dataset):
                    dataset[i] = convert_to_latest_pyg_format(event)
        sample_event = trainset[0]

    for feature in required_features:
        assert feature in sample_event, f"Feature {feature} not found in data, this is REQUIRED"
    
    missing_optional_features = [ feature for feature in optional_features if feature not in sample_event ]
    for feature in missing_optional_features:
        warnings.warn(f"OPTIONAL feature [{feature}] not found in data")

def convert_to_latest_pyg_format(event):
    """
    Convert the data to the latest PyG format.
    """
    return Data.from_dict(event.__dict__)

def construct_event_truth(event, config):
    event.edge_index, event.y = graph_intersection(event.edge_index, event.truth_graph)
    assert event.y.shape[0] == event.edge_index.shape[1], f"Input graph has {event.edge_index.shape[1]} edges, but {event.y.shape[0]} truth labels"

    event.weights = event.y.float()
    event.weights[event.weights == 0] = -1.0

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

    for weight_val, weight_conditions in weighting_config.items():
        weight_val = float(weight_val)
        condition_mask = torch.ones_like(event.weights, dtype=torch.bool)
        for condition_feature, condition_value in weight_conditions.items():
            if condition_feature in event:
                if event[condition_feature].shape[0] == event.x.shape[0]:
                    edge_features = event[condition_feature][event.edge_index[0]]
                elif event[condition_feature].shape[0] == event.edge_index.shape[1]:
                    edge_features = event[condition_feature]
                else:
                    raise ValueError(f"Feature {condition_feature} has incorrect shape {event[condition_feature].shape} for weighting")
            else:
                raise KeyError(f"Feature {condition_feature} not found in data for weighting")
            
            if isinstance(condition_value, (int, float, bool)):
                condition_mask = condition_mask & (edge_features == condition_value)
            elif isinstance(condition_value, (list, tuple)) and len(condition_value) == 2:
                print(edge_features)
                condition_mask = condition_mask & (edge_features.float() >= condition_value[0]) & (edge_features.float() <= condition_value[1])
            else:
                raise ValueError(f"Condition value {condition_value} not understood for weighting")
        event.weights[condition_mask] = weight_val



def graph_intersection(
    pred_graph, truth_graph, using_weights=False, weights_bidir=None
):
    """
    TODO: Tidy up this function - possibly it is more convoluted than it needs to be.
    """

    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.from_numpy(
        np.vstack([e_intersection.row, e_intersection.col])
    ).long()  # .to(device)
    y = torch.from_numpy(e_intersection.data > 0)  # .to(device)

    return new_pred_graph, y