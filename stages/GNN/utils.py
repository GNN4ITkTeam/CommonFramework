import sys
import os
import warnings

import torch
from torch_geometric.data import Data

from .models.interaction_gnn import InteractionGNN

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def load_dataset_from_dir(input_dir, data_name, data_num):
    """
    Load in the PyG Data dataset from the data directory.
    """
    data_dir = os.path.join(input_dir, "data", data_name)
    if not os.path.exists(data_dir):
        warnings.warn(f"Data directory {data_dir} does not exist. Looking in {input_dir}.")
        data_dir = os.path.join(input_dir, data_name)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")][:data_num]
    assert len(data_files) > 0, f"No data files found in {data_dir}"
    assert len(data_files) == data_num, f"Number of data files found ({len(data_files)}) is less than the number requested ({data_num})"

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