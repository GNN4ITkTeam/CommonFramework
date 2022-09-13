import sys
import os
import warnings

import torch

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

    return [ torch.load(f) for f in data_files ]
