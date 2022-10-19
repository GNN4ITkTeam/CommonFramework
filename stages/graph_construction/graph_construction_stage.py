"""
This class represents the entire logic of the graph construction stage. In particular, it
1. Loads events from the Athena-dumped csv files
2. Processes them into PyG Data objects with the specificied structure (see docs)
3. Runs the training of the metric learning or module map
4. Can run inference to build graphs
5. Can run evaluation to plot/print the performance of the graph construction

TODO: Update structure with the latest Gravnet base class
"""

import sys
sys.path.append("../")
import os
import warnings
import re

from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset
import torch
import pandas as pd

from ..utils import load_datafiles_in_dir, run_data_tests, construct_event_truth

class GraphConstructionStage:
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        
        self.trainset, self.valset, self.testset = None, None, None
        self.use_pyg = False # You can set this to True if you want to use the PyG LightningModule
        self.use_csv = False # You can set this to True if you want to use CSV Reading
        self.dataset_class = EventDataset

    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        self.load_data()
        if self.use_pyg and not self.use_csv:
            self.test_data()
    

    def load_data(self):
        """
        Load in the data for training, validation and testing.
        """

        for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
            dataset = self.dataset_class(self.hparams["input_dir"], data_name, data_num, use_pyg = self.use_pyg, use_csv = self.use_csv)
            setattr(self, data_name, dataset)

    def test_data(self):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = ["x", "track_edges"]
        optional_features = ["pid", "n_hits", "primary", "pdg_id", "ghost", "shared", "module_id", "region_id", "hit_id"]

        run_data_tests([self.trainset, self.valset, self.testset], required_features, optional_features)

        # TODO: Add test for the building of input data
        # assert self.trainset[0].x.shape[1] == self.hparams["spatial_channels"], "Input dimension does not match the data"

        # TODO: Add test for the building of truth data

    @classmethod
    def infer(cls, config):
        """ 
        The gateway for the inference stage. This class method is called from the infer_stage.py script.
        """
        if isinstance(cls, LightningModule):
            graph_constructor = cls.load_from_checkpoint(os.path.join(config["input_dir"], "checkpoints", "last.ckpt"))
            graph_constructor.hparams.update(config) # Update the configs used in training with those to be used in inference
        else:
            graph_constructor = cls(config)
    
        graph_constructor.setup(stage="predict")

        for data_name in ["trainset", "valset", "testset"]:
            if hasattr(graph_constructor, data_name):
                graph_constructor.build_graphs(dataset = getattr(graph_constructor, data_name), data_name = data_name)


    def build_graphs(self):
        """
        Build the graphs using the trained model. This is the only function that needs to be overwritten by the child class.
        """
        pass
    
    def eval(self):
        pass

    

class EventDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(self, input_dir, data_name, num_events, use_pyg=False, use_csv=False, preload=False, hparams=None, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        
        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.loaded = False
        self.use_pyg = use_pyg
        self.use_csv = use_csv
        
        self.evt_ids = self.find_evt_ids()
        
    def len(self):
        return len(self.evt_ids)

    def get(self, idx):
        """
        Handles the iteration through the dataset. Depending on how dataset is configured for PyG and CSV file loading,
        will return either PyG events (automatically batched by PyG), CSV files (not batched/concatenated), or both (not batched/concatenated).
        Assumes files are saved correctly from EventReader as: event000...-truth.csv, -particles.csv and -graph.pyg
        """        

        graph = None
        particles = None
        hits = None

        if self.use_pyg:
            graph = torch.load(os.path.join(self.input_dir, self.data_name, f"event{self.evt_ids[idx]}-graph.pyg"))
            if not self.use_csv:
                return graph

        if self.use_csv:
            particles = pd.read_csv(os.path.join(self.input_dir, self.data_name, f"event{self.evt_ids[idx]}-particles.csv"))
            hits = pd.read_csv(os.path.join(self.input_dir, self.data_name, f"event{self.evt_ids[idx]}-truth.csv"))
            if not self.use_pyg:
                return particles, hits

        return graph, particles, hits

    def find_evt_ids(self):
        """
        Returns a list of all event ids, which are the numbers in filenames that end in .csv and .pyg
        """

        all_files = os.listdir(os.path.join(self.input_dir, self.data_name))
        all_files = [f for f in all_files if f.endswith(".csv") or f.endswith(".pyg")]
        all_event_ids = sorted(list({re.findall("[0-9]+", file)[-1] for file in all_files}))
        if self.num_events is not None:
            all_event_ids = all_event_ids[:self.num_events]

        # Check that events are present for the requested filetypes
        if self.use_csv:
            all_event_ids = [evt_id for evt_id in all_event_ids if (f"event{evt_id}-truth.csv" in all_files) and (f"event{evt_id}-particles.csv" in all_files)]
        if self.use_pyg:
            all_event_ids = [evt_id for evt_id in all_event_ids if f"event{evt_id}-graph.pyg" in all_files]

        return all_event_ids

    def construct_truth(self, event):
        """
        Construct the truth and weighting labels for the model training.
        """
        # TODO: Implement!
        pass