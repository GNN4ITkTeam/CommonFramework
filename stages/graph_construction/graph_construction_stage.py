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

from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset
import torch

from .utils import str_to_class, load_datafiles_in_dir, run_data_tests, construct_event_truth

class GraphConstructionStage(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Assign hyperparameters
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = CsvEventDataset

        # Load in the model to be used
        self.model = str_to_class(self.hparams["model_name"])(self.hparams)


    def forward(self, batch):

        return self.model(batch)

    def setup(self, stage):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        self.load_data()
        self.test_data()

    def load_data(self):
        """
        Load in the data for training, validation and testing.
        """

        for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
            if data_num > 0:
                dataset = self.dataset_class(self.hparams["input_dir"], data_name, data_num, self.hparams)
                setattr(self, data_name, dataset)

    def test_data(self):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = ["x", "edge_index", "truth_graph"]
        optional_features = ["pid", "n_hits", "primary", "pdg_id", "ghost", "shared", "module_id", "region_id", "hit_id"]

        run_data_tests(self.trainset, self.valset, self.testset, required_features, optional_features)

        assert self.trainset[0].x.shape[1] == self.hparams["spatial_channels"], "Input dimension does not match the data"

    def build_infer_data(self):

        pass

    def produce_plots(self):

        pass

    

class CsvEventDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(self, input_dir, data_name, num_events, hparams=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        
        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        
        self.input_paths = load_datafiles_in_dir(self.input_dir, self.data_name, self.num_events)
        
    def len(self):
        return len(self.input_paths)

    def get(self, idx):

        event = torch.load(self.input_paths[idx], map_location=torch.device("cpu"))
        event = self.construct_truth(event)
                
        return event

    def construct_truth(self, event):
        """
        Construct the truth and weighting labels for the model training.
        """
        
        return construct_event_truth(event, self.hparams)