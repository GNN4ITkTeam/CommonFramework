import sys
import os
import warnings

from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch

from .utils import str_to_class, load_dataset_from_dir, run_data_tests


class GNNStage(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Assign hyperparameters
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None

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
        self.construct_truth()

    def load_data(self):
        """
        Load in the data for training, validation and testing.
        """
        for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
            if data_num > 0:
                dataset = load_dataset_from_dir(self.hparams["input_dir"], data_name, data_num)
                setattr(self, data_name, dataset)

    def test_data(self):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = ["x", "edge_index", "truth_graph"]
        optional_features = ["pid", "n_hits", "primary", "pdg_id", "ghost", "shared", "module_id", "region_id", "hit_id"]

        run_data_tests(self.trainset, self.valset, self.testset, required_features, optional_features)

        assert self.trainset[0].x.shape[1] == self.hparams["spatial_channels"], "Input dimension does not match the data"

        
    def construct_truth(self):
        """
        Construct the truth and weighting labels for the model training.
        """
        pass