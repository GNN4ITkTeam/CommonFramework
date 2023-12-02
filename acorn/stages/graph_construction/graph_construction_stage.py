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
import re

from pytorch_lightning import LightningModule
from torch_geometric.data import Dataset
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from acorn.utils import run_data_tests, plot_eff_pur_region
from acorn.utils import eval_utils
from acorn.utils.mapping_utils import get_condition_lambda


class GraphConstructionStage:
    def __init__(self):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        self.trainset, self.valset, self.testset = None, None, None
        self.use_pyg = (
            False  # You can set this to True if you want to use the PyG LightningModule
        )
        self.use_csv = False  # You can set this to True if you want to use CSV Reading
        self.dataset_class = EventDataset

    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        if stage in ["fit", "predict"]:
            self.load_data(self.hparams["input_dir"])

            if self.use_pyg and not self.use_csv:
                self.test_data()

        elif stage == "test":
            self.load_data(self.hparams["stage_dir"])

    def load_data(self, input_dir):
        """
        Load in the data for training, validation and testing.
        """

        for data_name, data_num in zip(
            ["trainset", "valset", "testset"], self.hparams["data_split"]
        ):
            dataset = self.dataset_class(
                input_dir,
                data_name,
                data_num,
                use_pyg=self.use_pyg,
                use_csv=self.use_csv,
                hparams=self.hparams,
            )
            setattr(self, data_name, dataset)

        print(
            f"Loaded {len(self.trainset)} training events,"
            f" {len(self.valset)} validation events and {len(self.testset)} testing"
            " events"
        )

    def test_data(self):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = ["x", "track_edges"]
        optional_features = [
            "pid",
            "n_hits",
            "primary",
            "pdg_id",
            "ghost",
            "shared",
            "module_id",
            "region_id",
            "hit_id",
        ]

        run_data_tests(
            [self.trainset, self.valset, self.testset],
            required_features,
            optional_features,
        )

        # TODO: Add test for the building of input data
        # assert self.trainset[0].x.shape[1] == self.hparams["spatial_channels"], "Input dimension does not match the data"

        # TODO: Add test for the building of truth data

    @classmethod
    def infer(cls, config):
        """
        The gateway for the inference stage. This class method is called from the infer_stage.py script.
        """
        if isinstance(cls, LightningModule):
            graph_constructor = cls.load_from_checkpoint(
                os.path.join(config["input_dir"], "checkpoints", "last.ckpt")
            )
            graph_constructor.hparams.update(
                config
            )  # Update the configs used in training with those to be used in inference
        else:
            graph_constructor = cls(config)

        graph_constructor.setup(stage="predict")

        for data_name in ["trainset", "valset", "testset"]:
            if hasattr(graph_constructor, data_name):
                graph_constructor.build_graphs(
                    dataset=getattr(graph_constructor, data_name), data_name=data_name
                )

    def build_graphs(self, dataset, data_name):
        """
        Build the graphs using the trained model. This is the only function that needs to be overwritten by the child class.
        """
        pass

    @classmethod
    def evaluate(cls, config, *args):
        """
        The gateway for the evaluation stage. This class method is called from the eval_stage.py script.
        """

        # Load data from testset directory
        graph_constructor = cls(config).to(device)
        graph_constructor.use_csv = False
        graph_constructor.setup(stage="test")

        all_plots = config["plots"]

        # TODO: Handle the list of plots properly
        for plot_function, plot_config in all_plots.items():
            if hasattr(eval_utils, plot_function):
                getattr(eval_utils, plot_function)(
                    graph_constructor, plot_config, config
                )
            else:
                print(f"Plot {plot_function} not implemented")

    def graph_region_efficiency_purity(self, plot_config, config):
        edge_truth, edge_regions = [], []
        node_r, node_z, node_regions = [], [], []

        for event in tqdm(self.testset):
            edge_truth.append(event.y)
            edge_regions.append(
                event.x_region[event.edge_index[0]]
            )  # Assign region depending on first node in edge

            node_r.append(event.x_r)
            node_z.append(event.x_z)
            node_regions.append(event.x_region)

        edge_truth = torch.cat(edge_truth).cpu().numpy()
        edge_positive = np.ones(len(edge_truth))
        edge_regions = torch.cat(edge_regions).cpu().numpy()

        node_r = torch.cat(node_r).cpu().numpy()
        node_z = torch.cat(node_z).cpu().numpy()
        node_regions = torch.cat(node_regions).cpu().numpy()

        fig, ax = plot_eff_pur_region(
            edge_truth,
            edge_positive,
            edge_regions,
            node_r,
            node_z,
            node_regions,
            plot_config,
        )
        fig.savefig(os.path.join(config["stage_dir"], "region_eff_pur.png"))

    def apply_target_conditions(self, event, target_tracks):
        """
        Apply the target conditions to the event. This is used for the evaluation stage.
        Target_tracks is a list of dictionaries, each of which contains the conditions to be applied to the event.
        """
        passing_tracks = torch.ones(event.truth_map.shape[0], dtype=torch.bool).to(
            self.device
        )

        for condition_key, condition_val in target_tracks.items():
            condition_lambda = get_condition_lambda(condition_key, condition_val)
            passing_tracks = passing_tracks * condition_lambda(event).to(self.device)

        event.target_mask = passing_tracks


class EventDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir,
        data_name,
        num_events,
        use_pyg=False,
        use_csv=False,
        preload=False,
        hparams=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **kwargs,
    ):
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
            # Check if file event{self.evt_ids[idx]}-graph.pyg exists
            if os.path.exists(
                os.path.join(
                    self.input_dir,
                    self.data_name,
                    f"event{self.evt_ids[idx]}-graph.pyg",
                )
            ):
                graph = torch.load(
                    os.path.join(
                        self.input_dir,
                        self.data_name,
                        f"event{self.evt_ids[idx]}-graph.pyg",
                    )
                )
            else:
                graph = torch.load(
                    os.path.join(
                        self.input_dir, self.data_name, f"event{self.evt_ids[idx]}.pyg"
                    )
                )
            if not self.use_csv:
                return graph

        if self.use_csv:
            particles = pd.read_csv(
                os.path.join(
                    self.input_dir,
                    self.data_name,
                    f"event{self.evt_ids[idx]}-particles.csv",
                )
            )
            hits = pd.read_csv(
                os.path.join(
                    self.input_dir,
                    self.data_name,
                    f"event{self.evt_ids[idx]}-truth.csv",
                )
            )
            if not self.use_pyg:
                return particles, hits

        return graph, particles, hits

    def find_evt_ids(self):
        """
        Returns a list of all event ids, which are the numbers in filenames that end in .csv and .pyg
        """

        all_files = os.listdir(os.path.join(self.input_dir, self.data_name))
        all_files = [f for f in all_files if f.endswith(".csv") or f.endswith(".pyg")]
        all_event_ids = sorted(
            list({re.findall("[0-9]+", file)[-1] for file in all_files})
        )
        if self.num_events is not None:
            all_event_ids = all_event_ids[: self.num_events]

        # Check that events are present for the requested filetypes
        if self.use_csv:
            all_event_ids = [
                evt_id
                for evt_id in all_event_ids
                if (f"event{evt_id}-truth.csv" in all_files)
                and (f"event{evt_id}-particles.csv" in all_files)
            ]
        if self.use_pyg:
            all_event_ids = [
                evt_id
                for evt_id in all_event_ids
                if f"event{evt_id}-graph.pyg" or f"event{evt_id}.pyg" in all_files
            ]

        return all_event_ids

    def construct_truth(self, event):
        """
        Construct the truth and weighting labels for the model training.
        """
        # TODO: Implement!
        pass
