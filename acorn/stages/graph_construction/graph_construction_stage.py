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
import warnings

from acorn.utils.loading_utils import add_variable_name_prefix_in_pyg, infer_num_nodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from acorn.utils import (
    run_data_tests,
    plot_eff_pur_region,
    eval_utils,
    get_condition_lambda,
    handle_hard_node_cuts,
    get_pyg_data_keys,
)


class GraphConstructionStage:
    def __init__(self):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        self.trainset, self.valset, self.testset = None, None, None
        self.use_csv = False
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

            if not self.use_csv:
                self.test_data()

        elif stage == "test":
            self.load_data(self.hparams["stage_dir"])

        if stage in ["predict", "test"]:
            torch.set_float32_matmul_precision("highest")

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
        required_features = ["hit_x", "track_edges"]
        optional_features = [
            "track_particle_pid",
            "track_particle_n_hits",
            "track_particle_primary",
            "track_particle_pdg_id",
            "hit_module_id",
            "hit_region_id",
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
            edge_truth.append(event.edge_y)
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
        passing_tracks = torch.ones(
            event.track_to_edge_map.shape[0], dtype=torch.bool
        ).to(self.device)

        for condition_key, condition_val in target_tracks.items():
            condition_lambda = get_condition_lambda(condition_key, condition_val)
            passing_tracks = passing_tracks * condition_lambda(event).to(self.device)

        event.track_target_mask = passing_tracks


class EventDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir,
        data_name=None,
        num_events=None,
        use_csv=False,
        stage="fit",
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
        self.use_csv = use_csv
        self.stage = stage
        self.evt_ids = self.find_evt_ids()

    def len(self):
        return len(self.evt_ids)

    def get(self, idx):
        """
        Handles the iteration through the dataset. Depending on how dataset is configured for PyG and CSV file loading,
        will return either PyG events (automatically batched by PyG), CSV files (not batched/concatenated), or both (not batched/concatenated).
        Assumes files are saved correctly from EventReader as: event000...-truth.csv, -particles.csv and -graph.pyg
        """
        event_file_root_name = f"event{self.evt_ids[idx]}"
        if self.hparams.get("event_prefix"):
            event_file_root_name = (
                f"{self.hparams['event_prefix']}_{event_file_root_name}"
            )

        event_path = os.path.join(self.input_dir, self.data_name, event_file_root_name)

        graph_path = (
            f"{event_path}-graph.pyg"
            if os.path.exists(f"{event_path}-graph.pyg")
            else f"{event_path}.pyg"
        )
        graph = torch.load(graph_path)
        graph = self.preprocess_graph(graph)

        if not self.use_csv:
            return graph

        particles = pd.read_csv(f"{event_path}-particles.csv")
        hits = pd.read_csv(f"{event_path}-truth.csv")
        hits = self.preprocess_hits(hits, graph)

        return graph, particles, hits

    def preprocess_graph(self, graph):
        """Preprocess the PyG graph before returning it."""
        if (not self.hparams.get("variable_with_prefix")) or self.hparams.get(
            "add_variable_name_prefix_in_pyg"
        ):
            graph = add_variable_name_prefix_in_pyg(graph)
        infer_num_nodes(graph)
        graph = self.apply_hard_cuts(graph)
        self.scale_features(graph)
        return graph

    def preprocess_hits(self, hits, graph):
        """Preprocess the hits dataframe before returning it."""
        hits = self.apply_hard_cuts(hits, passing_hit_ids=graph.hit_id.numpy())
        return hits

    def find_evt_ids(self):
        """
        Returns a list of all event ids, which are the numbers in filenames that end in .csv and .pyg
        """

        input_data_dir = os.path.join(self.input_dir, self.data_name)
        all_files = os.listdir(input_data_dir) if os.path.isdir(input_data_dir) else []
        all_files = [f for f in all_files if f.endswith(".csv") or f.endswith(".pyg")]
        all_event_ids = sorted(
            list({re.findall("[0-9]+", file)[-1] for file in all_files})
        )

        if len(all_event_ids) == 0:
            warnings.warn(f"No events found in {self.input_dir}/{self.data_name}")

        if self.num_events is not None:
            assert self.num_events <= len(
                all_event_ids
            ), f"Requested {self.num_events} events, but only found {len(all_event_ids)} in {self.input_dir}/{self.data_name}"
            all_event_ids = all_event_ids[: self.num_events]

        # Check that events are present for the requested filetypes
        if self.hparams.get("event_prefix"):
            prefix = self.hparams["event_prefix"] + "_"
        else:
            prefix = ""

        csv_event_ids = []
        if self.use_csv:
            csv_event_ids = [
                evt_id
                for evt_id in all_event_ids
                if (f"{prefix}event{evt_id}-truth.csv" in all_files)
                and (f"{prefix}event{evt_id}-particles.csv" in all_files)
            ]

        pyg_event_ids = [
            evt_id
            for evt_id in all_event_ids
            if f"{prefix}event{evt_id}-graph.pyg" in all_files
            or f"{prefix}event{evt_id}.pyg" in all_files
        ]

        if self.use_csv:
            all_event_ids = list(set(csv_event_ids) & set(pyg_event_ids))
        else:
            all_event_ids = pyg_event_ids

        return all_event_ids

    def apply_hard_cuts(self, event, passing_hit_ids=None):
        """
        Apply hard cuts to the event. This is implemented by
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """

        if (
            self.hparams is not None
            and "hard_cuts" in self.hparams.keys()
            and self.hparams["hard_cuts"]
        ):
            assert isinstance(
                self.hparams["hard_cuts"], dict
            ), "Hard cuts must be a dictionary"
            event = handle_hard_node_cuts(
                event, self.hparams["hard_cuts"], passing_hit_ids
            )

        return event

    def scale_features(self, graph):
        """
        Handle feature scaling for the graph
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in get_pyg_data_keys(
                    graph
                ), f"Feature {feature} not found in graph"
                graph[feature] = graph[feature] / self.hparams["node_scales"][i]

    def unscale_features(self, graph):
        """
        Unscale features when doing prediction
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in get_pyg_data_keys(
                    graph
                ), f"Feature {feature} not found in graph"
                graph[feature] = graph[feature] * self.hparams["node_scales"][i]
