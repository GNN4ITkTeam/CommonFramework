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

import os
import logging

from torch_geometric.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm

from gnn4itk_cf.utils import (
    run_data_tests,
    load_datafiles_in_dir,
    handle_hard_cuts,
    handle_weighting,
)
from . import utils


class TrackBuildingStage:
    def __init__(self, hparams):
        super().__init__()

        self.dataset_class = GraphDataset

        # Logging config
        self.log = logging.getLogger("TrackBuilding")
        log_level = hparams.get("log_level", "WARNING").upper()
        self.log.setLevel(logging._nameToLevel.get(log_level, logging.WARNING))
        self.log.info(f"Using log level {log_level}")

    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        if stage in ["fit", "predict"]:
            self.load_data(stage, self.hparams["input_dir"])
            self.test_data(stage)
        elif stage == "test":
            self.load_data(stage, self.hparams["stage_dir"])

    def load_data(self, stage, input_dir):
        """
        Load in the data for training, validation and testing.
        """

        for data_name, data_num in zip(
            ["trainset", "valset", "testset"], self.hparams["data_split"]
        ):
            if data_num > 0:
                dataset = self.dataset_class(
                    input_dir, data_name, data_num, stage, self.hparams
                )
                setattr(self, data_name, dataset)

    def test_data(self, stage):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = ["x", "edge_index", "track_edges", "truth_map", "y"]
        optional_features = [
            "particle_id",
            "nhits",
            "primary",
            "pdgId",
            "ghost",
            "shared",
            "module_id",
            "region",
            "hit_id",
            "pt",
            "radius",
            "eta_particle",
        ]

        # Test only non empty data set
        datasets = [
            getattr(self, data_name)
            for data_name in ["trainset", "valset", "testset"]
            if hasattr(self, data_name)
        ]

        run_data_tests(datasets, required_features, optional_features)

    @classmethod
    def infer(cls, config):
        """
        The gateway for the inference stage. This class method is called from the infer_stage.py script.
        """
        graph_constructor = cls(config)
        graph_constructor.setup(stage="predict")

        for data_name in ["trainset", "valset", "testset"]:
            if hasattr(graph_constructor, data_name):
                graph_constructor.build_tracks(
                    dataset=getattr(graph_constructor, data_name), data_name=data_name
                )

    def build_tracks(self, dataset, data_name):
        """
        Build the track candidates using the track building algorithm. This is the only function that needs to be overwritten by the child class.
        """
        pass

    @classmethod
    def evaluate(cls, config):
        """
        The gateway for the evaluation stage. This class method is called from the eval_stage.py script.
        """

        # Load data from testset directory
        graph_constructor = cls(config)
        graph_constructor.setup(stage="test")

        all_plots = config["plots"]

        # TODO: Handle the list of plots properly
        for plot_function, plot_config in all_plots.items():
            if hasattr(graph_constructor, plot_function):
                getattr(graph_constructor, plot_function)(plot_config, config)
            else:
                print(f"Plot {plot_function} not implemented")

    def tracking_efficiency(self, plot_config, config):
        """
        Plot the graph construction efficiency vs. pT of the edge.
        """
        all_y_truth, all_pt = [], []

        evaluated_events = []
        for event in tqdm(self.testset):
            evaluated_events.append(
                utils.evaluate_labelled_graph(
                    event,
                    matching_fraction=config["matching_fraction"],
                    matching_style=config["matching_style"],
                    min_track_length=config["min_track_length"],
                    min_particle_length=config["min_particle_length"],
                )
            )

        evaluated_events = pd.concat(evaluated_events)

        particles = evaluated_events[evaluated_events["is_reconstructable"]]
        reconstructed_particles = particles[
            particles["is_reconstructed"] & particles["is_matchable"]
        ]
        tracks = evaluated_events[evaluated_events["is_matchable"]]
        matched_tracks = tracks[tracks["is_matched"]]

        n_particles = len(particles.drop_duplicates(subset=["event_id", "particle_id"]))
        n_reconstructed_particles = len(
            reconstructed_particles.drop_duplicates(subset=["event_id", "particle_id"])
        )

        n_tracks = len(tracks.drop_duplicates(subset=["event_id", "track_id"]))
        n_matched_tracks = len(
            matched_tracks.drop_duplicates(subset=["event_id", "track_id"])
        )

        n_dup_reconstructed_particles = (
            len(reconstructed_particles) - n_reconstructed_particles
        )

        logging.info(f"Number of reconstructed particles: {n_reconstructed_particles}")
        logging.info(f"Number of particles: {n_particles}")
        logging.info(f"Number of matched tracks: {n_matched_tracks}")
        logging.info(f"Number of tracks: {n_tracks}")
        logging.info(
            "Number of duplicate reconstructed particles:"
            f" {n_dup_reconstructed_particles}"
        )

        # Plot the results across pT and eta
        eff = n_reconstructed_particles / n_particles
        fake_rate = 1 - (n_matched_tracks / n_tracks)
        dup_rate = n_dup_reconstructed_particles / n_reconstructed_particles

        logging.info(f"Efficiency: {eff:.3f}")
        logging.info(f"Fake rate: {fake_rate:.3f}")
        logging.info(f"Duplication rate: {dup_rate:.3f}")

        # First get the list of particles without duplicates
        grouped_reco_particles = particles.groupby("particle_id")[
            "is_reconstructed"
        ].any()
        # particles["is_reconstructed"] = particles["particle_id"].isin(grouped_reco_particles[grouped_reco_particles].index.values)
        particles.loc[
            particles["particle_id"].isin(
                grouped_reco_particles[grouped_reco_particles].index.values
            ),
            "is_reconstructed",
        ] = True
        particles = particles.drop_duplicates(subset=["particle_id"])

        # Plot the results across pT and eta
        pt_units = plot_config["pt_units"] if "pt_units" in plot_config else "GeV"
        utils.plot_pt_eff(
            particles,
            pt_units,
            save_path=os.path.join(
                self.hparams["stage_dir"], "track_reconstruction_eff_vs_pt.png"
            ),
        )

    def apply_target_conditions(self, event, target_tracks):
        """
        Apply the target conditions to the event. This is used for the evaluation stage.
        Target_tracks is a list of dictionaries, each of which contains the conditions to be applied to the event.
        """
        passing_tracks = torch.ones(event.truth_map.shape[0], dtype=torch.bool)

        for key, values in target_tracks.items():
            if isinstance(values, list):
                # passing_tracks = passing_tracks & (values[0] <= event[key]).bool() & (event[key] <= values[1]).bool()
                passing_tracks = (
                    passing_tracks
                    * (values[0] <= event[key].float())
                    * (event[key].float() <= values[1])
                )
            else:
                passing_tracks = passing_tracks * (event[key] == values)

        event.target_mask = passing_tracks


class GraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir,
        data_name=None,
        num_events=None,
        stage="fit",
        hparams=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(input_dir, transform, pre_transform, pre_filter)

        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.stage = stage

        self.input_paths = load_datafiles_in_dir(
            self.input_dir, self.data_name, self.num_events
        )
        self.input_paths.sort()  # We sort here for reproducibility

    def len(self):
        return len(self.input_paths)

    def get(self, idx):
        event_path = self.input_paths[idx]
        event = torch.load(event_path, map_location=torch.device("cpu"))
        self.preprocess_event(event)

        # return (event, event_path) if self.stage == "predict" else event
        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """

        self.apply_hard_cuts(event)
        self.construct_weighting(event)
        self.handle_edge_list(event)

    def apply_hard_cuts(self, event):
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
            handle_hard_cuts(event, self.hparams["hard_cuts"])

    def construct_weighting(self, event):
        """
        Construct the weighting for the event
        """

        assert event.y.shape[0] == event.edge_index.shape[1], (
            f"Input graph has {event.edge_index.shape[1]} edges, but"
            f" {event.y.shape[0]} truth labels"
        )

        if self.hparams is not None and "weighting" in self.hparams.keys():
            assert isinstance(self.hparams["weighting"], list) & isinstance(
                self.hparams["weighting"][0], dict
            ), "Weighting must be a list of dictionaries"
            event.weights = handle_weighting(event, self.hparams["weighting"])
        else:
            event.weights = torch.ones_like(event.y, dtype=torch.float32)

    def handle_edge_list(self, event):
        """
        TODO
        """
        pass
