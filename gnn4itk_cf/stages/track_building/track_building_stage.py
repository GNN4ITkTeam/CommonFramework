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
This class represents the entire logic of the track building stage. In particular, it
1. 

TODO: Update structure with the latest Gravnet base class
"""

import os
import logging

from torch_geometric.data import Dataset
import torch
import pandas as pd
import numpy as np
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
        """
        Initialise the Class that build and evaluate the tracks
        """

        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = GraphDataset

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

        # if stage == "fit":
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
        ]

        run_data_tests(
            [self.trainset, self.valset, self.testset],
            required_features,
            optional_features,
        )

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
        graph_constructor.cache_dfs(config)
        suffix = f"{config['matching_style']}_{config['matching_fraction']}"
        graph_constructor.write_high_level_stats(config, suffix)

        # TODO: Handle the list of plots properly
        for plot_function, plot_config in all_plots.items():
            if hasattr(graph_constructor, plot_function):
                getattr(graph_constructor, plot_function)(plot_config, suffix)
            else:
                print(f"Plot {plot_function} not implemented")
                
    def eval_preprocess_event(self, event, config):
        if not hasattr(event, "bgraph") and hasattr(event, "labels"):
            event.bgraph = torch.stack([
                torch.arange(event.labels.shape[0], device = event.labels.device)[event.labels >= 0],
                torch.as_tensor(event.labels[event.labels >= 0], device = event.labels.device)
            ])
        return event
    
    def cache_dfs(self, config):
        self.all_dfs = []
        for event in tqdm(self.testset):
            event = self.eval_preprocess_event(event, config)
            matching_df, truth_df = utils.evaluate_tracking(
                event, 
                event.bgraph,
                min_hits = config["min_track_length"],
                signal_selection = config["signal_selection"],
                target_selection = config["target_selection"],
                matching_fraction = config["matching_fraction"],
                style=config["matching_style"]
            )
            self.all_dfs.append((matching_df, truth_df))
            
    def write_high_level_stats(self, config, suffix):
        all_stats = {}
        for matching_df, truth_df in tqdm(self.all_dfs):
            stats = utils.get_statistics(matching_df, truth_df)
            if all_stats:
                for name in all_stats:
                    all_stats[name].append(stats[name])
            else:
                for name in stats:
                    all_stats[name] = [stats[name]]
        with open(
            os.path.join(
                self.hparams["stage_dir"], f"summary_{suffix}.txt"
            ),
            "w"
        ) as f:
            f.write(
f"""reconstructed particles: {sum(all_stats['reconstructed_particles'])},
total particles: {sum(all_stats['total_particles'])},
tracking efficiency: {sum(all_stats['reconstructed_particles']) / sum(all_stats['total_particles'])},
reconstructed signal: {sum(all_stats['reconstructed_signal'])},
total signal: {sum(all_stats['total_signal'])},
signal efficiency: {sum(all_stats['reconstructed_signal']) / sum(all_stats['total_signal'])},
num duplicated tracks: {sum(all_stats['num_duplicated_tracks'])},
num matched particles: {sum(all_stats['num_matched_particles'])},
duplicate rate: {sum(all_stats['duplicate_rate']) / len(all_stats['duplicate_rate'])},
num tracks: {sum(all_stats['num_tracks'])},
num_reconstructed particles: {sum(all_stats['num_reconstructed_particles'])},
fake rate: {sum(all_stats['fake_rate']) / len(all_stats['duplicate_rate'])}"""
            )

    def tracking_efficiency_pt(self, plot_config, suffix):
        """
        Plot the graph construction efficiency vs. pT of the tracks.
        """
        all_stats = {}
        pt_bins = np.logspace(np.log10(plot_config["min_pt"]), np.log10(plot_config["max_pt"]), plot_config["n_bins"])
        for matching_df, truth_df in tqdm(self.all_dfs):
            stats = utils.get_statistics(matching_df, truth_df, "pt", pt_bins)
            if all_stats:
                for name in all_stats:
                    all_stats[name].append(stats[name])
            else:
                for name in stats:
                    all_stats[name] = [stats[name]]

        # Plot the results across pT and eta
        utils.plot_eff(
            all_stats = all_stats,
            bins = pt_bins,
            xlabel = r"$p_T$ (MeV)",
            caption = plot_config["caption"],
            save_path = os.path.join(
                self.hparams["stage_dir"], f"eff_vs_pt_{suffix}.png"
            ),
        )
        
    def tracking_efficiency_eta(self, plot_config, suffix):
        """
        Plot the graph construction efficiency vs. eta of the tracks.
        """
        all_stats = {}
        eta_bins = np.linspace(plot_config["min_eta"], plot_config["max_eta"], plot_config["n_bins"])
        for matching_df, truth_df in tqdm(self.all_dfs):
            stats = utils.get_statistics(matching_df, truth_df, "eta", eta_bins)
            if all_stats:
                for name in all_stats:
                    all_stats[name].append(stats[name])
            else:
                for name in stats:
                    all_stats[name] = [stats[name]]

        # Plot the results across pT and eta
        utils.plot_eff(
            all_stats = all_stats,
            bins = eta_bins,
            xlabel = r"Pseudo rapidity $\eta$",
            caption = plot_config["caption"],
            save_path = os.path.join(
                self.hparams["stage_dir"], f"eff_vs_eta_{suffix}.png"
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
