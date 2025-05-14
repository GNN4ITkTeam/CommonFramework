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
import csv

from torch_geometric.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Dict

from acorn.utils import (
    run_data_tests,
    load_datafiles_in_dir,
    handle_hard_cuts,
    handle_weighting,
)
from . import utils
from acorn.utils.loading_utils import (
    add_variable_name_prefix_in_pyg,
    infer_num_nodes,
    remove_variable_name_prefix_in_pyg,
)


class TrackBuildingStage:
    def __init__(self, hparams, get_logger=True):
        super().__init__()

        self.dataset_class = GraphDataset
        self.hparams: Dict

        # Logging config
        if get_logger:
            self.log = logging.getLogger("TrackBuilding")
            log_level = hparams.get("log_level", "WARNING").upper()
            self.log.setLevel(logging._nameToLevel.get(log_level, logging.WARNING))
            self.log.info(f"Using log level {log_level}")

        self.event_prefix = hparams.get("event_prefix", "")
        if self.event_prefix != "":
            self.event_prefix += "_"

    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        if stage in ["fit", "predict"]:
            self.load_data(stage, self.hparams["input_dir"])
            # self.test_data(stage)
        elif stage == "test":
            torch.manual_seed(0)
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
        required_features = [
            "hit_x",
            "edge_index",
            "track_edges",
            "track_to_edge_map",
            "edge_y",
        ]
        optional_features = [
            "track_particle_id",
            "track_particle_nhits",
            "track_particle_primary",
            "track_particle_pdgId",
            "hit_region",
            "hit_id",
            "track_particle_pt",
            "track_particle_radius",
            "track_particle_eta",
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
        track_builder = cls(config)
        track_builder.setup(stage="test")

        all_plots = config["plots"]

        # TODO: Handle the list of plots properly
        for plot_function, plot_config in all_plots.items():
            if hasattr(track_builder, plot_function):
                getattr(track_builder, plot_function)(plot_config, config)
            else:
                print(f"Plot {plot_function} not implemented")

    def tracking_efficiency(self, plot_config, config):
        """
        Plot the track efficiency vs. pT of the edge.
        """
        all_y_truth, all_pt = [], []
        dataset = getattr(self, config["dataset"])

        evaluated_events = []
        times = []
        for event in tqdm(dataset):
            evaluated_events.append(
                utils.evaluate_labelled_graph(
                    event,
                    matching_fraction=config.get("matching_fraction", 0.5),
                    matching_style=config.get("matching_style", "ATLAS"),
                    sel_conf=config.get("target_tracks", {}),
                    min_track_length=config.get("min_track_length", 5),
                )
            )
            times.append(event.time_taken)

        times = np.array(times)
        time_avg = np.mean(times)
        time_std = np.std(times)

        evaluated_events = pd.concat(evaluated_events)

        # Save evaluated_events out as a CSV file to be used later
        if self.hparams.get("saveMatchingDF", False):
            save_csv_path = os.path.join(self.hparams["stage_dir"], "matching_df.csv")
            print("Saving matching_df CSV at:", save_csv_path)
            evaluated_events.to_csv(save_csv_path)

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

        eff = n_reconstructed_particles / n_particles
        fake_rate = 1 - (n_matched_tracks / n_tracks)
        dup_rate = n_dup_reconstructed_particles / n_reconstructed_particles

        result_summary = make_result_summary(
            n_reconstructed_particles,
            n_particles,
            n_matched_tracks,
            n_tracks,
            n_dup_reconstructed_particles,
            eff,
            fake_rate,
            dup_rate,
            time_avg,
            time_std,
        )

        res_fname = os.path.join(
            self.hparams["stage_dir"],
            f"results_summary_{self.hparams.get('matching_style', 'ATLAS')}.txt",
        )

        with open(res_fname, "w") as f:
            f.write(result_summary)

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

        # Plot the results across pT and eta (if provided in conf file)
        os.makedirs(self.hparams["stage_dir"], exist_ok=True)

        for var, varconf in plot_config["variables"].items():
            save_path = os.path.join(
                self.hparams["stage_dir"],
                f"track_reconstruction_eff_vs_{var}_{self.hparams.get('matching_style', 'ATLAS')}.png",
            )

            utils.plot_eff(
                particles,
                var,
                varconf,
                save_path=save_path,
                trackML_label=self.hparams.get("trackML_label", False),
            )

            print(
                "Finish plotting. Find the plot at"
                f' {os.path.join(config["stage_dir"], save_path)}'
            )

        return {
            "efficiency": eff,
            "fake_rate": fake_rate,
            "dup_rate": dup_rate,
            "num_reconstructed_particles": n_reconstructed_particles,
            "num_particles": n_particles,
            "num_matched_tracks": n_matched_tracks,
            "num_tracks": n_tracks,
            "num_dup_reconstructed_particles": n_dup_reconstructed_particles,
        }

    def apply_target_conditions(self, event, target_tracks):
        """
        Apply the target conditions to the event. This is used for the evaluation stage.
        Target_tracks is a list of dictionaries, each of which contains the conditions to be applied to the event.
        """
        passing_tracks = torch.ones(event.track_to_edge_map.shape[0], dtype=torch.bool)

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

    def save_tracks(self, graph, tracks, output_dir):
        tracks_dir = os.path.join(
            self.hparams["stage_dir"], f"{os.path.basename(output_dir)}_tracks"
        )
        os.makedirs(tracks_dir, exist_ok=True)

        if self.hparams.get("save_tracks_as_csv", False):
            _delimiter = ","
            filename = self.event_prefix
            if self.hparams.get("athena_csv_format", False):
                # In Athena, the GNNTrackReader needs CSV track file with the following name : "<prefix>_RUNNUMBER_EVTNUMBER.csv".
                # By default, the <prefix> is set to "track" in Athena, so if you want to produce track files with the equivalent filename, set 'event_prefix' in yaml config file to : "track_RUNNUMBER".
                # If you want to use a specific prefix, you have to define it in both ACORN and Athena as follows:
                # in ACORN yaml config file with 'event_prefix' : "<any_prefix_you_want>_RUNNUMBER" (The run number is mandatory, otherwise it will not work in Athena.)
                # in Athena with 'csvPrefix' : "<any_prefix_you_want>" (without the RUNNUMBER, it is already handled by Athena.)

                filename += f"{graph.event_id[0].lstrip('0')}.csv"
            else:
                filename += f"event{graph.event_id[0]}.csv"
        else:
            _delimiter = " "
            filename = f"{self.event_prefix}event{graph.event_id[0]}.txt"

        output_file = os.path.join(tracks_dir, filename)
        with open(output_file, "w", newline="") as f:
            csv.writer(f, delimiter=_delimiter).writerows(tracks)

    def save_graph(self, graph, output_dir):
        if not self.hparams.get("variable_with_prefix"):
            graph = remove_variable_name_prefix_in_pyg(graph)

        torch.save(
            graph,
            os.path.join(
                output_dir, f"{self.event_prefix}event{graph.event_id[0]}.pyg"
            ),
        )
        return graph


def make_result_summary(
    n_reconstructed_particles,
    n_particles,
    n_matched_tracks,
    n_tracks,
    n_dup_reconstructed_particles,
    eff,
    fake_rate,
    dup_rate,
    time_avg,
    time_std,
):
    summary = f"Number of reconstructed particles: {n_reconstructed_particles}\n"
    summary += f"Number of particles: {n_particles}\n"
    summary += f"Number of matched tracks: {n_matched_tracks}\n"
    summary += f"Number of tracks: {n_tracks}\n"
    summary += (
        "Number of duplicate reconstructed particles:"
        f" {n_dup_reconstructed_particles}\n"
    )
    summary += f"Efficiency: {eff:.3f}\n"
    summary += f"Fake rate: {fake_rate:.3f}\n"
    summary += f"Duplication rate: {dup_rate:.3f}\n"
    summary += f"Latency: {time_avg:.3f} ± {time_std:.3f}\n"

    return summary


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
        event = torch.load(
            event_path, map_location=torch.device("cpu"), weights_only=False
        )
        self.preprocess_event(event)

        # return (event, event_path) if self.stage == "predict" else event
        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """
        if (not self.hparams.get("variable_with_prefix")) or self.hparams.get(
            "add_variable_name_prefix_in_pyg"
        ):
            event = add_variable_name_prefix_in_pyg(event)
        infer_num_nodes(event)
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

        assert event.edge_y.shape[0] == event.edge_index.shape[1], (
            f"Input graph has {event.edge_index.shape[1]} edges, but"
            f" {event.edge_y.shape[0]} truth labels"
        )

        if self.hparams is not None and "weighting" in self.hparams.keys():
            assert isinstance(self.hparams["weighting"], list) & isinstance(
                self.hparams["weighting"][0], dict
            ), "Weighting must be a list of dictionaries"
            event.edge_weights = handle_weighting(event, self.hparams["weighting"])
        else:
            event.edge_weights = torch.ones_like(event.edge_y, dtype=torch.float32)

    def handle_edge_list(self, event):
        """
        TODO
        """
        pass
