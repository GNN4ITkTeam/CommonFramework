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

import os

import numpy as np
import pandas as pd
import yaml
from typing import Union
import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
import re
from itertools import chain, product, combinations
from torch_geometric.data import Data
import torch
import warnings
import logging

from acorn.utils.loading_utils import (
    remove_variable_name_prefix_in_pyg,
    variable_name_prefix_map,
)


class EventReader:
    """
    A general class for reading simulated particle collision events from a set of files. Several convenience utilities are built in,
    and conversion to CSV and Pytorch Geometric data objects is enforced. However the reading of the input files is left up to the user.
    It is expected that general usage is, e.g.
    AthenaReader(path/to/files), which consists of:
    1. Raw files -> CSV
    2. CSV -> PyG data objects
    """

    def __init__(self, config):
        self.files = None
        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, str):
            self.config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            raise NotImplementedError

        # Logging config
        self.log = logging.getLogger("EventReader")
        log_level = (
            self.config["log_level"].upper()
            if "log_level" in self.config
            else "WARNING"
        )

        if log_level == "WARNING":
            self.log.setLevel(logging.WARNING)
        elif log_level == "INFO":
            self.log.setLevel(logging.INFO)
        elif log_level == "DEBUG":
            self.log.setLevel(logging.DEBUG)
        else:
            raise ValueError(f"Unknown logging level {log_level}")

        self.log.info(
            "Using log level {}".format(
                logging.getLevelName(self.log.getEffectiveLevel())
            )
        )

    @classmethod
    def infer(cls, config):
        """
        The gateway for the inference stage. This class method is called from the infer_stage.py script.
        It assumes a set of basic steps. These are:
        1. Convert to CSV - This should be implemented by the user
        2. Convert to PyG
        """

        reader = cls(config)
        if not config.get("skip_csv_conversion"):
            reader.convert_to_csv()
            reader._test_csv_conversion()
        reader._convert_to_pyg()
        reader._test_pyg_conversion()

        return reader

    def convert_to_csv(self):
        """
        Convert the full set of Athena events to CSV. This produces files in /trainset, /valset and /testset to ensure no overlaps.
        """

        for dataset, dataset_name in zip(
            [self.trainset, self.valset, self.testset],
            ["trainset", "valset", "testset"],
        ):
            if dataset is not None:
                self._build_all_csv(dataset, dataset_name)

    def _build_all_csv(self, dataset, dataset_name):
        output_dir = os.path.join(self.config["stage_dir"], dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Build CSV files, optionally with multiprocessing
        max_workers = self.config.get("max_workers", 1)
        if max_workers != 1:
            process_map(
                partial(self._build_single_csv, output_dir=output_dir),
                dataset,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Building {dataset_name} CSV files",
            )
        else:
            for event in tqdm(dataset, desc=f"Building {dataset_name} CSV files"):
                self._build_single_csv(event, output_dir=output_dir)

    def _build_single_csv(self, event, output_dir=None):
        """
        This is the user's responsibility. It should take a single event and convert it to CSV.
        """
        raise NotImplementedError

    def _test_pyg_conversion(self):
        """
        This is called after the base class has finished processing the hits, particles and tracks.
        TODO: Implement some tests!
        """
        pass

    def _convert_to_pyg(self):
        for dataset_name in ["trainset", "valset", "testset"]:
            self._build_all_pyg(dataset_name)

    def _build_single_pyg_event(self, event, output_dir=None):
        # Trick to make all workers are using separate CPUs
        # https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
        os.sched_setaffinity(0, range(1000))

        event_id = event["event_id"]

        if os.path.exists(os.path.join(output_dir, f"event{event_id}-graph.pyg")):
            print(f"Graph {event_id} already exists, skipping...")
            return

        particles = pd.read_csv(event["particles"])
        particles = self._add_column_name_prefix(particles, "particle")
        hits = pd.read_csv(event["truth"])
        hits = self._add_column_name_prefix(hits, "hit")
        hits, particles = self._merge_particles_to_hits(hits, particles)
        hits = self._add_handengineered_features(hits)
        hits = self._clean_noise_duplicates(hits)
        tracks, track_features, hits = self._build_true_tracks(hits)
        if tracks.size == 0 or track_features == 0 or hits.size == 0:
            self.log.warning("Found issue in building true tracks... skipping event")
            return

        hits, particles, tracks = self._custom_processing(hits, particles, tracks)
        graph = self._build_graph(hits, tracks, track_features, event_id)
        self._save_pyg_data(graph, output_dir, event_id)

    def _build_all_pyg(self, dataset_name):
        stage_dir = os.path.join(self.config["stage_dir"], dataset_name)
        csv_events = self.get_file_names(
            stage_dir, filename_terms=["particles", "truth"]
        )
        if len(csv_events) == 0:
            warnings.warn(f"No event found in {stage_dir}")
            return
        max_workers = (
            self.config["max_workers"] if "max_workers" in self.config else None
        )

        if max_workers != 1:
            process_map(
                partial(self._build_single_pyg_event, output_dir=stage_dir),
                csv_events,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Building {dataset_name} graphs",
            )
        else:
            for event in tqdm(csv_events, desc=f"Building {dataset_name} graphs"):
                self._build_single_pyg_event(event, output_dir=stage_dir)

    def _build_graph(self, hits, tracks, track_features, event_id):
        """
        Builds a PyG data object from the hits, particles and tracks.
        """

        doRegion = self.config.get("doRegion", False)
        doModuleRegion = self.config.get("doModuleRegion", False)
        if doRegion and doModuleRegion:
            raise ValueError(
                "Multiple regionalization selections. Please select only one regionalization."
            )

        # If both doRegion and doModuleRegion are False (or not specified in the config file)
        # Then the default approach is to build_graph with ALL hits - no regions
        # There are two ways to define a region - either by region_num or by eta/phi position given in the config
        # If region_num, then find the corresponding eta/phi boundaries, else just use the eta/phi boundaries given in the config

        if doRegion:
            (
                eta_region_low,
                eta_region_high,
                phi_region_low,
                phi_region_high,
            ) = self.define_region()

            # For each hit in hits: determine whether it is in region based on z/phi spread
            # For hit inside region: flag true | else: flag false
            in_region = np.full(len(hits["hit_r"]), False)
            d0_low = -0.002  # in m
            d0_high = 0.002  # in m
            z0_low = -0.150  # in m
            z0_high = 0.150  # in m
            q_low = -1.0  # q/pT in 1/GeV
            q_high = 1.0  # q/pT in 1/GeV
            A = 0.3  # in GeV
            rho_low = 1 / (2.0 * A * q_low)
            rho_high = 1 / (2.0 * A * q_high)
            theta_region_low = 2.0 * np.arctan(np.exp(-1.0 * eta_region_low))
            theta_region_high = 2.0 * np.arctan(np.exp(-1.0 * eta_region_high))

            # I want a list of True/False based on whether the z and phi values from z_vals and phi_vals fit inside the z and phi spread vals

            for idx, value in hits["hit_id"].items():
                r = hits["hit_r"][idx] / 1000.0  # convert to m
                z = hits["hit_z"][idx] / 1000.0  # convert to m
                phi = hits["hit_phi"][idx]

                z_low = z0_low + 2.0 * rho_low * 1.0 / np.tan(
                    theta_region_low
                ) * np.arcsin(r / (2.0 * rho_low))
                z_high = z0_high + 2.0 * rho_high * 1.0 / np.tan(
                    theta_region_high
                ) * np.arcsin(r / (2.0 * rho_high))
                phi_low = phi_region_low + np.arcsin(r / (2.0 * rho_low)) + d0_low / r
                phi_high = (
                    phi_region_high + np.arcsin(r / (2.0 * rho_high)) + d0_high / r
                )

                # If the node is inside the new r/phi region, record the node index to use for subgraph construction
                if (z >= z_low) & (z < z_high) & (phi >= phi_low) & (phi < phi_high):
                    in_region[idx] = True

            # only keep hits in the region
            hits = hits[in_region]

            # remapping track edges after hit region selection -> changes hit_id to hit_index and replaces missing hit_id to -1
            tracks = self.remap_track_edges(tracks, hits)

        if doModuleRegion:
            list_modules = self.define_module_region()
            in_region = np.full(len(hits["hit_r"]), False)

            # for each hit, look at the module_id and if it is in the list of modules in the region, then keep the hit
            for idx, value in hits["hit_module_id"].items():
                if value in list_modules:
                    in_region[idx] = True

            # only keep hits in the region
            hits = hits[in_region]

            # remapping track edges after hit region selection -> changes hit_id to hit_index and replaces missing hit_id to -1
            tracks = self.remap_track_edges(tracks, hits)

        graph = Data()
        for feature in set(self.config["feature_sets"]["hit_features"]).intersection(
            set(hits.columns)
        ):
            graph[feature] = torch.from_numpy(hits[feature].values)

        graph.track_edges = torch.from_numpy(tracks)
        for feature in set(self.config["feature_sets"]["track_features"]).intersection(
            set(track_features.keys())
        ):
            graph[feature] = torch.from_numpy(track_features[feature])

        # Add config dictionary to the graph object, so every data has a record of how it was built
        graph.config = [self.config]
        graph.event_id = str(event_id)

        return graph

    def _save_pyg_data(self, graph, output_dir, event_id):
        """
        Save the PyG constructed graph
        """
        if not self.config.get("variable_with_prefix"):
            graph = remove_variable_name_prefix_in_pyg(graph)
        torch.save(graph, os.path.join(output_dir, f"event{event_id}-graph.pyg"))

    @staticmethod
    def calc_eta(r, z):
        theta = np.arctan2(r, z)
        return -1.0 * np.log(np.tan(theta / 2.0))

    def _merge_particles_to_hits(self, hits, particles):
        """
        Merge the particles and hits dataframes, and add some useful columns. These are defined in the config file.
        This is a bit messy, since Athena, ACTS and TrackML have a variety of different conventions and features for particles.
        """

        if "particle_barcode" in particles.columns:
            particles = particles.assign(
                particle_primary=(particles.particle_barcode < 200000).astype(int)
            )

        if "particle_nhits" not in particles.columns:
            hits["particle_nhits"] = hits.groupby("hit_particle_id")[
                "hit_particle_id"
            ].transform("count")

        assert all(
            vertex in particles.columns
            for vertex in ["particle_vx", "particle_vy", "particle_vz"]
        ), "Particles must have vertex information!"
        track_features = self.config["feature_sets"]["track_features"] + [
            "particle_vx",
            "particle_vy",
            "particle_vz",
        ]
        # Get intersection of particle features and the columns in particles
        track_features = [
            track_feature.replace("track_", "") for track_feature in track_features
        ]
        track_features = set(track_features).intersection(set(particles.columns))
        particle_features = list(track_features)

        assert (
            "hit_particle_id" in hits.columns and "particle_id" in particles.columns
        ), "Hits and particles must have a particle_id column!"
        hits = hits.merge(
            particles[particle_features],
            left_on="hit_particle_id",
            right_on="particle_id",
            how="left",
        )

        hits["hit_particle_id"] = hits["hit_particle_id"].fillna(0).astype(int)
        hits["particle_id"] = hits["particle_id"].fillna(0).astype(int)
        hits.loc[hits.hit_particle_id == 0, "particle_nhits"] = -1

        return hits, particles

    @staticmethod
    def _add_column_name_prefix(df, prefix):
        rename_map = {
            column: f"{prefix}_{column}"
            for column in df.columns
            if not column.startswith(prefix + "_")
        }
        df = df.rename(columns=rename_map)
        return df

    @staticmethod
    def _clean_noise_duplicates(hits):
        """
        This handles the case where a hit is assigned to both a particle and noise (e.g. in a ghost spacepoint).
        This is not sensible, so we remove those duplicated noise hits.
        """

        noise_hits = hits[hits.hit_particle_id == 0].drop_duplicates(subset="hit_id")
        signal_hits = hits[hits.hit_particle_id != 0]

        non_duplicate_noise_hits = noise_hits[
            ~noise_hits.hit_id.isin(signal_hits.hit_id)
        ]
        hits = pd.concat([signal_hits, non_duplicate_noise_hits], ignore_index=True)
        # Sort hits by hit_id for ease of processing
        hits = hits.sort_values("hit_id").reset_index(drop=True)

        return hits

    def get_pixel_regions_index(self, hits):
        pixel_regions_index = hits.hit_hardware == "PIXEL"
        return pixel_regions_index

    def _add_handengineered_features(self, hits):
        # Assert that the necessary columns are present in the hits dataframe
        requested_features = self.config["feature_sets"]["hit_features"]

        # Ensure basic geometric features are calculated if configured
        if "hit_r" in requested_features:
            if not all(col in hits.columns for col in ["hit_x", "hit_y"]):
                raise ValueError(
                    "Missing coordinates for calculating 'hit_r'. Required: 'hit_x', 'hit_y'."
                )
            hits["hit_r"] = np.sqrt(hits["hit_x"] ** 2 + hits["hit_y"] ** 2)
        if "hit_phi" in requested_features:
            if not all(col in hits.columns for col in ["hit_x", "hit_y"]):
                raise ValueError(
                    "Missing coordinates for calculating 'phi'. Required: 'hit_x', 'hit_y'."
                )
            hits["hit_phi"] = np.arctan2(hits["hit_y"], hits["hit_x"])
        if "hit_eta" in requested_features:
            if not all(col in hits.columns for col in ["hit_x", "hit_y", "hit_z"]):
                raise ValueError(
                    "Missing coordinates for calculating 'eta'. Required: 'hit_x', 'hit_y', 'hit_z'."
                )
            hits["hit_eta"] = self.calc_eta(hits["hit_r"], hits["hit_z"])

        # Calculate cluster features if the respective coordinates are available
        for i in [1, 2]:  # For each cluster
            for feature in ["r", "phi", "eta"]:
                if f"hit_cluster_{feature}_{i}" in requested_features:
                    required_coords = (
                        ["x", "y"] if feature != "eta" else ["x", "y", "z"]
                    )
                    available_coords = [
                        coord
                        for coord in required_coords
                        if f"hit_cluster_{coord}_{i}" in hits.columns
                    ]
                    if len(available_coords) != len(required_coords):
                        raise ValueError(
                            f"Missing coordinates for calculating 'cluster_{feature}_{i}'. Required: {', '.join(required_coords)}."
                        )

                    # Calculate 'r' and 'phi' if both 'x' and 'y' are available
                    if feature in ["r", "phi"]:
                        hits[f"hit_cluster_r_{i}"] = np.sqrt(
                            hits[f"hit_cluster_x_{i}"] ** 2
                            + hits[f"hit_cluster_y_{i}"] ** 2
                        )
                        hits[f"hit_cluster_phi_{i}"] = np.arctan2(
                            hits[f"hit_cluster_y_{i}"],
                            hits[f"hit_cluster_x_{i}"],
                        )

                    # Calculate 'eta' if 'z' is also available
                    if feature == "eta":
                        hits[f"hit_cluster_eta_{i}"] = self.calc_eta(
                            hits[f"hit_cluster_r_{i}"],
                            hits[f"hit_cluster_z_{i}"],
                        )

        # Apply pixel region adjustments if applicable
        pixel_regions_idx = self.get_pixel_regions_index(hits)
        for feature in ["r", "phi", "eta"]:
            for i in [1, 2]:
                if (
                    f"hit_cluster_{feature}_{i}" in hits.columns
                    and feature in hits.columns
                ):
                    hits.loc[
                        pixel_regions_idx, f"hit_cluster_{feature}_{i}"
                    ] = hits.loc[pixel_regions_idx, feature]

        return hits

    def _build_true_tracks(self, hits):
        assert all(
            col in hits.columns
            for col in [
                "hit_particle_id",
                "hit_id",
                "hit_x",
                "hit_y",
                "hit_z",
                "particle_vx",
                "particle_vy",
                "particle_vz",
            ]
        ), (
            "Need to add (particle_id, hit_id), (x,y,z) and (vx,vy,vz) features to hits"
            " dataframe in custom EventReader class"
        )

        # Sort by increasing distance from production
        hits = hits.assign(
            R=np.sqrt(
                (hits.hit_x - hits.particle_vx) ** 2
                + (hits.hit_y - hits.particle_vy) ** 2
                + (hits.hit_z - hits.particle_vz) ** 2
            )
        )

        signal = hits[(hits.hit_particle_id != 0)]
        signal = signal.sort_values("R").reset_index(drop=False)

        # Group by particle ID
        if "module_columns" not in self.config or self.config["module_columns"] is None:
            module_columns = [
                "hit_barrel_endcap",
                "hit_hardware",
                "hit_layer_disk",
                "hit_eta_module",
                "hit_phi_module",
            ]
        else:
            module_columns = self.config["module_columns"]
            if not self.config.get("variable_with_prefix"):
                for i in range(len(module_columns)):
                    if module_columns[i] in variable_name_prefix_map:
                        module_columns[i] = variable_name_prefix_map[module_columns[i]]

        signal_index_list = (
            signal.groupby(
                ["hit_particle_id"] + module_columns,
                sort=False,
            )["index"]
            .agg(lambda x: list(x))
            .groupby(level=0)
            .agg(lambda x: list(x))
        )

        track_index_edges = []
        for row in signal_index_list.values:
            for i, j in zip(row[:-1], row[1:]):
                track_index_edges.extend(list(product(i, j)))

        track_index_edges = np.array(track_index_edges).T

        # Check against empty track edge indices
        # Can appear in single particle samples
        # where not enough hits in detector
        if track_index_edges.size == 0:
            return np.array([]), np.array([]), np.array([])

        track_edges = hits.hit_id.values[track_index_edges]

        assert (
            hits[hits.hit_id.isin(track_edges.flatten())].hit_particle_id == 0
        ).sum() == 0, "There are hits in the track edges that are noise"

        track_features = self._get_track_features(hits, track_index_edges, track_edges)

        # Remap
        track_edges, track_features, hits = self.remap_edges(
            track_edges, track_features, hits
        )

        return track_edges, track_features, hits

    def _get_track_features(self, hits, track_index_edges, track_edges):
        track_features = {}
        # There may be track_features in the config that are not in the hits dataframe, so loop over the intersection of the two
        track_feature_names = self.config["feature_sets"]["track_features"]
        track_feature_names = [
            track_feature.replace("track_", "") for track_feature in track_feature_names
        ]
        for track_feature in set(track_feature_names).intersection(set(hits.columns)):
            assert (
                hits[track_feature].values[track_index_edges][0]
                == hits[track_feature].values[track_index_edges][1]
            ).all(), f"Track features must be the same for each side of edge: {track_feature}"
            track_features["track_" + track_feature] = hits[track_feature].values[
                track_index_edges[0]
            ]

        if (
            "track_redundant_split_edges"
            in self.config["feature_sets"]["track_features"]
        ):
            track_features[
                "track_redundant_split_edges"
            ] = self._get_redundant_split_edges(track_edges, hits, track_features)

        return track_features

    def _get_redundant_split_edges(self, track_edges, hits, track_features):
        """
        This is a boolean value of each truth track edge. If true, then the edge is leading to a split cluster, which is not the
        "primary" cluster; that is the cluster that is closest to the true particle production point.
        """

        truth_track_df = pd.concat(
            [
                pd.DataFrame(track_edges.T, columns=["hit_id_0", "hit_id_1"]),
                pd.DataFrame(track_features),
            ],
            axis=1,
        )
        hits_unique = hits.drop_duplicates(subset="hit_id")[
            ["hit_id", "hit_module_id", "R"]
        ]
        truth_track_df = (
            truth_track_df[["hit_id_0", "hit_id_1", "track_particle_id"]]
            .merge(hits_unique, left_on="hit_id_0", right_on="hit_id", how="left")
            .drop(columns=["hit_id"])
            .merge(hits_unique, left_on="hit_id_1", right_on="hit_id", how="left")
            .drop(columns=["hit_id"])
        )
        primary_cluster_df = truth_track_df.sort_values(
            by=["hit_module_id_y", "R_x", "R_y"]
        ).drop_duplicates(subset=["hit_module_id_y", "track_particle_id"], keep="first")
        secondary_clusters = ~truth_track_df.index.isin(primary_cluster_df.index)

        return secondary_clusters

    def remap_edges(self, track_edges, track_features, hits):
        """
        Here we do two things:
        1. Remove duplicate hits from the hit list (since a hit is a node and therefore only exists once), and remap the corresponding truth track edge indices
        2. Remove duplicate truth track edges. This is a SMALL simplification for conceptual simplicity,
        but we apply a test to ensure the simplification does not throw away too many duplicate edges.
        """

        unique_hid = np.unique(hits.hit_id)
        hid_mapping = np.zeros(unique_hid.max() + 1).astype(int)
        hid_mapping[unique_hid] = np.arange(len(unique_hid))

        hits = hits.drop_duplicates(subset="hit_id").sort_values("hit_id")
        assert (
            hits.hit_id == unique_hid
        ).all(), "If hit IDs are not sequential, this will mess up graph structure!"

        track_edges = hid_mapping[track_edges]

        # Remove duplicate edges
        unique_track_edges, unique_track_edge_indices = np.unique(
            track_edges, axis=1, return_index=True
        )
        track_features = {
            k: v[unique_track_edge_indices] for k, v in track_features.items()
        }

        # This test imposes a limit to how we simplify the graph: We don't allow shared EDGES (i.e. two different particles can share a hit, but not an edge between the same two hits). We want to ensure these are in a tiny minority
        n_shared_edges = track_edges.shape[1] - unique_track_edges.shape[1]
        if n_shared_edges > 50:
            self.log.warning(
                f"WARNING : high number of shared EDGES ({n_shared_edges} shared edges for {track_edges.shape[1]} edges in total)"
            )

        assert n_shared_edges < 100, "Too many shared edges!"

        return unique_track_edges, track_features, hits

    def get_file_names(self, inputdir, filename_terms: Union[str, list] = None):
        """
        Takes a list of filename terms and searches for all files containing those terms AND a number. Returns the files and numbers.
        For the list of numbers, search for each of the matching terms and files containing that number AND ONLY THAT NUMBER.
        """
        self.log.info("Getting input file names")
        if isinstance(filename_terms, str):
            filename_terms = [filename_terms]
        elif filename_terms is None:
            filename_terms = ["*"]

        all_files_in_template = [
            glob.glob(os.path.join(inputdir, f"*{template}*"))
            for template in filename_terms
        ]
        all_files_in_template = list(chain.from_iterable(all_files_in_template))
        all_event_ids = sorted(
            list({re.findall("[0-9]+", file)[-1] for file in all_files_in_template})
        )

        all_events = []
        for event_id in all_event_ids:
            event = {"event_id": event_id}
            for term in filename_terms:
                if template_file := [
                    file
                    for file in all_files_in_template
                    if term in os.path.basename(file)
                    and re.findall("[0-9]+", file)[-1] == event_id
                ]:
                    event[term] = template_file[0]
                else:
                    print(
                        f"Could not find file for term {term} and event id {event_id}"
                    )
                    break
            else:
                all_events.append(event)

        return all_events

    def _custom_processing(self, hits, particles, tracks):
        """
        This is called after the base class has finished processing the hits, particles and tracks in PyG format
        """
        return hits, particles, tracks

    def _test_csv_conversion(self):
        for data_name in ["trainset", "valset", "testset"]:
            dataset = getattr(self, data_name)
            if dataset is None:
                continue
            self.csv_events = self.get_file_names(
                os.path.join(self.config["stage_dir"], data_name),
                filename_terms=["truth", "particles"],
            )
            assert len(self.csv_events) > 0, (
                "No CSV files found in output directory matching the formats"
                " (event[eventID]-truth.csv, event[eventID]-particles.csv). Please"
                " check that the conversion to CSV was successful."
            )

            # Probe the first event
            event = self.csv_events[0]
            truth = pd.read_csv(event["truth"])
            particles = pd.read_csv(event["particles"])
            assert len(truth) > 0, (
                f"No truth spacepoints found in CSV file in {data_name}. Please check"
                " that the conversion to CSV was successful."
            )
            assert len(particles) > 0, (
                f"No particles found in CSV file in {data_name}. Please check that the"
                " conversion to CSV was successful."
            )

        for dataset1, dataset2 in combinations(["trainset", "valset", "testset"], 2):
            dataset1_files = {
                event["event_id"]
                for event in self.get_file_names(
                    os.path.join(self.config["stage_dir"], dataset1),
                    filename_terms=["truth", "particles"],
                )
            }
            dataset2_files = {
                event["event_id"]
                for event in self.get_file_names(
                    os.path.join(self.config["stage_dir"], dataset2),
                    filename_terms=["truth", "particles"],
                )
            }
            if dataset1_files.intersection(dataset2_files):
                warnings.warn(
                    f"There are overlapping files between the {dataset1} and"
                    f" {dataset2}. You should remove these overlapping files from one"
                    f" of the datasets: {dataset1_files.intersection(dataset2_files)}"
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def remap_track_edges(self, tracks, hits):
        hit_ids_0, hit_ids_1 = [], []
        hit_index_0, hit_index_1 = [], []
        track_index_0, track_index_1 = [], []

        for i, element in enumerate(hits["hit_id"]):
            matches_0 = np.where(tracks[0] == element)[0]
            matches_1 = np.where(tracks[1] == element)[0]
            for match_0 in matches_0:
                hit_ids_0.append(element)
                hit_index_0.append(i)
                track_index_0.append(match_0)
            for match_1 in matches_1:
                hit_ids_1.append(element)
                hit_index_1.append(i)
                track_index_1.append(match_1)

        track_edges_0 = np.full(len(tracks[0]), -1)
        track_edges_0[track_index_0] = hit_index_0

        track_edges_1 = np.full(len(tracks[1]), -1)
        track_edges_1[track_index_1] = hit_index_1

        return np.asarray([track_edges_0, track_edges_1])

    def define_region(self):
        eta_region_low, eta_region_high, phi_region_low, phi_region_high = (
            0.0,
            0.0,
            0.0,
            0.0,
        )

        if set(["region", "region_lookup_file_path"]).issubset(self.config):
            df = pd.read_csv(self.config["region_lookup_file_path"])
            region = df[(df["region"] == int(self.config["region"]))]
            if len(region["region"]) == 1:
                eta_region_low = region["eta_region_low"]
                eta_region_high = region["eta_region_high"]
                phi_region_low = region["phi_region_low"]
                phi_region_high = region["phi_region_high"]
            else:
                raise ValueError(
                    "Region provided was not in the list of accepted regions in the region lookup file. Please correct config file."
                )
        elif set(
            ["eta_region_low", "eta_region_high", "phi_region_low", "phi_region_high"]
        ).issubset(self.config):
            eta_region_low = self.config["eta_region_low"]
            eta_region_high = self.config["eta_region_high"]
            phi_region_low = self.config["phi_region_low"]
            phi_region_high = self.config["phi_region_high"]
        elif "region" in self.config:
            raise ValueError(
                "Region was provided but no lookup file was given. Please include lookup file or instead pass eta/phi coordinates of region"
            )
        else:
            raise ValueError(
                "No region was defined and provided in config. Either pass eta/phi boundaries or a region number and lookup file"
            )

        return eta_region_low, eta_region_high, phi_region_low, phi_region_high

    def define_module_region(self):
        list_modules = list()

        if set(["region", "module_region_lookup_path"]).issubset(self.config):
            df = pd.read_csv(self.config["module_region_lookup_path"])
            region = df[(df["region"] == int(self.config["region"]))]
            if len(region["region"]) == 1:
                list_modules = list(
                    map(
                        int,
                        region["module_id_list"]
                        .values[0]
                        .replace("[", "")
                        .replace("]", "")
                        .split(", "),
                    )
                )
            else:
                raise ValueError(
                    "Region provided was not in the list of accepted regions in the region lookup file. Please correct config file."
                )
        elif set(
            [
                "eta_region_low",
                "eta_region_high",
                "phi_region_low",
                "phi_region_high",
                "module_region_lookup_path",
            ]
        ).issubset(self.config):
            df = pd.read_csv(self.config["module_region_lookup_path"])
            region = df[
                (df["eta_region_low"] == int(self.config["eta_region_low"]))
                & (df["eta_region_high"] == int(self.config["eta_region_high"]))(
                    df["phi_region_low"] == int(self.config["phi_region_low"])
                )(df["phi_region_high"] == int(self.config["phi_region_eta"]))
            ]
            if len(region["region"]) == 1:
                list_modules = list(
                    map(
                        int,
                        region["module_id_list"]
                        .values[0]
                        .replace("[", "")
                        .replace("]", "")
                        .split(", "),
                    )
                )
            else:
                raise ValueError(
                    "Eta/Phi boundaries provided was not in the list of accepted regions in the region lookup file. Please correct config file."
                )
        else:
            raise ValueError(
                "No region was defined or no module_region_lookup file was provided. Please correct config file."
            )

        return list_modules
