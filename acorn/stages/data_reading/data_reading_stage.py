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
        max_workers = self.config["max_workers"] if "max_workers" in self.config else 1
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
        particles = particles.rename(columns={"eta": "eta_particle"})
        hits = pd.read_csv(event["truth"])
        hits, particles = self._merge_particles_to_hits(hits, particles)
        hits = self._add_handengineered_features(hits)
        hits = self._clean_noise_duplicates(hits)
        tracks, track_features, hits = self._build_true_tracks(hits)
        hits, particles, tracks = self._custom_processing(hits, particles, tracks)
        graph = self._build_graph(hits, tracks, track_features, event_id)
        self._save_pyg_data(graph, output_dir, event_id)

    def _build_all_pyg(self, dataset_name):
        dataset = getattr(self, dataset_name)
        if dataset is None:
            return
        stage_dir = os.path.join(self.config["stage_dir"], dataset_name)
        csv_events = self.get_file_names(
            stage_dir, filename_terms=["particles", "truth"]
        )
        assert len(csv_events) > 0, "No CSV files found!"
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

        if "barcode" in particles.columns:
            particles = particles.assign(
                primary=(particles.barcode < 200000).astype(int)
            )

        if "nhits" not in particles.columns:
            hits["nhits"] = hits.groupby("particle_id")["particle_id"].transform(
                "count"
            )

        assert all(
            vertex in particles.columns for vertex in ["vx", "vy", "vz"]
        ), "Particles must have vertex information!"
        particle_features = self.config["feature_sets"]["track_features"] + [
            "vx",
            "vy",
            "vz",
        ]

        # Get intersection of particle features and the columns in particles
        particle_features = [
            feature for feature in particle_features if feature in particles.columns
        ]

        assert (
            "particle_id" in hits.columns and "particle_id" in particles.columns
        ), "Hits and particles must have a particle_id column!"
        hits = hits.merge(
            particles[particle_features],
            on="particle_id",
            how="left",
        )

        hits["particle_id"] = hits["particle_id"].fillna(0).astype(int)
        hits.loc[hits.particle_id == 0, "nhits"] = -1

        return hits, particles

    @staticmethod
    def _clean_noise_duplicates(hits):
        """
        This handles the case where a hit is assigned to both a particle and noise (e.g. in a ghost spacepoint).
        This is not sensible, so we remove those duplicated noise hits.
        """

        noise_hits = hits[hits.particle_id == 0].drop_duplicates(subset="hit_id")
        signal_hits = hits[hits.particle_id != 0]

        non_duplicate_noise_hits = noise_hits[
            ~noise_hits.hit_id.isin(signal_hits.hit_id)
        ]
        hits = pd.concat([signal_hits, non_duplicate_noise_hits], ignore_index=True)
        # Sort hits by hit_id for ease of processing
        hits = hits.sort_values("hit_id").reset_index(drop=True)

        return hits

    def get_pixel_regions_index(self, hits):
        pixel_regions_index = pd.Index([])
        for region_id, desc in self.config["region_labels"].items():
            if desc["hardware"] == "PIXEL":
                pixel_regions_index = pixel_regions_index.append(
                    hits.index[hits.region == region_id]
                )
        return pixel_regions_index

    def _add_handengineered_features(self, hits):
        pixel_regions_idx = self.get_pixel_regions_index(hits)
        assert all(
            col in hits.columns
            for col in [
                "x",
                "y",
                "z",
                "cluster_x_1",
                "cluster_y_1",
                "cluster_z_1",
                "cluster_x_2",
                "cluster_y_2",
                "cluster_z_2",
            ]
        ), "Need to add (x,y,z) features"
        if "r" in self.config["feature_sets"]["hit_features"]:
            r = np.sqrt(hits.x**2 + hits.y**2)
            hits = hits.assign(r=r)
        if "phi" in self.config["feature_sets"]["hit_features"]:
            phi = np.arctan2(hits.y, hits.x)
            hits = hits.assign(phi=phi)
        if "eta" in self.config["feature_sets"]["hit_features"]:
            eta = self.calc_eta(
                r, hits.z
            )  # TODO check if r is defined (same for clusters, below)
            hits = hits.assign(eta=eta)
        if "cluster_r_1" in self.config["feature_sets"]["hit_features"]:
            cluster_r_1 = np.sqrt(hits.cluster_x_1**2 + hits.cluster_y_1**2)
            cluster_r_1.loc[pixel_regions_idx] = r.loc[pixel_regions_idx]
            hits = hits.assign(cluster_r_1=cluster_r_1)
        if "cluster_phi_1" in self.config["feature_sets"]["hit_features"]:
            cluster_phi_1 = np.arctan2(hits.cluster_y_1, hits.cluster_x_1)
            cluster_phi_1.loc[pixel_regions_idx] = phi.loc[pixel_regions_idx]
            hits = hits.assign(cluster_phi_1=cluster_phi_1)
        if "cluster_eta_1" in self.config["feature_sets"]["hit_features"]:
            cluster_eta_1 = self.calc_eta(cluster_r_1, hits.cluster_z_1)
            cluster_eta_1.loc[pixel_regions_idx] = eta.loc[pixel_regions_idx]
            hits = hits.assign(cluster_eta_1=cluster_eta_1)
        if "cluster_r_2" in self.config["feature_sets"]["hit_features"]:
            cluster_r_2 = np.sqrt(hits.cluster_x_2**2 + hits.cluster_y_2**2)
            cluster_r_2.loc[pixel_regions_idx] = r.loc[pixel_regions_idx]
            hits = hits.assign(cluster_r_2=cluster_r_2)
        if "cluster_phi_2" in self.config["feature_sets"]["hit_features"]:
            cluster_phi_2 = np.arctan2(hits.cluster_y_2, hits.cluster_x_2)
            cluster_phi_2.loc[pixel_regions_idx] = phi.loc[pixel_regions_idx]
            hits = hits.assign(cluster_phi_2=cluster_phi_2)
        if "cluster_eta_2" in self.config["feature_sets"]["hit_features"]:
            cluster_eta_2 = self.calc_eta(cluster_r_2, hits.cluster_z_2)
            cluster_eta_2.loc[pixel_regions_idx] = eta.loc[pixel_regions_idx]
            hits = hits.assign(cluster_eta_2=cluster_eta_2)

        return hits

    def _build_true_tracks(self, hits):
        assert all(
            col in hits.columns
            for col in ["particle_id", "hit_id", "x", "y", "z", "vx", "vy", "vz"]
        ), (
            "Need to add (particle_id, hit_id), (x,y,z) and (vx,vy,vz) features to hits"
            " dataframe in custom EventReader class"
        )

        # Sort by increasing distance from production
        hits = hits.assign(
            R=np.sqrt(
                (hits.x - hits.vx) ** 2
                + (hits.y - hits.vy) ** 2
                + (hits.z - hits.vz) ** 2
            )
        )

        signal = hits[(hits.particle_id != 0)]
        signal = signal.sort_values("R").reset_index(drop=False)

        # Group by particle ID
        if "module_columns" not in self.config or self.config["module_columns"] is None:
            module_columns = [
                "barrel_endcap",
                "hardware",
                "layer_disk",
                "eta_module",
                "phi_module",
            ]
        else:
            module_columns = self.config["module_columns"]

        signal_index_list = (
            signal.groupby(
                ["particle_id"] + module_columns,
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
        track_edges = hits.hit_id.values[track_index_edges]

        assert (
            hits[hits.hit_id.isin(track_edges.flatten())].particle_id == 0
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
        for track_feature in set(
            self.config["feature_sets"]["track_features"]
        ).intersection(set(hits.columns)):
            assert (
                hits[track_feature].values[track_index_edges][0]
                == hits[track_feature].values[track_index_edges][1]
            ).all(), f"Track features must be the same for each side of edge: {track_feature}"
            track_features[track_feature] = hits[track_feature].values[
                track_index_edges[0]
            ]

        if "redundant_split_edges" in self.config["feature_sets"]["track_features"]:
            track_features["redundant_split_edges"] = self._get_redundant_split_edges(
                track_edges, hits, track_features
            )

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
            ["hit_id", "module_id", "R"]
        ]
        truth_track_df = (
            truth_track_df[["hit_id_0", "hit_id_1", "particle_id"]]
            .merge(hits_unique, left_on="hit_id_0", right_on="hit_id", how="left")
            .drop(columns=["hit_id"])
            .merge(hits_unique, left_on="hit_id_1", right_on="hit_id", how="left")
            .drop(columns=["hit_id"])
        )
        primary_cluster_df = truth_track_df.sort_values(
            by=["module_id_y", "R_x", "R_y"]
        ).drop_duplicates(subset=["module_id_y", "particle_id"], keep="first")
        secondary_clusters = ~truth_track_df.index.isin(primary_cluster_df.index)

        return secondary_clusters

    @staticmethod
    def remap_edges(track_edges, track_features, hits):
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
        assert n_shared_edges < 50, "The number of shared EDGES is unusually high!"

        return unique_track_edges, track_features, hits

    def get_file_names(self, inputdir, filename_terms: Union[str, list] = None):
        """
        Takes a list of filename terms and searches for all files containing those terms AND a number. Returns the files and numbers.
        For the list of numbers, search for each of the matching terms and files containing that number AND ONLY THAT NUMBER.
        """

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
