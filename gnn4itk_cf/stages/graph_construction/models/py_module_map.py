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

# 3rd party imports
import os
import logging
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import time

try:
    import cudf
except ImportError:
    logging.warning("cuDF not found, using pandas instead")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local imports
from ..graph_construction_stage import GraphConstructionStage
from . import utils


class PyModuleMap(GraphConstructionStage):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
        """
        self.hparams = hparams
        self.use_pyg = True
        self.use_csv = True
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def to(self, device):
        return self

    def load_module_map(self):
        """
        Load in the module map dataframe. Should be in CSV format.
        """

        names = [
            "mid_1",
            "mid_2",
            "mid_3",
            "occurence",
            "z0max_12",
            "z0min_12",
            "dphimax_12",
            "dphimin_12",
            "phiSlopemax_12",
            "phiSlopemin_12",
            "detamax_12",
            "detamin_12",
            "z0max_23",
            "z0min_23",
            "dphimax_23",
            "dphimin_23",
            "phiSlopemax_23",
            "phiSlopemin_23",
            "detamax_23",
            "detamin_23",
            "diff_dzdr_max",
            "diff_dzdr_min",
            "diff_dydx_max",
            "diff_dydx_min",
        ]
        self.MM = pd.read_csv(
            self.hparams["module_map_path"],
            names=names,
            header=None,
            delim_whitespace=True,
        )
        self.MM_1, self.MM_2, self.MM_triplet = self.get_module_features(self.MM)

        if self.gpu_available:
            self.MM_1 = cudf.from_pandas(self.MM_1)
            self.MM_2 = cudf.from_pandas(self.MM_2)
            self.MM_triplet = cudf.from_pandas(self.MM_triplet)

    def build_graphs(self, dataset, data_name):
        """
        Build the graphs for the data.
        """
        self.load_module_map()
        output_dir = os.path.join(self.hparams["stage_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Building graphs for {data_name}")

        for graph, _, truth in tqdm(dataset):
            if graph is None:
                continue
            if os.path.exists(os.path.join(output_dir, f"event{graph.event_id}.pyg")):
                continue

            # Get timing
            start_time = time.time()

            hits = (
                cudf.from_pandas(truth.copy()) if self.gpu_available else truth.copy()
            )

            hits = self.get_hit_features(hits)
            merged_hits_1 = hits.merge(
                self.MM_1, how="inner", left_on="mid", right_on="mid_2"
            ).to_pandas()  # .drop(columns="mid")
            merged_hits_2 = hits.merge(
                self.MM_2, how="inner", left_on="mid", right_on="mid_2"
            ).to_pandas()  # .drop(columns="mid")

            # print(f"Time to merge: {time.time() - start_time}")

            doublet_edges_1 = self.get_doublet_edges(
                hits, merged_hits_1, "mid", "mid_1", first_doublet=True
            )
            doublet_edges_1 = doublet_edges_1[
                [
                    "hid_1",
                    "hid_2",
                    "mid_1",
                    "mid_2",
                    "mid_3",
                    "x_1",
                    "x_2",
                    "y_1",
                    "y_2",
                    "r_1",
                    "r_2",
                    "z_1",
                    "z_2",
                ]
            ]

            # print(f"Time to get doublet edges: {time.time() - start_time}")

            doublet_edges_2 = self.get_doublet_edges(
                hits, merged_hits_2, "mid", "mid_3", first_doublet=False
            )
            doublet_edges_2 = doublet_edges_2[
                [
                    "hid_1",
                    "hid_2",
                    "mid_1",
                    "mid_2",
                    "mid_3",
                    "x_2",
                    "y_2",
                    "r_2",
                    "z_2",
                ]
            ].rename(
                columns={
                    "hid_1": "hid_2",
                    "hid_2": "hid_3",
                    "x_2": "x_3",
                    "y_2": "y_3",
                    "r_2": "r_3",
                    "z_2": "z_3",
                }
            )
            doublet_edges_2 = doublet_edges_2.merge(
                self.MM_triplet,
                how="inner",
                left_on=["mid_1", "mid_2", "mid_3"],
                right_on=["mid_1", "mid_2", "mid_3"],
            )

            # print(f"Time to get doublet edges 2: {time.time() - start_time}")

            triplet_edges = doublet_edges_1.merge(
                doublet_edges_2,
                how="inner",
                left_on=["mid_1", "hid_2", "mid_3"],
                right_on=["mid_1", "hid_2", "mid_3"],
                suffixes=("_1", "_2"),
            )
            triplet_edges = self.apply_triplet_cuts(triplet_edges)

            # print(f"Time to get triplet edges: {time.time() - start_time}")

            if self.gpu_available:
                doublet_edges = cudf.concat(
                    [
                        triplet_edges[["hid_1", "hid_2"]],
                        triplet_edges[["hid_2", "hid_3"]].rename(
                            columns={"hid_2": "hid_1", "hid_3": "hid_2"}
                        ),
                    ]
                )
                doublet_edges = doublet_edges.to_pandas()
            else:
                doublet_edges = pd.concat(
                    [
                        triplet_edges[["hid_1", "hid_2"]],
                        triplet_edges[["hid_2", "hid_3"]].rename(
                            columns={"hid_2": "hid_1", "hid_3": "hid_2"}
                        ),
                    ]
                )

            # print(f"Time to concat: {time.time() - start_time}")

            doublet_edges = doublet_edges.drop_duplicates()
            graph.edge_index = torch.tensor(doublet_edges.values.T, dtype=torch.long)
            y, truth_map = utils.graph_intersection(
                graph.edge_index.to(device),
                graph.track_edges.to(device),
                return_y_pred=True,
                return_truth_to_pred=True,
            )
            graph.y = y.cpu()
            graph.truth_map = truth_map.cpu()

            # print(f"Time to get y: {time.time() - start_time}")

            # TODO: Graph name file??
            torch.save(graph, os.path.join(output_dir, f"event{graph.event_id}.pyg"))
            # print(f"Time to save graph: {time.time() - start_time}")

    def get_hit_features(self, hits):
        hits = hits.rename(columns={"hit_id": "hid", "module_id": "mid"})
        hits["r"] = np.sqrt(hits.x**2 + hits.y**2)
        hits["z"] = hits.z
        hits["eta"] = self.calc_eta(hits.r, hits.z)
        hits["phi"] = np.arctan2(hits.y, hits.x)

        # Drop all other columns
        hits = hits[["hid", "mid", "r", "x", "y", "z", "eta", "phi"]]
        return hits

    @staticmethod
    def calc_eta(r, z):
        theta = np.arctan2(r, z)
        return -1.0 * np.log(np.tan(theta / 2.0))

    def get_doublet_edges(
        self, hits, merged_hits, left_merge, right_merge, first_doublet=True
    ):
        """
        Get the doublet edges for the merged hits.
        """
        suffixes = ("_1", "_2") if first_doublet else ("_2", "_1")
        doublet_edges = []

        # Print memory usage with cudf
        # print(f"Memory usage 1: {merged_hits.memory_usage(deep=True).sum() / 1e9} GB")

        for batch_idx in np.arange(0, merged_hits.shape[0], self.batch_size):
            start_idx, end_idx = batch_idx, batch_idx + self.batch_size
            # subset_edges = hits.merge(merged_hits.iloc[start_idx:end_idx], how="inner", left_on=left_merge, right_on=right_merge, suffixes=suffixes)
            # print(f"Memory usage a: {subset_edges.memory_usage(deep=True).sum() / 1e9} GB")
            subset_merged = cudf.from_pandas(merged_hits.iloc[start_idx:end_idx])
            subset_edges = hits.merge(
                subset_merged,
                how="inner",
                left_on=left_merge,
                right_on=right_merge,
                suffixes=suffixes,
            )
            subset_edges = self.apply_doublet_cuts(subset_edges, first_doublet)
            # print(f"Memory usage b: {subset_edges.memory_usage(deep=True).sum() / 1e9} GB")
            doublet_edges.append(subset_edges)
            # print(f"Memory usage c: {subset_edges.memory_usage(deep=True).sum() / 1e9} GB")
            # Delete everything
            del subset_edges

        # print(f"Memory usage 2: {merged_hits.memory_usage(deep=True).sum() / 1e9} GB")

        return (
            cudf.concat(doublet_edges)
            if isinstance(doublet_edges[0], cudf.DataFrame)
            else pd.concat(doublet_edges)
        )

    @staticmethod
    def get_module_features(MM):
        cols_1 = [
            "mid_1",
            "mid_2",
            "mid_3",
            "z0max_12",
            "z0min_12",
            "dphimax_12",
            "dphimin_12",
            "phiSlopemax_12",
            "phiSlopemin_12",
            "detamax_12",
            "detamin_12",
        ]
        MM_ids_1 = MM[cols_1]

        cols_2 = [
            "mid_1",
            "mid_2",
            "mid_3",
            "z0max_23",
            "z0min_23",
            "dphimax_23",
            "dphimin_23",
            "phiSlopemax_23",
            "phiSlopemin_23",
            "detamax_23",
            "detamin_23",
        ]
        MM_ids_2 = MM[cols_2]

        cols_3 = [
            "mid_1",
            "mid_2",
            "mid_3",
            "diff_dzdr_max",
            "diff_dzdr_min",
            "diff_dydx_max",
            "diff_dydx_min",
        ]
        MM_triplet = MM[cols_3]

        return MM_ids_1, MM_ids_2, MM_triplet

    def get_deltas(self, hits):
        delta_eta = hits.eta_1 - hits.eta_2
        delta_z = hits.z_2 - hits.z_1
        delta_r = hits.r_2 - hits.r_1

        delta_phi = hits.phi_2 - hits.phi_1
        delta_phi = self.reset_angle(delta_phi)

        z0 = hits.z_1 - (hits.r_1 * delta_z / delta_r)

        phi_slope = delta_phi / delta_r

        return delta_eta, delta_phi, z0, phi_slope

    @staticmethod
    def reset_angle(angles):
        angles[angles > np.pi] = angles[angles > np.pi] - 2 * np.pi
        angles[angles < -np.pi] = angles[angles < -np.pi] + 2 * np.pi

        return angles

    def apply_doublet_cuts(self, hits, first_doublet=True):
        suffix = "12" if first_doublet else "23"

        delta_eta = hits.eta_1 - hits.eta_2
        eta_mask = (delta_eta < hits[f"detamax_{suffix}"]) & (
            delta_eta > hits[f"detamin_{suffix}"]
        )

        hits = hits[eta_mask]

        delta_z = hits.z_2 - hits.z_1
        delta_r = hits.r_2 - hits.r_1
        z0 = hits.z_1 - (hits.r_1 * delta_z / delta_r)

        r_z_mask = (z0 < hits[f"z0max_{suffix}"]) & (z0 > hits[f"z0min_{suffix}"])
        hits = hits[r_z_mask]

        delta_phi = hits.phi_2 - hits.phi_1
        delta_phi = self.reset_angle(delta_phi)

        phi_mask = (delta_phi < hits[f"dphimax_{suffix}"]) & (
            delta_phi > hits[f"dphimin_{suffix}"]
        )
        hits = hits[phi_mask]

        delta_phi = hits.phi_2 - hits.phi_1
        delta_phi = self.reset_angle(delta_phi)
        delta_r = hits.r_2 - hits.r_1
        phi_slope = delta_phi / delta_r
        phi_slope_mask = (phi_slope < hits[f"phiSlopemax_{suffix}"]) & (
            phi_slope > hits[f"phiSlopemin_{suffix}"]
        )

        hits = hits[phi_slope_mask]

        return hits

    @staticmethod
    def apply_triplet_cuts(triplet_edges):
        dy_12 = triplet_edges.y_2 - triplet_edges.y_1
        dy_23 = triplet_edges.y_2 - triplet_edges.y_3
        dx_12 = triplet_edges.x_1 - triplet_edges.x_2
        dx_23 = triplet_edges.x_2 - triplet_edges.x_3
        diff_dydx = dy_12 / dx_12 - dy_23 / dx_23
        diff_dydx[(dx_12 == 0) & (dx_23 == 0)] = 0

        dydz_mask = (diff_dydx < triplet_edges.diff_dydx_max) & (
            diff_dydx > triplet_edges.diff_dydx_min
        )
        triplet_edges = triplet_edges[dydz_mask]

        dz_12 = triplet_edges.z_2 - triplet_edges.z_1
        dz_23 = triplet_edges.z_3 - triplet_edges.z_2
        dr_12 = triplet_edges.r_2 - triplet_edges.r_1
        dr_23 = triplet_edges.r_3 - triplet_edges.r_2

        diff_dzdr = dz_12 / dr_12 - dz_23 / dr_23
        diff_dzdr[(dr_12 == 0) & (dr_23 == 0)] = 0

        dzdr_mask = (diff_dzdr < triplet_edges.diff_dzdr_max) & (
            diff_dzdr > triplet_edges.diff_dzdr_min
        )
        triplet_edges = triplet_edges[dzdr_mask]

        return triplet_edges

    # TODO: Refactor the apply_triplet_cuts function to use this function
    @staticmethod
    def get_triplet_mask(triplet_edges, feats_A, feats_B, feats_min_max):
        dA_12 = triplet_edges[feats_A[1]] - triplet_edges[feats_A[0]]
        dA_23 = triplet_edges[feats_A[1]] - triplet_edges[feats_A[2]]
        dB_12 = triplet_edges[feats_B[1]] - triplet_edges[feats_B[0]]
        dB_23 = triplet_edges[feats_B[1]] - triplet_edges[feats_B[2]]
        diff_dAdB = dA_12 / dB_12 - dA_23 / dB_23
        diff_dAdB[(dB_12 == 0) & (dB_23 == 0)] = 0

        return (diff_dAdB < triplet_edges[feats_min_max[1]]) & (
            diff_dAdB > triplet_edges[feats_min_max[0]]
        )

    @property
    def batch_size(self):
        return self.hparams["batch_size"] if "batch_size" in self.hparams else int(1e6)
