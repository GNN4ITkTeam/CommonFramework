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

try:
    import cudf
except ImportError:
    logging.warning("cuDF not found, using pandas instead")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local imports
from ..graph_construction_stage import GraphConstructionStage
from . import utils
from acorn.utils.loading_utils import remove_variable_name_prefix_in_pyg


class PyModuleMap(GraphConstructionStage):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
        """
        self.hparams = hparams
        self.use_csv = True
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Logging config
        self.log = logging.getLogger("PyModuleMap")
        log_level = (
            self.hparams["log_level"].upper()
            if "log_level" in self.hparams
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

    def to(self, device):
        return self

    def load_module_map(self):
        """
        Load in the module map dataframe. Should be in CSV format.
        """

        # Module Map config
        self.module_map = self.hparams.get("module_map", None)

        if not self.module_map:
            raise ValueError("Missing 'module_map' settings in yaml config file")

        if self.module_map.get("method") == "minmax":
            self.log.info("Using module map with min/max selection")
        elif self.module_map.get("method") == "meanrms":
            self.log.info(
                f"Using module map with mean +/- {self.module_map.get('threshold_factor_rms')}*rms selection"
            )
        elif self.module_map.get("method") == "hybrid":
            self.log.info(
                f"Using module map hybrid method with with mean +/- {self.module_map.get('threshold_factor_rms')}*rms selection for occurence > {self.module_map.get('occurence_threshold')}"
            )
        else:
            raise ValueError(
                f"Unsupported module map method {self.module_map.get('method')}, shoudl be either 'minmax', 'meanrms'or 'hybrid'"
            )

        module_map_filename = self.module_map.get("path")

        self.log.info(f"Loading module map from {module_map_filename}")

        if module_map_filename.endswith(".root"):
            self.MM = utils.load_module_map_uproot(module_map_filename)
        elif module_map_filename.endswith(".csv") or module_map_filename.endswith(
            ".txt"
        ):
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

            if self.module_map.get("method") != "minmax":
                raise ValueError(
                    "Unsupported method with module map in csv, please use a root file."
                )

            self.MM = pd.read_csv(
                module_map_filename,
                names=names,
                header=None,
                delim_whitespace=True,
            )

        else:
            raise ValueError(
                "Unsupported module map file extension, should be .root, .txt or .csv"
            )

        self.MM_1, self.MM_2, self.MM_triplet = self.get_module_features()

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
        self.log.info(f"Building graphs for {data_name}")

        for graph, _, truth in tqdm(dataset):
            if graph is None:
                continue
            if os.path.exists(os.path.join(output_dir, f"event{graph.event_id}.pyg")):
                print(f"Graph {graph.event_id} already exists, skipping...")
                continue

            graph = self.build_graph(graph, truth)
            if not self.hparams.get("variable_with_prefix"):
                graph = remove_variable_name_prefix_in_pyg(graph)
            torch.save(graph, os.path.join(output_dir, f"event{graph.event_id}.pyg"))

    def build_graph(self, graph, truth):
        """
        Build the graph for the data.
        """

        hits = cudf.from_pandas(
            pd.DataFrame(
                {
                    "hid": graph.hit_id,
                    "mid": graph.hit_module_id,
                    "r": graph.hit_r,
                    "x": graph.hit_x,
                    "y": graph.hit_y,
                    "z": graph.hit_z,
                    "eta": graph.hit_eta,
                    "phi": graph.hit_phi,
                }
            )
        )

        merged_hits_1 = hits.merge(
            self.MM_1, how="inner", left_on="mid", right_on="mid_2"
        ).to_pandas()  # .drop(columns="mid")
        merged_hits_2 = hits.merge(
            self.MM_2, how="inner", left_on="mid", right_on="mid_2"
        ).to_pandas()  # .drop(columns="mid")

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

        triplet_edges = doublet_edges_1.merge(
            doublet_edges_2,
            how="inner",
            left_on=["mid_1", "hid_2", "mid_3"],
            right_on=["mid_1", "hid_2", "mid_3"],
            suffixes=("_1", "_2"),
        )
        triplet_edges = self.apply_triplet_cuts(triplet_edges)

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

        doublet_edges = doublet_edges.drop_duplicates()
        graph.edge_index = torch.tensor(doublet_edges.values.T, dtype=torch.long)
        if not torch.equal(
            graph.hit_id, torch.arange(graph.hit_id.size(0), device=graph.hit_id.device)
        ):
            # We need to re-index the edge index since there are missing nodes
            self.reindex_edge_index(graph)
        y, truth_map = utils.graph_intersection(
            graph.edge_index.to(device),
            graph.track_edges.to(device),
            return_y_pred=True,
            return_truth_to_pred=True,
        )
        graph.edge_y = y.cpu()
        graph.track_to_edge_map = truth_map.cpu()

        return graph

    def reindex_edge_index(self, graph):
        """
        Reindex the edge index to account for missing nodes.
        Missing nodes are likely due to a hard cut, and are not a problem.
        """
        hit_id_to_vector_index = torch.full(
            (graph.hit_id.max() + 1,), -1, dtype=torch.long
        )
        hit_id_to_vector_index[graph.hit_id] = torch.arange(graph.hit_id.size(0))
        graph.edge_index = hit_id_to_vector_index[graph.edge_index]

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
        # theta = np.arctan2(r, z)
        # return -1.0 * np.log(np.tan(theta / 2.0))

        # Aligned to the way it is computed in ModuleMapGraph, to avoid O(1e-13) differences in double
        r3 = np.sqrt(r**2 + z**2)
        theta = 0.5 * np.arccos(z / r3)
        return -np.log(np.tan(theta))

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

    def get_module_features(self):
        """Make doublets 12, 23 and triplets datasets"""

        if self.module_map.get("method") == "minmax":
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

            cols_3 = [
                "mid_1",
                "mid_2",
                "mid_3",
                "diff_dzdr_max",
                "diff_dzdr_min",
                "diff_dydx_max",
                "diff_dydx_min",
            ]
        else:
            features_doublet = ["z0", "dphi", "phiSlope", "deta"]
            features_triplet = ["diff_dydx", "diff_dzdr"]

            cols_1 = [f"{feat}_12_mean" for feat in features_doublet]
            cols_1 += [f"{feat}_12_rms" for feat in features_doublet]
            cols_1 += [f"{feat}min_12" for feat in features_doublet]
            cols_1 += [f"{feat}max_12" for feat in features_doublet]

            cols_2 = [f"{feat}_23_mean" for feat in features_doublet]
            cols_2 += [f"{feat}_23_rms" for feat in features_doublet]
            cols_2 += [f"{feat}min_23" for feat in features_doublet]
            cols_2 += [f"{feat}max_23" for feat in features_doublet]

            cols_3 = [f"{feat}_mean" for feat in features_triplet]
            cols_3 += [f"{feat}_rms" for feat in features_triplet]
            cols_3 += [f"{feat}_min" for feat in features_triplet]
            cols_3 += [f"{feat}_max" for feat in features_triplet]

            mids = ["mid_1", "mid_2", "mid_3"]

            cols_1 += mids
            cols_2 += mids
            cols_3 += mids

            if self.module_map.get("method") == "hybrid":
                cols_1 += ["occurence"]
                cols_2 += ["occurence"]
                cols_3 += ["occurence"]

        MM_ids_1 = self.MM[cols_1]
        MM_ids_2 = self.MM[cols_2]
        MM_triplet = self.MM[cols_3]

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

        method = self.module_map.get("method")
        tolerance = self.module_map.get("tolerance", 0.0)
        threshold_factor_rms = self.module_map.get("threshold_factor_rms", -1)
        occurence_threshold = self.module_map.get("occurence_threshold", -1)

        # Delta eta
        # ---------
        delta_eta = hits.eta_1 - hits.eta_2
        eta_mask = get_doublet_mask(
            method,
            delta_eta,
            "deta",
            hits,
            suffix,
            tolerance,
            threshold_factor_rms,
            occurence_threshold,
        )
        hits = hits[eta_mask]

        # z0
        # --
        delta_z = hits.z_2 - hits.z_1
        delta_r = hits.r_2 - hits.r_1
        z0 = hits.z_1 - (hits.r_1 * delta_z / delta_r)
        z0[delta_r == 0] = 0
        z0_mask = get_doublet_mask(
            method,
            z0,
            "z0",
            hits,
            suffix,
            tolerance,
            threshold_factor_rms,
            occurence_threshold,
        )
        hits = hits[z0_mask]

        # Delta phi
        # ---------
        delta_phi = hits.phi_2 - hits.phi_1
        delta_phi = self.reset_angle(delta_phi)
        phi_mask = get_doublet_mask(
            method,
            delta_phi,
            "dphi",
            hits,
            suffix,
            tolerance,
            threshold_factor_rms,
            occurence_threshold,
        )
        hits = hits[phi_mask]

        # Phi slope
        # ---------
        delta_phi = hits.phi_2 - hits.phi_1
        delta_phi = self.reset_angle(delta_phi)
        delta_r = hits.r_2 - hits.r_1
        phi_slope = delta_phi / delta_r
        phi_slope[delta_r == 0] = 0
        phi_slope_mask = get_doublet_mask(
            method,
            phi_slope,
            "phiSlope",
            hits,
            suffix,
            tolerance,
            threshold_factor_rms,
            occurence_threshold,
        )
        hits = hits[phi_slope_mask]

        return hits

    def apply_triplet_cuts(self, triplet_edges):
        method = self.module_map.get("method")
        tolerance = self.module_map.get("tolerance", 0.0)
        threshold_factor_rms = self.module_map.get("threshold_factor_rms", -1)
        occurence_threshold = self.module_map.get("occurence_threshold", -1)

        # Diff dydx
        # ---------
        dy_12 = triplet_edges.y_2 - triplet_edges.y_1
        dy_23 = triplet_edges.y_2 - triplet_edges.y_3
        dx_12 = triplet_edges.x_1 - triplet_edges.x_2
        dx_23 = triplet_edges.x_2 - triplet_edges.x_3

        diff_dydx = dy_12 / dx_12 - dy_23 / dx_23
        diff_dydx[(dx_12 == 0) & (dx_23 == 0)] = 0
        diff_dydx[(dx_12 != 0) & (dx_23 == 0)] = dy_12 / dx_12
        diff_dydx[(dx_12 == 0) & (dx_23 != 0)] = -dy_23 / dx_23

        dydx_mask = get_triplet_mask(
            method,
            diff_dydx,
            "diff_dydx",
            triplet_edges,
            tolerance,
            threshold_factor_rms,
            occurence_threshold,
        )
        triplet_edges = triplet_edges[dydx_mask]

        # Diff dzdr
        # ---------
        dz_12 = triplet_edges.z_2 - triplet_edges.z_1
        dz_23 = triplet_edges.z_3 - triplet_edges.z_2
        dr_12 = triplet_edges.r_2 - triplet_edges.r_1
        dr_23 = triplet_edges.r_3 - triplet_edges.r_2

        diff_dzdr = dz_12 / dr_12 - dz_23 / dr_23
        diff_dzdr[(dr_12 == 0) & (dr_23 == 0)] = 0
        diff_dzdr[(dr_12 != 0) & (dr_23 == 0)] = dz_12 / dr_12
        diff_dzdr[(dr_12 == 0) & (dr_23 != 0)] = -dz_23 / dr_23

        dzdr_mask = get_triplet_mask(
            method,
            diff_dzdr,
            "diff_dzdr",
            triplet_edges,
            tolerance,
            threshold_factor_rms,
            occurence_threshold,
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


def get_mean_rms_mask(feature, feat_mean, feat_rms, rms_threshold_factor):
    """Compute a mask for a [ mean-n*rms , mean+n*rms ] interval of selection"""

    if rms_threshold_factor < 0:
        raise ValueError(
            f"RMS threshold factor must be >= 0 if you want to use this method, not {rms_threshold_factor}"
        )

    rms_mask = (feature > feat_mean - feat_rms * rms_threshold_factor) & (
        feature < feat_mean + feat_rms * rms_threshold_factor
    )

    return rms_mask


def get_capped_mean_rms_mask(
    feature, feat_min, feat_max, feat_mean, feat_rms, tolerance, rms_threshold_factor
):
    """Compute a mask for a [ mean-n*rms , mean+n*rms ] interval of selection, capped with min/max"""

    if rms_threshold_factor < 0:
        raise ValueError(
            f"RMS threshold factor must be >= 0 if you want to use this method, not {rms_threshold_factor}"
        )

    min_rms = feat_mean - feat_rms * rms_threshold_factor
    max_rms = feat_mean + feat_rms * rms_threshold_factor

    # Try to mitigate outlier effect: if mean+/-n*rms gives looser selection than simple min/max
    # stick to min/max
    epsilon = tolerance
    tol_min = feat_min * (1 - np.sign(feat_min) * epsilon)
    tol_max = feat_max * (1 + np.sign(feat_max) * epsilon)

    capped_min = np.maximum(tol_min, min_rms)
    capped_max = np.minimum(tol_max, max_rms)

    mask = (feature > capped_min) & (feature < capped_max)

    return mask


def get_minmax_mask(feature, feat_min, feat_max, tolerance):
    """Compute a mask for a [min, max] interval of selection"""

    epsilon = tolerance
    tol_min = feat_min * (1 - np.sign(feat_min) * epsilon)
    tol_max = feat_max * (1 + np.sign(feat_max) * epsilon)

    minmax_mask = (feature > tol_min) & (feature < tol_max)

    return minmax_mask


def get_hybrid_mask(
    feature,
    feat_min,
    feat_max,
    feat_mean,
    feat_rms,
    tolerance,
    rms_threshold_factor,
    occurence,
    occurence_threshold,
):
    """Compute a mask with an hybird version of min/max or mean+/-n*rms
    if occurence > occurence_threshold: use mean+/-n*rms (capped)
    min/max otherwise
    """

    minmax_mask = get_minmax_mask(feature, feat_min, feat_max, tolerance)

    capped_rms_mask = get_capped_mean_rms_mask(
        feature,
        feat_min,
        feat_max,
        feat_mean,
        feat_rms,
        tolerance,
        rms_threshold_factor,
    )

    hybrid_mask = (occurence > occurence_threshold & capped_rms_mask) | (
        occurence <= occurence_threshold & minmax_mask
    )

    return hybrid_mask


def get_doublet_mask(
    method,
    feature,
    feat_name,
    hits,
    suffix,
    tolerance,
    rms_threshold_factor=None,
    occurence_threshold=None,
):
    if method == "minmax":
        epsilon = tolerance
        mask = (
            feature
            <= hits[f"{feat_name}max_{suffix}"]
            * (1 + np.sign(hits[f"{feat_name}max_{suffix}"]) * epsilon)
        ) & (
            feature
            >= hits[f"{feat_name}min_{suffix}"]
            * (1 - np.sign(hits[f"{feat_name}min_{suffix}"]) * epsilon)
        )
    elif method == "meanrms":
        mask = get_capped_mean_rms_mask(
            feature,
            hits[f"{feat_name}min_{suffix}"],
            hits[f"{feat_name}max_{suffix}"],
            hits[f"{feat_name}_{suffix}_mean"],
            hits[f"{feat_name}_{suffix}_rms"],
            tolerance,
            rms_threshold_factor,
        )
    elif method == "hybrid":
        mask = get_hybrid_mask(
            feature,
            hits[f"{feat_name}min_{suffix}"],
            hits[f"{feat_name}max_{suffix}"],
            hits[f"{feat_name}_{suffix}_mean"],
            hits[f"{feat_name}_{suffix}_rms"],
            tolerance,
            rms_threshold_factor,
            hits["occurence"],
            occurence_threshold,
        )
    else:
        raise ValueError(f"Unsupported module map method {method}.")

    return mask


def get_triplet_mask(
    method,
    feature,
    feat_name,
    hits,
    tolerance,
    rms_threshold_factor=None,
    occurence_threshold=None,
):
    if method == "minmax":
        epsilon = tolerance
        mask = (
            feature
            <= hits[f"{feat_name}_max"]
            * (1 + np.sign(hits[f"{feat_name}_max"]) * epsilon)
        ) & (
            feature
            >= hits[f"{feat_name}_min"]
            * (1 - np.sign(hits[f"{feat_name}_min"]) * epsilon)
        )
    elif method == "meanrms":
        mask = get_capped_mean_rms_mask(
            feature,
            hits[f"{feat_name}_min"],
            hits[f"{feat_name}_max"],
            hits[f"{feat_name}_mean"],
            hits[f"{feat_name}_rms"],
            tolerance,
            rms_threshold_factor,
        )
    elif method == "hybrid":
        mask = get_hybrid_mask(
            feature,
            hits[f"{feat_name}_min"],
            hits[f"{feat_name}_max"],
            hits[f"{feat_name}_mean"],
            hits[f"{feat_name}_rms"],
            rms_threshold_factor,
            hits["occurence"],
            tolerance,
            occurence_threshold,
        )
    else:
        raise ValueError(f"Unsupported module map method {method}.")

    return mask
