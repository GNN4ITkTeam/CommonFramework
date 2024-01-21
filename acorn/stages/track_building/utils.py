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

from __future__ import annotations

import torch
import pandas as pd
import numpy as np
from atlasify import atlasify
from torch_geometric.data import Data
from typing import Dict, Tuple, Optional, Sequence, Union

import matplotlib.pyplot as plt
from scipy.sparse.csgraph import (
    connected_components,
    min_weight_full_bipartite_matching,
)
from scipy.sparse import coo_matrix, csr_matrix

from acorn.utils import get_ratio
from acorn.stages.graph_construction.models.utils import graph_intersection
from acorn.utils.version_utils import get_pyg_data_keys

# ------------- HGNN UTILS ----------------


class PartialData(object):
    """
    A data object that removes isolated hits and select the good tracks
    """

    def __init__(
        self,
        event: Data,
        edge_cut: float = 1e-3,
        loose_cut: float = 0.01,
        tight_cut: Optional[float] = None,
        min_hits: int = 9,
        random_drop: float = 0.0,
        signal_weight: float = 2,
        signal_selection: Optional[Dict[str, Tuple[float, float]]] = {
            "pt": ("range", [1000, None]),
            "nhits": ("range", [3, None]),
        },
        target_selection: Optional[Dict[str, Tuple[float, float]]] = {
            "pt": ("range", [500, None]),
            "nhits": ("range", [3, None]),
        },
    ):
        """
        Arguments:
            event: a `pytorch_geometric` event record
            edge_cut: the cut that is applied edge-wise
            loose_cut: any hit that is isolated by the cut will be removed
            tight_cut: any connected component that is longer than `min_hits`
                under this cut will be selected and disregarded.
            min_hits: see `tight_cut`
            random_drop: randonly drop some edge for data augmentation
            signal_weight: the weight to weigh signals over other positives.
            signal_selection: a dictionary map the name of the feature to the signal selection,
            which is a tuple that is one of the following:
                ("range", (vmin, vmax)): interval of that feature. Use `None` for unbounded.
                ("isin", [allowed values]): is in a set of values
                ("notin", [disallowed values]): is not in the set of values
            target_selection: the same idea as selection but this will REMOVE the tracks from consideration.
        """
        self.device = "cpu"

        self.full_event = event
        self.edge_cut = edge_cut
        self.loose_cut = loose_cut
        self.tight_cut = tight_cut
        self.min_hits = min_hits
        self.random_drop = random_drop
        self.signal_weight = signal_weight
        self.signal_selection = signal_selection
        self.target_selection = target_selection

        self._partial_event = self._preprocess_event()
        self.truth_bgraph, self._truth_info = self._build_truth()

    def _preprocess_event(self) -> Data:
        """
        everything is done on cpu then moved to the device!
        """
        event = self.full_event.cpu()

        # Initialize the array to track which hit to keep
        to_keep = torch.zeros_like(event.hit_id, dtype=torch.bool)

        # Filter out noise hit with loose cut
        edge_mask = torch.rand(event.edge_index.shape[1]) >= self.random_drop
        loose_edges = event.edge_index[:, (event.scores > self.loose_cut) & edge_mask]
        to_keep[loose_edges.unique()] = True

        # Select good tracks with at least `min_hits` hits
        if self.tight_cut:
            tight_edges = event.edge_index[
                :, (event.scores > self.tight_cut) & edge_mask
            ].numpy()
            graph = coo_matrix(
                (np.ones(tight_edges.shape[1]), tight_edges),
                shape=(event.hit_id.shape[0], event.hit_id.shape[0]),
            ).tocsr()
            _, track_id = connected_components(graph, directed=False)
            _, inverse, nhits = np.unique(
                track_id, return_counts=True, return_inverse=True
            )
            track_id[nhits[inverse] < self.min_hits] = -1

            # store the good tracks and update tracker
            track_id = torch.as_tensor(track_id, device=self.device)
            self.cc_tracks = torch.stack(
                [
                    torch.arange(track_id.shape[0])[track_id >= 0],
                    track_id[track_id >= 0],
                ]
            ).to(self.device)
            to_keep[track_id >= 0] = False

        # prepare to mask out the noise hits
        masked_idx = torch.cumsum(to_keep, dim=0) - 1
        masked_idx[
            ~to_keep
        ] = -1  # Can be removed for performance, this is for sanity check

        # build new `Data` instance
        edge_mask = (
            to_keep[event.edge_index].all(0)
            & edge_mask
            & (event.scores > self.edge_cut)
        )
        track_edge_mask = to_keep[event.track_edges].all(0)
        partial_event = Data(
            x=event.x[to_keep],
            hit_id=event.hit_id[to_keep],
            z=event.z[to_keep],
            r=event.r[to_keep],
            phi=event.phi[to_keep],
            eta=event.eta[to_keep],
            edge_index=masked_idx[event.edge_index[:, edge_mask]],
            y=event.y[edge_mask],
            track_edges=masked_idx[event.track_edges[:, track_edge_mask]],
            eta_particle=event.eta_particle[track_edge_mask],
            radius=event.radius[track_edge_mask],
            nhits=event.nhits[track_edge_mask],
            particle_id=event.particle_id[track_edge_mask],
            pt=event.pt[track_edge_mask],
            pdgId=event.pdgId[track_edge_mask],
            primary=event.primary[track_edge_mask],
        )

        return partial_event

    def _build_truth(self):
        """
        build truth informaiton
        """
        truth_bgraph, truth_info = build_truth_bgraph(
            self._partial_event,
            signal_selection=self.signal_selection,
            target_selection=self.target_selection,
        )

        pid_truth_graph = (truth_bgraph @ truth_bgraph.T).tocoo()
        self.pid_truth_graph = torch.stack(
            [
                torch.as_tensor(pid_truth_graph.row),
                torch.as_tensor(pid_truth_graph.col),
            ],
            dim=0,
        )

        return truth_bgraph, truth_info

    def fetch_emb_truth(self, edges):
        edges, y = graph_intersection(
            edges,
            self.pid_truth_graph,
            return_pred_to_truth=False,
            return_truth_to_pred=False,
            unique_pred=False,
            unique_truth=True,
        )
        emb_weights = torch.ones(y.shape, device=y.device)
        emb_weights[y] /= y.sum() * 5
        emb_weights[~y] /= (~y).sum() * 5 / 4
        hinge = 2 * y.float() - 1
        return edges, hinge, emb_weights

    def fetch_truth(
        self,
        tracks: torch.Tensor,
        scores: torch.Tensor,
    ):
        """
        Everything should be done on cpu.
        """
        tracks, scores = tracks.cpu(), scores.cpu()
        pred_bgraph, pred_info = build_pred_bgraph(
            self._partial_event, tracks, 0, scores=scores.numpy()
        )
        graph = (self.truth_bgraph.T @ pred_bgraph).tocoo()
        graph.eliminate_zeros()
        graph.data = (
            np.square(graph.data)
            / self._truth_info["nhits"][graph.row].numpy()
            / pred_info["track_size"][graph.col].numpy()
        )
        num_particles, num_tracks = graph.shape
        graph = coo_matrix(
            (
                np.concatenate([graph.data, 1e-12 * np.ones(num_particles)]),
                (
                    np.concatenate([graph.row, np.arange(num_particles)]),
                    np.concatenate(
                        [graph.col, np.arange(num_particles) + num_tracks]
                    ),  # include virtual tracks for each particle
                ),
            ),
            shape=(num_particles, num_particles + num_tracks),
        )
        row, col = min_weight_full_bipartite_matching(graph, maximize=True)
        row, col = row[col < num_tracks], col[col < num_tracks]
        track_to_particle = -torch.ones(
            num_tracks, dtype=torch.long, device=self.device
        )
        track_to_particle[col] = torch.tensor(row, dtype=torch.long, device=self.device)

        input_pred_graph = torch.stack(
            [tracks[0].to(self.device), track_to_particle[pred_info["track_id"]]], dim=0
        )

        relavent_mask = input_pred_graph[1] >= 0

        y = torch.zeros(relavent_mask.sum(), dtype=torch.bool, device=self.device)

        input_truth_graph = self.truth_bgraph.tocoo()
        input_truth_graph = torch.stack(
            [
                torch.tensor(input_truth_graph.row, device=self.device),
                torch.tensor(input_truth_graph.col, device=self.device),
            ],
            dim=0,
        )

        y = graph_intersection(
            input_pred_graph[:, relavent_mask],
            input_truth_graph,
            return_pred_to_truth=False,
            return_truth_to_pred=False,
            unique_pred=True,
            unique_truth=True,
        )

        weights = torch.ones(y.shape[0], dtype=torch.float, device=self.device)
        weights[
            self.truth_info["is_signal"][input_pred_graph[1, relavent_mask]]
        ] *= self.signal_weight
        weights[~y] /= weights[~y].sum() * 2
        weights[y] /= weights[y].sum() * 2

        return y, weights, relavent_mask

    def get_tracks(self, tracks: torch.Tensor) -> torch.Tensor:
        tracks = tracks.to(self.device)
        if self.tight_cut and self.cc_tracks.numel() > 0:
            tracks = torch.stack(
                [
                    self.partial_event.hit_id[tracks[0]],
                    tracks[1] + self.cc_tracks.max() + 1,
                ],
                dim=0,
            )
            torch.cat([self.cc_tracks, tracks], dim=1)
        else:
            tracks = tracks.clone()
            tracks[0] = self.partial_event.hit_id[tracks[0]]
        return tracks

    @property
    def partial_event(self):
        return self._partial_event.to(self.device)

    @property
    def truth_info(self):
        return {key: value.to(self.device) for key, value in self._truth_info.items()}

    def to(self, device: torch.device) -> PartialData:
        if self.tight_cut:
            self.cc_tracks = self.cc_tracks.to(device)
        self.pid_truth_graph = self.pid_truth_graph.to(device)
        self.device = device
        return self


# ------------- MATCHING UTILS ----------------
def build_truth_bgraph(
    event: Data,
    signal_selection: Optional[Dict[str, Tuple[float, float]]] = {},
    target_selection: Optional[Dict[str, Tuple[float, float]]] = {},
) -> Tuple[csr_matrix, Dict[str, torch.Tensor]]:
    """
    Build the truth information
    Argument:
        event: a torch_geometric event record
        signal_selection: a dictionary map the name of the feature to the signal selection,
        which is a tuple that is one of the following:
            ("range", (vmin, vmax)): interval of that feature. Use `None` for unbounded.
            ("isin", [allowed values]): is in a set of values
            ("notin", [disallowed values]): is not in the set of values
        target_selection: the same idea as selection but this will REMOVE the tracks from consideration.
    Returns:
        truth_bgraph: a csr_matrix sparse graph where (i, j) denotes hit i belongs to particle j
        truth_info: a dictionary of truth-level information of each particle, which contains:
            original_pid: the particle id in the original event record
            pid: the re-indexed particle id ranging from (0, num_particles)
            pt: transverse momentum of the particle,
            eta: the pseudo rapidity of the particle,
            radius: radius of the vertex,
            nhits: number of hits recorded,
            pdgId: the PDG Id of the particle,
            primary: whether it originated from a primary vertex,
    """
    # put event to cpu
    event = event.cpu()

    # define truth_info dictionary
    truth_info = {}

    # select targets
    is_target = torch.ones(event.particle_id.shape[0], dtype=bool)
    for name, (method, queries) in target_selection.items():
        name = "eta_particle" if name == "eta" else name
        if method == "range":
            if queries[0] is not None:
                is_target &= event[name] >= queries[0]
            if queries[1] is not None:
                is_target &= event[name] <= queries[1]
        elif method == "isin":
            is_target &= torch.isin(event[name], torch.as_tensor(queries))
        elif method == "notin":
            is_target &= ~torch.isin(event[name], torch.as_tensor(queries))
        else:
            raise ValueError(
                f"Method must be one of range, isin, notin, but got {method}"
            )

    # relabel tracks and particles:
    original_pid, pid = event.particle_id[is_target].unique(return_inverse=True)
    num_hits = event.hit_id.shape[0]
    num_particles = original_pid.shape[0]
    truth_info["original_pid"] = original_pid
    truth_info["pid"] = pid

    # build attribute arrays
    for attr_name in ["pt", "eta_particle", "radius", "nhits", "pdgId", "primary"]:
        truth_info[attr_name] = torch.scatter(
            torch.zeros(num_particles, dtype=event[attr_name].dtype),
            0,
            pid,
            event[attr_name][is_target],
        )
    truth_info["eta"] = truth_info.pop("eta_particle")

    # build scipy bipartite graphs
    truth_bgraph = coo_matrix(
        (
            np.ones(2 * pid.shape[0]),
            (event.track_edges[:, is_target].reshape(-1).numpy(), pid.repeat(2)),
        ),
        shape=(num_hits, num_particles),
    ).tocsr()
    truth_bgraph.data = np.ones_like(truth_bgraph.data)

    # select signals
    truth_info["is_signal"] = torch.ones(num_particles, dtype=bool)
    for name, (method, queries) in signal_selection.items():
        if method == "range":
            if queries[0] is not None:
                truth_info["is_signal"] &= truth_info[name] >= queries[0]
            if queries[1] is not None:
                truth_info["is_signal"] &= truth_info[name] <= queries[1]
        elif method == "isin":
            truth_info["is_signal"] &= torch.isin(
                truth_info[name], torch.as_tensor(queries)
            )
        elif method == "notin":
            truth_info["is_signal"] &= ~torch.isin(
                truth_info[name], torch.as_tensor(queries)
            )
        else:
            raise ValueError(
                f"Method must be one of range, isin, notin, but got {method}"
            )

    return truth_bgraph, truth_info


def build_pred_bgraph(
    event: Data,
    tracks: torch.Tensor,
    min_hits: Optional[int] = 0,
    scores: Optional[np.ndarray] = None,
) -> Tuple[csr_matrix, Dict[str, torch.Tensor]]:
    """
    Build the prediction information
    Arguments:
        event: a torch_geometric event record
        tracks: the predicted tracks where (i, j) means hit i belongs to track j
        min_hits: minimum number of hits required for a track candidate
        scores: optional, the scores of the assignment
    Returns:
        pred_bgraph: the predicted bipartite graph
        pred_info: a dictionary containing prediction level informaiton, which contains:
            track_size: the size of the track
            is_selected: has at least min_hit hits
            original_track_id: the original track id
            track_id: the re-indexed track ids ranging from (0, num_tracks)
    """
    event, tracks = event.cpu(), tracks.cpu()
    # count tracks
    if min_hits > 0:
        tracks = tracks.unique(dim=1)
        _, inverse, track_size = tracks[1].unique(
            return_counts=True, return_inverse=True
        )
        tracks = tracks[:, track_size[inverse] >= min_hits]

    # relabel tracks and particles:
    original_track_id, track_id, track_size = tracks[1].unique(
        return_counts=True, return_inverse=True
    )
    num_hits = event.hit_id.shape[0]
    num_tracks = original_track_id.shape[0]

    # build scipy bipartite graphs
    if scores is None:
        pred_bgraph = coo_matrix(
            (np.ones(track_id.shape[0]), (tracks[0], track_id)),
            shape=(num_hits, num_tracks),
        ).tocsr()
        pred_bgraph.data = np.ones_like(pred_bgraph.data)
    else:
        pred_bgraph = coo_matrix(
            (scores, (tracks[0], track_id)), shape=(num_hits, num_tracks)
        ).tocsr()

    pred_info = {
        "track_size": track_size,
        "original_track_id": original_track_id,
        "track_id": track_id,
    }
    return pred_bgraph, pred_info


def match_bgraphs(
    truth_bgraph: csr_matrix,
    pred_bgraph: csr_matrix,
    truth_info: Dict[str, torch.Tensor],
    pred_info: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    match the graphs by overlaping the graphs.
    Argument:
        truth_bgraph: the truth_bgraph from build_truth_bgraph
        pred_bgraph: the pred_bgraph from build_pred_bgraph
        truth_info: the truth_info from build_truth_bgraph
        pred_info: the pred_info from build_pred_bgraph
    Return:
        matching: a tensor where (i, j) means the eff and pur are corresponding to particle i and track j
        eff: # overlapped hits / # particle hits
        pur: # overlapped hits / # track hits
    """
    overlaps = (truth_bgraph.T @ pred_bgraph).tocoo()

    num_overlaps = overlaps.data
    eff = torch.as_tensor(num_overlaps) / truth_info["nhits"][overlaps.row]
    pur = torch.as_tensor(num_overlaps) / pred_info["track_size"][overlaps.col]

    matching = torch.stack(
        [torch.as_tensor(overlaps.row), torch.as_tensor(overlaps.col)], dim=0
    ).long()

    return matching, eff, pur


def evaluate_tracking(
    event: Data,
    tracks: torch.tensor,
    min_hits: Optional[int] = 5,
    signal_selection: Optional[Dict[str, Tuple[float, float]]] = {},
    target_selection: Optional[Dict[str, Tuple[float, float]]] = {},
    matching_fraction: Optional[float] = 0.5,
    style: Optional[str] = "ATLAS",
) -> pd.DataFrame:
    """
    Arguments:
        event: a `Data` object containing the truth information
        tracks: a `torch.tensor` of shape [2, N] where the tracks[0, i] node belongs to the tracks[1, i] track
        min_hits: minimum number of hits to be considered a particle
        signal_selection: a dictionary map the name of the feature to the signal selection,
        which is a tuple that is one of the following:
            ("range", (vmin, vmax)): interval of that feature. Use `None` for unbounded.
            ("isin", [allowed values]): is in a set of values
            ("notin", [disallowed values]): is not in the set of values
        target_selection: the same idea as selection but this will REMOVE the tracks from consideration.
        matching_fraction: the matching fraction to be used.
        style: the style of matching. Can either be one-way, two-way, or ATLAS.
    Return:
        matchind_df: a pd.DataFrame containing particle's features and whether it is reconstructed
        truth_df: a pd.DataFrame contraining truth level information
    """
    truth_bgraph, truth_info = build_truth_bgraph(
        event, signal_selection=signal_selection, target_selection=target_selection
    )
    pred_bgraph, pred_info = build_pred_bgraph(event, tracks, min_hits)
    matching, eff, pur = match_bgraphs(
        truth_bgraph,
        pred_bgraph,
        truth_info,
        pred_info,
    )
    matching_df = pd.DataFrame(
        {
            "efficiency": eff,
            "purity": pur,
            "pid": truth_info["original_pid"][matching[0]],
            "pt": truth_info["pt"][matching[0]],
            "eta": truth_info["eta"][matching[0]],
            "radius": truth_info["radius"][matching[0]],
            "nhits": truth_info["nhits"][matching[0]],
            "pdgId": truth_info["pdgId"][matching[0]],
            "primary": truth_info["primary"][matching[0]],
            "is_signal": truth_info["is_signal"][matching[0]],
            "track_id": pred_info["original_track_id"][matching[1]],
            "track_size": pred_info["track_size"][matching[1]],
        }
    )

    truth_df = pd.DataFrame(
        {
            "pid": truth_info["original_pid"],
            "pt": truth_info["pt"],
            "eta": truth_info["eta"],
            "radius": truth_info["radius"],
            "nhits": truth_info["nhits"],
            "pdgId": truth_info["pdgId"],
            "primary": truth_info["primary"],
            "is_signal": truth_info["is_signal"],
        }
    )

    if matching_fraction < 0.5:
        raise ValueError("Matching fraction must be greater or equal to 0.5!")
    elif matching_fraction == 0.5:
        matching_fraction += 1e-12

    if style == "one-way":
        matching_df["reconstructed"] = matching_df["efficiency"] >= matching_fraction
        matching_df["matched"] = matching_df["purity"] >= matching_fraction
    elif style == "two-way":
        matching_df["reconstructed"] = (
            matching_df["efficiency"] >= matching_fraction
        ) & (matching_df["purity"] >= matching_fraction)
        matching_df["matched"] = matching_df["reconstructed"]
    elif style == "ATLAS":
        matching_df["matched"] = matching_df["purity"] >= matching_fraction
        matching_df["reconstructed"] = matching_df.pid.isin(
            matching_df.pid[matching_df["matched"]]
        )
    else:
        raise ValueError("Undefined style!")

    return matching_df, truth_df


def get_statistics(
    matching_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    bin_name: Optional[str] = None,
    bins: Optional[Sequence[float]] = None,
) -> Dict[str, Union[pd.Series, float]]:
    """
    Arguments:
        matching_df: a `DataFrame` returned by `evaluate_tracking`
        truth_df: a `DataFrame` returned by `evaluate_tracking`
        bin_name: the name of variable to bin
        bins: the bins to use while evaluating the graph.
    Return:
        A dictionary containing:
            tracking efficiency: # particle reconstructed / # true particles
            tracking efficiency: # signal particle reconstructed / # true signal particles
            fake rate: 1 - # particle reconstructed / #tracks
            duplicate rate: # matched track / # particles matched to - 1
    """
    num_duplicated_tracks = (
        matching_df[matching_df["matched"]].track_id.unique().size
        - matching_df[matching_df["matched"]].pid.unique().size
    )
    num_matched_particles = matching_df[matching_df["matched"]].pid.unique().size
    duplicate_rate = num_duplicated_tracks / (num_matched_particles + 1e-12)
    num_tracks = matching_df.track_id.unique().size
    num_reconstructed_particles = (
        matching_df.groupby("pid").any()["reconstructed"].sum()
    )
    fake_rate = 1 - num_reconstructed_particles / (num_tracks + 1e-12)
    if bin_name is None:
        reconstructed_particles = (
            matching_df.groupby("pid").any()["reconstructed"].sum()
        )
        total_particles = truth_df.pid.unique().size
        reconstructed_signal = (
            matching_df[matching_df["is_signal"]]
            .groupby("pid")
            .any()["reconstructed"]
            .sum()
        )
        total_signal = truth_df[truth_df["is_signal"]].pid.unique().size
        return {
            "reconstructed_particles": reconstructed_particles,
            "total_particles": total_particles,
            "reconstructed_signal": reconstructed_signal,
            "total_signal": total_signal,
            "num_duplicated_tracks": num_duplicated_tracks,
            "num_matched_particles": num_matched_particles,
            "duplicate_rate": duplicate_rate,
            "num_tracks": num_tracks,
            "num_reconstructed_particles": num_reconstructed_particles,
            "fake_rate": fake_rate,
        }
    matching_df["bin_id"] = pd.cut(matching_df[bin_name], bins, labels=False)
    truth_df["bin_id"] = pd.cut(truth_df[bin_name], bins, labels=False)
    reconstructed_particles = (
        matching_df.groupby(["pid", "bin_id"])
        .any()
        .reset_index()
        .groupby("bin_id")["reconstructed"]
        .sum()
    )
    total_particles = truth_df.groupby("bin_id").pid.agg(lambda x: x.unique().size)
    reconstructed_signal = (
        matching_df[matching_df["is_signal"]]
        .groupby(["pid", "bin_id"])
        .any()
        .reset_index()
        .groupby("bin_id")["reconstructed"]
        .sum()
    )
    total_signal = (
        truth_df[truth_df["is_signal"]]
        .groupby("bin_id")
        .pid.agg(lambda x: x.unique().size)
    )
    return {
        "reconstructed_particles": reconstructed_particles.rename(
            "reconstructed_particles"
        ),
        "total_particles": total_particles.rename("total_particles"),
        "reconstructed_signal": reconstructed_signal.rename("reconstructed_signal"),
        "total_signal": total_signal.rename("total_signal"),
        "num_duplicated_tracks": num_duplicated_tracks,
        "num_matched_particles": num_matched_particles,
        "duplicate_rate": duplicate_rate,
        "num_tracks": num_tracks,
        "num_reconstructed_particles": num_reconstructed_particles,
        "fake_rate": fake_rate,
    }


# ------------- PLOTTING UTILS ----------------


def plot_eff(
    all_stats: Dict[str, pd.Series],
    bins: np.ndarray,
    xlabel: str,
    caption: str,
    save_path: Optional[str] = "track_reconstruction_eff_vs_pt.png",
):

    denominator = "total_signal"
    numerator = "reconstructed_signal"

    df = pd.concat(all_stats[denominator]).groupby("bin_id").agg("sum").reset_index()
    df = df.merge(
        pd.concat(all_stats[numerator]).groupby("bin_id").agg("sum"), on="bin_id"
    ).reset_index()
    xerrs = np.stack([bins[df.bin_id.astype(int)], bins[df.bin_id.astype(int) + 1]])
    xvals = xerrs.mean(0)
    xerrs = xerrs - xvals
    xerrs[0] = -xerrs[0]
    eff, err = get_ratio(df[numerator], df[denominator])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        xvals,
        eff,
        xerr=xerrs,
        yerr=err,
        fmt="o",
        color="black",
        label="Track efficiency",
    )
    # Add x and y labels
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Tracking efficiency", fontsize=16)

    atlasify(
        "Internal",
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
        r" \bar{t}$ and soft interactions) " + "\n" + caption,
    )

    # Save the plot
    fig.savefig(save_path)


# ------------- MAPPING UTILS ----------------


def rearrange_by_distance(event, edge_index):
    assert "r" in get_pyg_data_keys(event) and "z" in get_pyg_data_keys(
        event
    ), "event must contain r and z"
    distance = event.r**2 + event.z**2

    # flip edges that are pointing inward
    edge_mask = distance[edge_index[0]] > distance[edge_index[1]]
    edge_index[:, edge_mask] = edge_index[:, edge_mask].flip(0)

    return edge_index
