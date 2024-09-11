from __future__ import annotations
import torch
import pandas as pd
from torch_geometric.data import Data
from typing import Dict, Tuple, Optional, Sequence, Union
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

# Local imports
from acorn.utils.mapping_utils import get_condition_lambda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            refer to `get_condition_lambda` for the formatting
        target_selection: the same idea as selection but this will REMOVE the tracks from consideration.
            should only be used when speed is important, otherwise could lead to incorrect result.
    Returns:
        truth_bgraph: a csr_matrix sparse graph where (i, j) denotes hit i belongs to particle j
        truth_info: a dictionary of truth-level information of each particle, which contains:
            original_pid: the particle id in the original event record
            pid: the re-indexed particle id ranging from (0, num_particles)
            pt: transverse momentum of the particle,
            eta_particle: the pseudo rapidity of the particle,
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
    for condition_key, (condition_val) in target_selection.items():
        assert condition_key in event
        is_target &= get_condition_lambda(condition_key, condition_val)(event)

    # relabel tracks and particles:
    original_pid, pid = event.particle_id[is_target].unique(return_inverse=True)
    num_hits = event.hit_id.shape[0]
    num_particles = original_pid.shape[0]
    truth_info["original_pid"] = original_pid
    truth_info["pid"] = pid

    # build attribute arrays
    for attr_name in [
        "track_particle_pt",
        "track_particle_eta",
        "track_particle_radius",
        "track_particle_nhits",
        "track_particle_pdgId",
        "track_particle_primary",
    ]:
        truth_info[attr_name] = torch.scatter(
            torch.zeros(num_particles, dtype=event[attr_name].dtype),
            0,
            pid,
            event[attr_name][is_target],
        )

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
    for condition_key, (condition_val) in signal_selection.items():
        assert condition_key in event
        truth_info["is_signal"] &= get_condition_lambda(condition_key, condition_val)(
            truth_info
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
    eff = (
        torch.as_tensor(num_overlaps) / truth_info["track_particle_nhits"][overlaps.row]
    )
    pur = torch.as_tensor(num_overlaps) / pred_info["track_size"][overlaps.col]

    matching = torch.stack(
        [torch.as_tensor(overlaps.row), torch.as_tensor(overlaps.col)], dim=0
    ).long()

    return matching, eff, pur


def evaluate_tracking(
    event: Data,
    tracks: torch.tensor,
    min_hits: Optional[int] = 5,
    target_tracks: Optional[Dict[str, Tuple[float, float]]] = {},
    matching_fraction: Optional[float] = 0.5,
    style: Optional[str] = "ATLAS",
) -> pd.DataFrame:
    """
    Arguments:
        event: a `Data` object containing the truth information
        tracks: a `torch.tensor` of shape [2, N] where the tracks[0, i] node belongs to the tracks[1, i] track
        min_hits: minimum number of hits to be considered a particle
        target_tracks: a dictionary map the name of the feature to the signal selection,
        which is a tuple that is one of the following:
            ("range", (vmin, vmax)): interval of that feature. Use `None` for unbounded.
            ("isin", [allowed values]): is in a set of values
            ("notin", [disallowed values]): is not in the set of values
        matching_fraction: the matching fraction to be used.
        style: the style of matching. Can either be one-way, two-way, or ATLAS.
    Return:
        matchind_df: a pd.DataFrame containing particle's features and whether it is reconstructed
        truth_df: a pd.DataFrame contraining truth level information
    """
    truth_bgraph, truth_info = build_truth_bgraph(
        event, signal_selection=target_tracks, target_selection={}
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
            "pt": truth_info["track_particle_pt"][matching[0]],
            "eta_particle": truth_info["track_particle_eta"][matching[0]],
            "radius": truth_info["track_particle_radius"][matching[0]],
            "nhits": truth_info["track_particle_nhits"][matching[0]],
            "pdgId": truth_info["track_particle_pdgId"][matching[0]],
            "primary": truth_info["track_particle_primary"][matching[0]],
            "is_signal": truth_info["is_signal"][matching[0]],
            "track_id": pred_info["original_track_id"][matching[1]],
            "track_size": pred_info["track_size"][matching[1]],
        }
    )

    truth_df = pd.DataFrame(
        {
            "pid": truth_info["original_pid"],
            "pt": truth_info["track_particle_pt"],
            "eta_particle": truth_info["track_particle_eta"],
            "radius": truth_info["track_particle_radius"],
            "nhits": truth_info["track_particle_nhits"],
            "pdgId": truth_info["track_particle_pdgId"],
            "primary": truth_info["track_particle_primary"],
            "is_signal": truth_info["is_signal"],
        }
    )

    if matching_fraction < 0.5:
        raise ValueError("Matching fraction must not be less than 0.5!")
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


def from_bgraph_to_df(graph):
    return pd.DataFrame(
        {"hit_id": graph.bgraph[0].cpu(), "track_id": graph.bgraph[1].cpu()}
    )
