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

from typing import Dict
import torch
import pandas as pd
import numpy as np
from atlasify import atlasify

import matplotlib.pyplot as plt

from acorn.utils import get_ratio
from acorn.utils.version_utils import get_pyg_data_keys

# ------------- MATCHING UTILS ----------------


def load_reconstruction_df(graph):
    """Load the reconstructed tracks from a file."""
    if hasattr(graph, "hit_id"):
        hit_id = graph.hit_id
    else:
        hit_id = torch.arange(graph.num_nodes)
    pids = torch.zeros(hit_id.shape[0], dtype=torch.int64)
    pids[graph.track_edges[0]] = graph.particle_id
    pids[graph.track_edges[1]] = graph.particle_id

    return pd.DataFrame(
        {"hit_id": hit_id, "track_id": graph.labels, "particle_id": pids}
    )


def load_particles_df(graph, sel_conf: dict):
    """Load the particles from a file."""
    # Get the particle dataframe

    # By default have only particle pt
    cols = {"particle_id": graph.particle_id, "pt": graph.pt}

    # Add more variable if needed for th fiducial selection
    for var in sel_conf:
        if var not in cols:
            if var == "n_true_hits":
                # Specific case: not embedded in graphs but added in the dataframe later on
                # So we ignore it at this stage
                continue
            cols[var] = graph[var]

    # particles_df = pd.DataFrame({"particle_id": graph.particle_id,
    #                              "pt": graph.pt, "eta_particle": graph.eta_particle,
    #                              "pdgId": graph.pdgId, "radius": graph.radius,
    #                              "primary": graph.primary})

    particles_df = pd.DataFrame(cols)

    # Reduce to only unique particle_ids
    particles_df = particles_df.drop_duplicates(subset=["particle_id"])

    return particles_df


def apply_fiducial_sel(df: pd.DataFrame, sel_conf: dict):
    """Add 'is_reconstructable' item to the dataframe based on the fiducial selection defined in config"""

    df["is_reconstructable"] = True

    for key, values in sel_conf.items():
        if isinstance(values, list):
            if values[0] == "not_in":
                for val in values[1]:
                    df["is_reconstructable"] = df["is_reconstructable"] * (
                        df[key] != val
                    )
            else:
                df["is_reconstructable"] = (
                    df["is_reconstructable"]
                    * (df[key] >= values[0])
                    * (df[key] <= values[1])
                )
        else:
            df["is_reconstructable"] = df["is_reconstructable"] * (df[key] >= values)

    return df


def get_matching_df(reconstruction_df, particles_df, sel_conf, min_track_length=1):
    # Get track lengths
    candidate_lengths = (
        reconstruction_df.track_id.value_counts(sort=False)
        .reset_index()
        .rename(columns={"index": "track_id", "track_id": "n_reco_hits"})
    )

    # Get true track lengths
    particle_lengths = (
        reconstruction_df.drop_duplicates(subset=["hit_id"])
        .particle_id.value_counts(sort=False)
        .reset_index()
        .rename(columns={"index": "particle_id", "particle_id": "n_true_hits"})
    )

    spacepoint_matching = (
        reconstruction_df.groupby(["track_id", "particle_id"])
        .size()
        .reset_index()
        .rename(columns={0: "n_shared"})
    )

    spacepoint_matching = spacepoint_matching.merge(
        candidate_lengths, on=["track_id"], how="left"
    )
    spacepoint_matching = spacepoint_matching.merge(
        particle_lengths, on=["particle_id"], how="left"
    )
    spacepoint_matching = spacepoint_matching.merge(
        particles_df, on=["particle_id"], how="left"
    )

    # Filter out tracks with too few shared spacepoints
    spacepoint_matching["is_matchable"] = (
        spacepoint_matching.n_reco_hits >= min_track_length
    )

    spacepoint_matching = apply_fiducial_sel(spacepoint_matching, sel_conf)

    return spacepoint_matching


def calculate_matching_fraction(spacepoint_matching_df):
    spacepoint_matching_df = spacepoint_matching_df.assign(
        purity_reco=np.true_divide(
            spacepoint_matching_df.n_shared, spacepoint_matching_df.n_reco_hits
        )
    )
    spacepoint_matching_df = spacepoint_matching_df.assign(
        eff_true=np.true_divide(
            spacepoint_matching_df.n_shared, spacepoint_matching_df.n_true_hits
        )
    )

    return spacepoint_matching_df


def evaluate_labelled_graph(
    graph,
    sel_conf,
    matching_fraction=0.5,
    matching_style="ATLAS",
    min_track_length=1,
):
    if matching_fraction < 0.5:
        raise ValueError("Matching fraction must be >= 0.5")

    if matching_fraction == 0.5:
        # Add a tiny bit of noise to the matching fraction to avoid double-matched tracks
        matching_fraction += 1e-6

    # Load the labelled graphs as reconstructed dataframes
    reconstruction_df = load_reconstruction_df(graph)
    particles_df = load_particles_df(graph, sel_conf)

    # Get matching dataframe
    matching_df = get_matching_df(
        reconstruction_df,
        particles_df,
        sel_conf,
        min_track_length=min_track_length,
    )
    # Flatten event_id if it's a list
    event_id = graph.event_id
    while type(event_id) == list:
        event_id = event_id[0]
    matching_df["event_id"] = int(event_id)

    # calculate matching fraction
    matching_df = calculate_matching_fraction(matching_df)

    # Run matching depending on the matching style
    if matching_style == "ATLAS":
        matching_df["is_matched"] = matching_df["is_reconstructed"] = (
            matching_df.purity_reco >= matching_fraction
        )
    elif matching_style == "one_way":
        matching_df["is_matched"] = matching_df.purity_reco >= matching_fraction
        matching_df["is_reconstructed"] = matching_df.eff_true >= matching_fraction
    elif matching_style == "two_way":
        matching_df["is_matched"] = matching_df["is_reconstructed"] = (
            matching_df.purity_reco >= matching_fraction
        ) & (matching_df.eff_true >= matching_fraction)

    return matching_df


# ------------- PLOTTING UTILS ----------------


default_eta_bins = np.arange(-4.0, 4.4, step=0.4)
default_eta_configs = {
    "bins": default_eta_bins,
    "histtype": "step",
    "lw": 2,
    "log": False,
}


def plot_eff(
    particles, var: str, varconf: Dict, save_path="track_reconstruction_eff_vs_XXX.png"
):
    if var not in ["pt", "eta"]:
        raise ValueError(f"Unsupported variable {var}, should be either `pt` or `eta`.")

    if var == "pt":
        x = particles.pt.values
        if "x_bins" in varconf:
            x_bins = varconf["x_bins"]
        elif "x_lim" in varconf:
            x_bins = np.logspace(
                np.log10(varconf["x_lim"][0]), np.log10(varconf["x_lim"][1]), 10
            )
        else:
            x_bins = 20
    elif var == "eta":
        x = particles.eta_particle.values
        if "x_bins" in varconf:
            x_bins = varconf["x_bins"]
        elif "x_lim" in varconf:
            x_bins = np.arange(varconf["x_lim"][0], varconf["x_lim"][1], step=0.4)
        else:
            x_bins = default_eta_bins

    if "x_scale" in varconf:
        x = x * float(varconf["x_scale"])

    true_x = x[particles["is_reconstructable"]]
    reco_x = x[particles["is_reconstructable"] & particles["is_reconstructed"]]

    # Get histogram values of true_pt and reco_pt
    true_vals, true_bins = np.histogram(true_x, bins=x_bins)
    reco_vals, reco_bins = np.histogram(reco_x, bins=x_bins)

    # Plot the ratio of the histograms as an efficiency
    eff, err = get_ratio(reco_vals, true_vals)

    xvals = (true_bins[1:] + true_bins[:-1]) / 2
    xerrs = (true_bins[1:] - true_bins[:-1]) / 2

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
    ax.set_xlabel(varconf.get("x_label", "x_label is None"), fontsize=16)
    ax.set_ylabel("Track Efficiency", fontsize=16)
    if "y_lim" in varconf:
        ax.set_ylim(ymin=varconf["y_lim"][0], ymax=varconf["y_lim"][1])

    atlasify(
        "Internal",
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries ($t"
        r" \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$GeV, $|\eta| < 4$",
        enlarge=1,
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
