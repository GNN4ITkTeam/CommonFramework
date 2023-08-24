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

import torch
import pandas as pd
import numpy as np
from atlasify import atlasify

import matplotlib.pyplot as plt
from gnn4itk_cf.utils.plotting_utils import get_ratio

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


def load_particles_df(graph):
    """Load the particles from a file."""
    # Get the particle dataframe
    particles_df = pd.DataFrame({"particle_id": graph.particle_id, "pt": graph.pt})

    # Reduce to only unique particle_ids
    particles_df = particles_df.drop_duplicates(subset=["particle_id"])

    return particles_df


def get_matching_df(
    reconstruction_df, particles_df, min_track_length=1, min_particle_length=1
):
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
    spacepoint_matching["is_reconstructable"] = (
        spacepoint_matching.n_true_hits >= min_particle_length
    )

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
    matching_fraction=0.5,
    matching_style="ATLAS",
    min_track_length=1,
    min_particle_length=1,
):
    if matching_fraction < 0.5:
        raise ValueError("Matching fraction must be >= 0.5")

    if matching_fraction == 0.5:
        # Add a tiny bit of noise to the matching fraction to avoid double-matched tracks
        matching_fraction += 1e-6

    # Load the labelled graphs as reconstructed dataframes
    reconstruction_df = load_reconstruction_df(graph)
    particles_df = load_particles_df(graph)

    # Get matching dataframe
    matching_df = get_matching_df(
        reconstruction_df,
        particles_df,
        min_track_length=min_track_length,
        min_particle_length=min_particle_length,
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


def plot_pt_eff(particles, pt_units, save_path="track_reconstruction_eff_vs_pt.png"):
    pt = particles.pt.values

    true_pt = pt[particles["is_reconstructable"]]
    reco_pt = pt[particles["is_reconstructable"] & particles["is_reconstructed"]]

    pt_min, pt_max = 1, 20
    if pt_units == "MeV":
        pt_min, pt_max = pt_min * 1000, pt_max * 1000
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)

    # Get histogram values of true_pt and reco_pt
    true_vals, true_bins = np.histogram(true_pt, bins=pt_bins)
    reco_vals, reco_bins = np.histogram(reco_pt, bins=pt_bins)

    # Plot the ratio of the histograms as an efficiency
    eff, err = get_ratio(reco_vals, true_vals)

    xvals = (true_bins[1:] + true_bins[:-1]) / 2
    xerrs = (true_bins[1:] - true_bins[:-1]) / 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        xvals, eff, xerr=xerrs, yerr=err, fmt="o", color="black", label="Efficiency"
    )
    # Add x and y labels
    ax.set_xlabel(f"$p_T [{pt_units}]$", fontsize=16)
    ax.set_ylabel("Efficiency", fontsize=16)

    atlasify(
        "Internal",
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
        r" \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$GeV, $|\eta < 4$",
    )

    # Save the plot
    fig.savefig(save_path)


# ------------- MAPPING UTILS ----------------


def rearrange_by_distance(event, edge_index):
    assert "r" in event.keys and "z" in event.keys, "event must contain r and z"
    distance = event.r**2 + event.z**2

    # flip edges that are pointing inward
    edge_mask = distance[edge_index[0]] > distance[edge_index[1]]
    edge_index[:, edge_mask] = edge_index[:, edge_mask].flip(0)

    return edge_index
