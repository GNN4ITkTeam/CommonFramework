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

import pandas as pd
import numpy as np
import warnings


def read_particles(filename):
    """
    Parse in the CSV files containing each particle, dumped from Athena
    """

    field_names = [
        "subevent",
        "barcode",
        "px",
        "py",
        "pz",
        "pt",
        "eta",
        "vx",
        "vy",
        "vz",
        "radius",
        "status",
        "charge",
        "pdgId",
        "pass",
        "vProdNIn",
        "vProdNOut",
        "vProdStatus",
        "vProdBarcode",
    ]

    particles = pd.read_csv(filename, header=None, sep=r",#", engine="python")

    particles = particles[0].str.split(",", expand=True)
    particles.columns = field_names

    # Prepare barcodes for later processing
    particles = particles.astype({"barcode": str, "subevent": str})

    return particles


def get_max_length(series):
    """
    A utility to get the maximum string length of a particle barcode, used for padding
    with zeros
    """

    max_entry = str(series.astype(int).max())

    return len(max_entry)


def convert_barcodes(particles):
    """
    Build a particle ID from concatenating the subevent and the barcode,
    padded with zeros such that there is no danger of accidental duplications.

    E.g. Assuming the longest barcode is 2965491 (length of 7), then some particle:
    (subevent, barcode) = (496, 17) -> particle_id = 4960000017
    """

    max_length = get_max_length(particles.barcode)

    # Use "insert" to put the particle_id at the start of the DF
    particles.insert(
        0,
        "particle_id",
        particles.subevent + particles.barcode.str.pad(width=max_length, fillchar="0"),
    )

    return particles


def get_detectable_particles(particles, clusters):
    """
    Apply some detectability cuts for statistical analysis. Note that this is not a reconstructability cut - we simply require
    that particles can be detected in some way.
    """
    num_clusters = (
        clusters.groupby("particle_id")["cluster_id"]
        .count()
        .reset_index(name="num_clusters")
    )
    particles = pd.merge(particles, num_clusters, on="particle_id").fillna(
        method="ffill"
    )

    cut1 = particles[particles.charge.abs() > 0]  # Keep charged particles
    return cut1[
        cut1.num_clusters > 0
    ]  # Keep particles which are leaved at least one cluster


def read_spacepoints(filename):
    hits = pd.read_csv(
        filename,
        header=None,
        names=["hit_id", "x", "y", "z", "cluster_index_1", "cluster_index_2"],
    )

    # Wherever cluster_index_2 is NaN (i.e. for pixel hits), replace with cluster_index_1
    hits["cluster_index_2"] = (
        hits["cluster_index_2"].fillna(hits["cluster_index_1"]).astype("int64")
    )

    # Check that hit_ids are a sequential list, if they're not, set them as such and return a warning
    if not hits.hit_id.equals(pd.Series(range(len(hits)))):
        warnings.warn("Hit IDs are not sequential, fixing")
        hits["hit_id"] = range(len(hits))

    return hits


def split_particle_entries(cluster_df, particles):
    """
    Do some fiddling to split the cluster entry for truth particle, which could have 0 true
    particles, 1 true particle, or many true particles
    """

    cleaned_cell_pids = cluster_df[["cluster_id", "particle_id"]].astype(
        {"particle_id": str}
    )

    split_pids = pd.DataFrame(
        [
            [c, p]
            for c, P in cleaned_cell_pids.itertuples(index=False)
            for p in P.split("),(")
        ],
        columns=cleaned_cell_pids.columns,
    )

    split_pids = split_pids.join(
        split_pids.particle_id.str.strip("()")
        .str.split(",", expand=True)
        .rename({0: "subevent", 1: "barcode"}, axis=1)
    ).drop(columns=["particle_id", 2])

    split_pids["particle_id"] = (
        split_pids.merge(
            particles[["subevent", "barcode", "particle_id"]].astype(
                {"subevent": str, "barcode": str}
            ),
            how="left",
            on=["subevent", "barcode"],
        )["particle_id"]
        .fillna(0)
        .astype(int)
    )

    split_pids = split_pids[["cluster_id", "particle_id"]]

    return split_pids


def read_clusters(clusters_file, particles, column_lookup):
    """
    Read the cluster CSV files by splitting into sections around the #'s
    """
    column_sets = [
        "coordinates",
        "region",
        "barcodes",
        "cells",
        "shape",
        "norms",
        "covariance",
    ]

    clusters_raw = pd.read_csv(
        clusters_file, header=None, sep=r",#,|#,|,#", engine="python"
    )
    clusters_raw.columns = column_sets

    clusters_processed, shape_list = split_cluster_entries(
        clusters_raw, particles, column_lookup
    )

    split_pids = split_particle_entries(clusters_processed, particles)

    # Fix some types
    clusters_processed = clusters_processed.drop(columns=["particle_id"])
    clusters_processed = clusters_processed.merge(split_pids, on="cluster_id").astype(
        {"cluster_id": int}
    )

    # Fix indexing mismatch in DumpObjects - is this still necessary????
    clusters_processed["cluster_id"] = clusters_processed["cluster_id"] - 1

    return clusters_processed, shape_list


def split_cluster_entries(clusters_raw, particles, column_lookup):
    """
    Split the cluster text file into separate columns, as defined in the config file for
    """

    clusters_processed = pd.DataFrame()

    # First read the co-ordinates of each cluster
    clusters_processed[column_lookup["coordinates"]] = clusters_raw[
        "coordinates"
    ].str.split(",", expand=True)

    # Split the detector geometry information
    clusters_processed[column_lookup["region"]] = clusters_raw["region"].str.split(
        ",", expand=True
    )

    # Split the module norms
    clusters_processed[column_lookup["norms"]] = clusters_raw["norms"].str.split(
        ",", expand=True
    )

    # Handle the two versions of dumpObjects - one with more shape information
    cluster_shape = clusters_raw["shape"].str.split(",", expand=True)
    if cluster_shape.shape[1] == 2:
        shape_list = column_lookup["shape_b"]
    elif cluster_shape.shape[1] == 14:
        shape_list = column_lookup["shape_a"]
    else:
        raise ValueError("Unknown shape information")
    clusters_processed[shape_list] = cluster_shape

    # Split the particle IDs
    clusters_processed[["particle_id"]] = clusters_raw[["barcodes"]]

    return clusters_processed, shape_list


def truth_match_clusters(spacepoints, clusters):
    """
    Here we handle the case where a pixel spacepoint belongs to exactly one cluster, but
    a strip spacepoint belongs to 0, 1, or 2 clusters, and we only accept the case of 2 clusters
    with shared truth particle_id
    """
    spacepoints = spacepoints.merge(
        clusters, left_on="cluster_index_1", right_on="cluster_id", how="left"
    )
    spacepointlike_fields = [
        "hardware",
        "barrel_endcap",
        "layer_disk",
        "eta_module",
        "phi_module",
    ]
    spacepoints = spacepoints.merge(
        clusters.drop(spacepointlike_fields, axis=1),
        left_on="cluster_index_2",
        right_on="cluster_id",
        how="left",
        suffixes=("_1", "_2"),
    ).drop(["cluster_id_1", "cluster_id_2"], axis=1)

    # Get clusters that share particle ID
    matching_clusters = spacepoints.particle_id_1 == spacepoints.particle_id_2
    spacepoints["particle_id"] = spacepoints["particle_id_1"].where(
        matching_clusters, other=0
    )
    spacepoints.astype(
        {"particle_id": "int64", "particle_id_1": "int64", "particle_id_2": "int64"}
    )
    return spacepoints


def clean_spacepoints(spacepoints):
    """
    Remove the duplicate occurences of spacepoints with the same particle ID, or where a spacepoint
    belongs to both a true particle and noise
    """

    # Ignore duplicate entries (possible if a particle has duplicate hits in the same clusters)
    spacepoints = spacepoints.drop_duplicates(
        ["hit_id", "cluster_index_1", "cluster_index_2", "particle_id"]
    ).fillna(-1)

    # Handle the case where a spacepoint belongs to both a true particle AND noise - we should remove the noise row
    noise_hits = spacepoints[spacepoints.particle_id == 0].drop_duplicates(
        subset="hit_id"
    )
    signal_hits = spacepoints[spacepoints.particle_id != 0]
    non_duplicate_noise_hits = noise_hits[~noise_hits.hit_id.isin(signal_hits.hit_id)]
    cleaned_hits = pd.concat([signal_hits, non_duplicate_noise_hits], ignore_index=True)

    # Sort hits by hit_id for ease of processing
    cleaned_hits = cleaned_hits.sort_values("hit_id").reset_index(drop=True)
    return cleaned_hits


def get_truth_spacepoints(spacepoints, clusters, spacepoints_datatypes):
    # Build truth list of spacepoints by handling matching clusters
    truth_spacepoints = truth_match_clusters(spacepoints, clusters)
    # Tidy up the truth dataframe and add in all cluster information
    truth_spacepoints = clean_spacepoints(truth_spacepoints)

    # Set spacepoint datatypes
    truth_spacepoints = truth_spacepoints.astype(
        {
            k: v
            for k, v in spacepoints_datatypes.items()
            if k in truth_spacepoints.columns
        }
    )

    return truth_spacepoints


def remove_undetectable_particles(truth, particles):
    """
    This method is opinionated, so we place it in the AthenaReader, rather than the core DataReader
    The idea is to set truth particle ID to 0 for particles that are undetectable by the detector
    """

    unreconstructable_hits = truth[~truth.particle_id.isin(particles.particle_id)]
    truth.loc[unreconstructable_hits.index, "particle_id"] = 0

    return truth


def add_module_id(hits, module_lookup):
    """
    Add the module ID to the hits dataframe
    """
    if "module_id" in hits:
        return hits

    if "module_id_1" in hits:
        # Duplicate module_id_1 to module_id
        hits["module_id"] = hits["module_id_1"]
        return hits

    cols_to_merge = [
        "hardware",
        "barrel_endcap",
        "layer_disk",
        "eta_module",
        "phi_module",
    ]
    merged_hits = hits.merge(
        module_lookup[cols_to_merge + ["ID"]], on=cols_to_merge, how="left"
    )
    merged_hits = merged_hits.rename(columns={"ID": "module_id"})

    # make sure we had all modules in the config file
    assert (
        not merged_hits["module_id"].isnull().values.any()
    ), "Some modules are unknown, please check your module lookup file"

    assert hits.shape[0] == merged_hits.shape[0], (
        "Merged hits dataframe has different number of rows - possibly missing modules"
        " from lookup"
    )
    assert merged_hits.shape[1] - hits.shape[1] == 1, (
        "Merged hits dataframe has different number of columns; should only have added"
        " module_id column"
    )

    return merged_hits


def add_region_labels(hits, region_labels: dict):
    """
    Label the 6 detector regions (forward-endcap pixel, forward-endcap strip, etc.)
    """

    for region_label, conditions in region_labels.items():
        condition_mask = np.logical_and.reduce(
            [
                hits[condition_column] == condition
                for condition_column, condition in conditions.items()
            ]
        )
        hits.loc[condition_mask, "region"] = region_label

    assert (
        hits.region.isna()
    ).sum() == 0, "There are hits that do not belong to any region!"

    return hits
