import pandas as pd
import numpy as np
import csv

def read_particles(filename):
    """
    Parse in the CSV files containing each particle, dumped from Athena
    """
    
    field_names = ['subevent', 'barcode', 'px', 'py', 'pz', 'pt', 
               'eta', 'vx', 'vy', 'vz', 'radius', 'status', 'charge', 
               'pdgId', 'pass', 'vProdNIn', 'vProdNOut', 'vProdStatus', 'vProdBarcode']
    
    particles = pd.read_csv(filename, header=None, sep=r",#", engine='python')
    
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
    particles.insert(0, "particle_id",
        particles.subevent + particles.barcode.str.pad(width=max_length, fillchar='0'))
    
    return particles

def get_detectable_particles(particles, clusters):
    """
    Apply some detectability cuts for statistical analysis. Note that this is not a reconstructability cut - we simply require
    that particles can be detected in some way.
    """
    num_clusters =  clusters.groupby("particle_id")["cluster_id"].count().reset_index(name="num_clusters")
    particles = pd.merge(particles, num_clusters, on='particle_id').fillna(method='ffill')

    cut1 = particles[particles.charge.abs() > 0]  # Keep charged particles
    cut2 = cut1[cut1.num_clusters > 0]  # Keep particles which are leaved at least one cluster
    return cut2

def read_spacepoints(filename):
    
    hits = pd.read_csv(
        filename,
        header=None,
        names=["hit_id", "x", "y", "z", "cluster_index_1", "cluster_index_2"],
    )

    pixel_hits = hits[pd.isna(hits["cluster_index_2"])]
    strip_hits = hits[~pd.isna(hits["cluster_index_2"])]

    return pixel_hits, strip_hits

def split_particle_entries(cluster_df, particles):
    
    """
    Do some fiddling to split the cluster entry for truth particle, which could have 0 true
    particles, 1 true particle, or many true particles
    """
    
    cleaned_cell_pids = cluster_df[["cluster_id", "particle_id"]].astype({"particle_id": str})

    split_pids = pd.DataFrame(
        [
            [c, p]
            for c, P in cleaned_cell_pids.itertuples(index=False)
            for p in P.split("),(")
        ],
        columns=cleaned_cell_pids.columns,
    )

    split_pids = split_pids.join(
                split_pids.particle_id.str.strip("()").str.split(",", expand=True).rename(
                    {0: "subevent", 1: "barcode"}, axis=1)
                ).drop(columns=["particle_id", 2])
    
    split_pids["particle_id"] = split_pids.merge(particles[["subevent", "barcode", "particle_id"]].astype({"subevent": str, "barcode": str}), how="left", 
                                 on=["subevent", "barcode"])["particle_id"].fillna(0).astype(int)
    
    split_pids = split_pids[["cluster_id", "particle_id"]]
    
    return split_pids

def read_clusters(clusters_file, particles, column_lookup):
    
    """
    Read the cluster CSV files by splitting into sections around the #'s
    """
    column_sets = ["coordinates", "region", "barcodes", "cells", "shape", "norms", "covariance"]
                    
    clusters_raw = pd.read_csv(clusters_file, header=None, sep=r",#,|#,|,#", engine='python')
    clusters_raw.columns = column_sets
    
    clusters_processed = split_cluster_entries(clusters_raw, particles, column_lookup)

    split_pids = split_particle_entries(clusters_processed, particles)
    
    # Fix some types
    clusters_processed = clusters_processed.drop(columns=["particle_id"])
    clusters_processed = clusters_processed.merge(split_pids, on="cluster_id").astype({"cluster_id": int})
    
    # Fix indexing mismatch in DumpObjects - is this still necessary????
    clusters_processed["cluster_id"] = clusters_processed["cluster_id"] - 1
    
    return clusters_processed

def split_cluster_entries(clusters_raw, particles, column_lookup):
    """
    Split the cluster text file into separate columns, as defined in the config file for 
    """

    clusters_processed = pd.DataFrame()
    
    # First read the co-ordinates of each cluster
    clusters_processed[column_lookup["coordinates"]] = clusters_raw["coordinates"].str.split(",", expand=True)
    
    # Split the detector geometry information
    clusters_processed[column_lookup["region"]] = clusters_raw["region"].str.split(",", expand=True)

    # Split the module norms
    clusters_processed[column_lookup["norms"]] = clusters_raw["norms"].str.split(",", expand=True)

    # Handle the two versions of dumpObjects - one with more shape information
    cluster_shape = clusters_raw["shape"].str.split(",", expand=True)
    if cluster_shape.shape[1] == 2:
        clusters_processed[column_lookup["shape_b"]] = cluster_shape
    elif cluster_shape.shape[1] == 14:
        clusters_processed[column_lookup["shape_a"]] = cluster_shape
    else:
        raise ValueError("Unknown shape information")

    # Split the particle IDs
    clusters_processed[["particle_id"]] = clusters_raw[["barcodes"]]

    return clusters_processed

def truth_match_clusters(pixel_hits, strip_hits, clusters):
    """
    Here we handle the case where a pixel spacepoint belongs to exactly one cluster, but
    a strip spacepoint belongs to 0, 1, or 2 clusters, and we only accept the case of 2 clusters
    with shared truth particle_id
    """
    pixel_clusters = pixel_hits.merge(clusters[['cluster_id', 'particle_id']], left_on='cluster_index_1', right_on='cluster_id', how='left').drop("cluster_id", axis=1)
    pixel_clusters["particle_id_1"] = pixel_clusters["particle_id"] 
    pixel_clusters["particle_id_2"] = -1
    strip_clusters = strip_hits.merge(clusters[['cluster_id', 'particle_id']], left_on='cluster_index_1', right_on='cluster_id', how='left')
    strip_clusters = strip_clusters.merge(clusters[['cluster_id', 'particle_id']], left_on='cluster_index_2', right_on='cluster_id', how='left', suffixes=('_1', '_2')).drop(['cluster_id_1', 'cluster_id_2'], axis=1)
    
    # Get clusters that share particle ID
    matching_clusters = strip_clusters.particle_id_1 == strip_clusters.particle_id_2
    strip_clusters['particle_id'] = strip_clusters["particle_id_1"].where(matching_clusters, other=0)
    strip_clusters["particle_id_1"].astype('int64')
    strip_clusters["particle_id_2"].astype('int64')
    truth_spacepoints = pd.concat([pixel_clusters, strip_clusters], ignore_index=True)
    return truth_spacepoints

def merge_spacepoints_clusters(spacepoints, clusters):
    """
    Finally, we merge the features of each cluster with the spacepoints - where a spacepoint may
    own 1 or 2 signal clusters, and thus we give the suffixes _1, _2
    """

    spacepoints = spacepoints.merge(clusters.drop(["particle_id", "side"], axis=1), left_on='cluster_index_1', right_on='cluster_id', how='left').drop("cluster_id", axis=1)
    
    unique_cluster_fields = ['cluster_id', 'cluster_x', 'cluster_y', 'cluster_z', 'eta_angle', 'phi_angle', 'norm_z'] # These are fields that is unique to each cluster (therefore they need the _1, _2 suffix)
    spacepoints = spacepoints.merge(clusters[unique_cluster_fields], left_on='cluster_index_2', right_on='cluster_id', how='left', suffixes=("_1", "_2")).drop("cluster_id", axis=1)
    
    # Ignore duplicate entries (possible if a particle has duplicate hits in the same clusters)
    spacepoints = spacepoints.drop_duplicates(["hit_id", "cluster_index_1", "cluster_index_2", "particle_id"]).fillna(-1)
    
    return spacepoints

def get_truth_spacepoints(pixel_spacepoints, strip_spacepoints, clusters, spacepoints_datatypes):

    # # Build truth list of spacepoints by handling matching clusters
    truth_spacepoints = truth_match_clusters(pixel_spacepoints, strip_spacepoints, clusters)
    # # Tidy up the truth dataframe and add in all cluster information
    truth_spacepoints["cluster_index_2"] = truth_spacepoints["cluster_index_2"].fillna(-1)
    truth_spacepoints = merge_spacepoints_clusters(truth_spacepoints, clusters)

    # # Fix spacepoint datatypes
    truth_spacepoints = truth_spacepoints.astype(spacepoints_datatypes)

    return truth_spacepoints
