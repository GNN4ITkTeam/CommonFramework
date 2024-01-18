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

translator = {
    "Part_event_number": "subevent",
    "Part_barcode": "barcode",
    "Part_px": "px",
    "Part_py": "py",
    "Part_pz": "pz",
    "Part_pt": "pt",
    "Part_eta": "eta",
    "Part_vx": "vx",
    "Part_vy": "vy",
    "Part_vz": "vz",
    "Part_radius": "radius",
    "Part_status": "status",
    "Part_charge": "charge",
    "Part_pdg_id": "pdgId",
    "Part_passed": "pass",
    "Part_vProdNin": "vProdNIn",
    "Part_vProdNout": "vProdNOut",
    "Part_vProdStatus": "vProdStatus",
    "Part_vProdBarcode": "vProdBarcode",
    "SPindex": "hit_id",
    "SPx": "x",
    "SPy": "y",
    "SPz": "z",
    "SPCL1_index": "cluster_index_1",
    "SPCL2_index": "cluster_index_2",
    "CLindex": "cluster_id",
    "CLhardware": "hardware",
    "CLx": "cluster_x",
    "CLy": "cluster_y",
    "CLz": "cluster_z",
    "CLbarrel_endcap": "barrel_endcap",
    "CLlayer_disk": "layer_disk",
    "CLeta_module": "eta_module",
    "CLphi_module": "phi_module",
    "CLside": "side",
    "CLmoduleID": "module_id",  # warning, this field is not present in origianl .txt dump
    "CLpixel_count": "count",
    "CLcharge_count": "charge_count",
    "CLloc_eta": "loc_eta",
    "CLloc_phi": "loc_phi",
    "CLloc_direction1": "localDir0",
    "CLloc_direction2": "localDir1",
    "CLloc_direction3": "localDir2",
    "CLJan_loc_direction1": "lengthDir0",
    "CLJan_loc_direction2": "lengthDir1",
    "CLJan_loc_direction3": "lengthDir2",
    "CLglob_eta": "glob_eta",
    "CLglob_phi": "glob_phi",
    "CLeta_angle": "eta_angle",
    "CLphi_angle": "phi_angle",
    "CLnorm_x": "norm_x",
    "CLnorm_y": "norm_y",
    "CLnorm_z": "norm_z",
    "CLparticleLink_eventIndex": "subevent",
    "CLparticleLink_barcode": "barcode",
}

event_branch_names = ["run_number", "event_number"]
particle_branch_names = [b for b in translator.keys() if b.startswith("Part_")]
particle_col_names = [c for b, c in translator.items() if b.startswith("Part_")]
spacepoint_branch_names = [b for b in translator.keys() if b.startswith("SP")]
spacepoint_col_names = [c for b, c in translator.items() if b.startswith("SP")]
cluster_branch_names = [b for b in translator.keys() if b.startswith("CL")]
cluster_col_names = [c for b, c in translator.items() if b.startswith("CL")]

all_branches = (
    event_branch_names
    + particle_branch_names
    + spacepoint_branch_names
    + cluster_branch_names
)  # + ["CLhardware"] #\
# + cluster_link_branch_names


def translate_branch_name(root_branch_name: str):
    """
    Translate Branch names of Dumps in ROOT to pandas Dataframe columns used in common framework
    """
    if root_branch_name not in translator.keys():
        raise ValueError(
            f"Unknown ROOT branch name {root_branch_name}, cannot translate it."
        )

    return translator[root_branch_name]


def uproot2pandaDF(branches: dict):
    """
    Make pandas dataframe from ROOT branches (already transformed in dict of np arrays)
    """
    arrays = dict()

    for k, val in branches.items():
        # sanity check
        if len(val) == 0:
            return None
        if len(val) > 1:
            warnings.warn("WARNING val has length larger than 1:", len(val))

        if "particleLink" in k:
            arrays[translate_branch_name(k)] = val[
                0
            ].tolist()  # special case to deal with vector<vector<>>
        else:
            arrays[translate_branch_name(k)] = np.asarray(
                val[0]
            )  # val must have always length 1 (only one event)

    # Make the 'flat' panda data frame
    df = pd.DataFrame.from_dict(arrays)

    return df


def read_particles(branches: dict):
    """
    Parse the dict of np arrays made from the athena dump in ROOT, for only one event
    return a panda dataframe containing each particles
    """
    particles = uproot2pandaDF(branches)
    if particles is None:
        return None

    # Prepare barcodes for later processing
    particles = particles.astype({"barcode": str, "subevent": str})

    return particles


def read_spacepoints(branches: dict):
    """
    Parse the dict of np arrays made from the athena dump in ROOT, for only one event
    return a panda dataframe containing the space points
    """
    spacepoints = uproot2pandaDF(branches)

    # print(spacepoints)

    # Wherever cluster_index_2 is NaN (i.e. for pixel hits), replace with cluster_index_1
    # small hack because in root dump we have -1 for pixel cluster_index_2
    spacepoints["cluster_index_2"] = spacepoints["cluster_index_2"].replace(-1, np.nan)
    spacepoints["cluster_index_2"] = (
        spacepoints["cluster_index_2"]
        .fillna(spacepoints["cluster_index_1"])
        .astype("int64")
    )

    # To align types as in txt file reading
    spacepoints = spacepoints.astype({"cluster_index_1": "int64"})
    spacepoints = spacepoints.astype({"hit_id": "int64"})

    # Check that hit_ids are a sequential list, if they're not, set them as such and return a warning
    if not spacepoints.hit_id.equals(pd.Series(range(len(spacepoints)))):
        warnings.warn("Hit IDs are not sequential, fixing")
        spacepoints["hit_id"] = range(len(spacepoints))

    return spacepoints


def read_clusters(branches: dict, particles: pd.DataFrame):
    """
    Parse the dict of np arrays made from the athena dump in ROOT, for only one event
    return a panda dataframe containing the clusters
    """

    clusters = uproot2pandaDF(branches)

    # print("\nClusters prelim\n")
    # print(clusters)
    # print(clusters.dtypes)

    # Fix indexing mismatch in DumpObjects
    # clusters['cluster_id'] = clusters['cluster_id'] - 1

    # Make a dataframe with only clusters associated to a particle
    # If a cluster is associated to multiple particles, split in several rows
    split_pids = pd.DataFrame(
        [
            [c, e, b]
            for c, E, B in clusters[["cluster_id", "subevent", "barcode"]].itertuples(
                index=False
            )
            for e, b in zip(E, B)
        ],
        columns=["cluster_id", "subevent", "barcode"],
    )

    split_pids = split_pids.astype({"subevent": str, "barcode": str})

    # print("\nSplit pids\n")
    # print(split_pids)
    # print(split_pids.dtypes)

    # Not sure why this merge is needed, but make as Daniel did for txt reader
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

    # Keep only need info
    split_pids = split_pids[["cluster_id", "particle_id"]]

    # print("\nSplit pids 2\n")
    # print(split_pids)
    # print(split_pids.dtypes)

    clusters = clusters.merge(split_pids, on="cluster_id", how="left").astype(
        {"cluster_id": int}
    )

    # Use same types as in txt reading
    # clusters = clusters.astype({'hardware':str, 'barrel_endcap':str, 'layer_disk':str, 'eta_module':str, 'phi_module':str})
    # clusters = clusters.astype( { k:str for k in cluster_col_names if k!='cluster_id' } )
    clusters["particle_id"] = clusters["particle_id"].fillna(0).astype(int)

    # Fix indexing mismatch in DumpObjects
    clusters["cluster_id"] = clusters["cluster_id"] - 1

    # Remove useless coloumns (particle_id has now all the info)
    clusters = clusters.drop(["subevent", "barcode"], axis=1)

    return clusters


# To get the same column order as with txt reading
truth_col_order = [
    "hit_id",
    "x",
    "y",
    "z",
    "cluster_index_1",
    "cluster_index_2",
    "hardware",
    "cluster_x_1",
    "cluster_y_1",
    "cluster_z_1",
    "barrel_endcap",
    "layer_disk",
    "eta_module",
    "phi_module",
    "side_1",
    "norm_x_1",
    "norm_y_1",
    "norm_z_1",
    "count_1",
    "charge_count_1",
    "loc_eta_1",
    "loc_phi_1",
    "localDir0_1",
    "localDir1_1",
    "localDir2_1",
    "lengthDir0_1",
    "lengthDir1_1",
    "lengthDir2_1",
    "glob_eta_1",
    "glob_phi_1",
    "eta_angle_1",
    "phi_angle_1",
    "particle_id_1",
    "cluster_x_2",
    "cluster_y_2",
    "cluster_z_2",
    "side_2",
    "norm_x_2",
    "norm_y_2",
    "norm_z_2",
    "count_2",
    "charge_count_2",
    "loc_eta_2",
    "loc_phi_2",
    "localDir0_2",
    "localDir1_2",
    "localDir2_2",
    "lengthDir0_2",
    "lengthDir1_2",
    "lengthDir2_2",
    "glob_eta_2",
    "glob_phi_2",
    "eta_angle_2",
    "phi_angle_2",
    "particle_id_2",
    "particle_id",
    "region",
    "module_id",
]

particles_col_order = [
    "particle_id",
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
    "num_clusters",
]
