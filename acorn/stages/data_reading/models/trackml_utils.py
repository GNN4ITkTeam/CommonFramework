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

import numpy as np
import pandas as pd
import logging


#####################################################
#                   UTILD PANDAS                    #
#####################################################
def select_min(test_val, current_val):
    return min(test_val, current_val)


def select_max(test_val, current_val):
    if current_val == -1:
        return test_val
    else:
        return max(test_val, current_val)


def find_ch0_min(cells_in, nb_hits):
    cell_idx = cells_in.index.values.reshape(-1, 1)
    cells = cells_in[["hit_id", "ch0"]].values
    where_min = find_ch0_property(cells, nb_hits, select_min, 10**8)
    return where_min


def find_ch0_max(cells_in, nb_hits):
    cells = cells_in[["hit_id", "ch0"]].values
    where_max = find_ch0_property(cells, nb_hits, select_max, -(10**8))
    return where_max


def find_ch0_property(cells, nb_hits, comparator, init_val):
    nb_cells = cells.shape[0]
    cells = sort_cells_by_hit_id(cells)

    hit_property = [init_val] * nb_hits
    cell_property = [0] * nb_cells
    cell_values = cells[:, 2].tolist()
    hit_ids = cells[:, 1].tolist()

    hit_property_id = 0
    current_hit_id = hit_ids[0]
    for i, (h, v) in enumerate(zip(hit_ids, cell_values)):
        if h > current_hit_id:
            hit_property_id += 1
            current_hit_id = h
        hit_property[hit_property_id] = comparator(v, hit_property[hit_property_id])

    hit_property_id = 0
    current_hit_id = hit_ids[0]
    for i, (h, v) in enumerate(zip(hit_ids, cell_values)):
        if h > current_hit_id:
            hit_property_id += 1
            current_hit_id = h
        if v == hit_property[hit_property_id]:
            cell_property[i] = 1

    original_order = np.argsort(cells[:, 0])
    cell_property = np.array(cell_property, dtype=bool)[original_order]
    return cell_property


def sort_cells_by_hit_id(cells):
    orig_order = np.arange(len(cells)).reshape(-1, 1)
    cells = np.concatenate((orig_order, cells), 1)
    sort_idx = np.argsort(cells[:, 1])  # Sort by hit ID
    cells = cells[sort_idx]
    return cells


#################################################
#                   EXTRACT DIR                 #
#################################################


def local_angle(cell, module):
    n_u = max(cell["ch0"]) - min(cell["ch0"]) + 1
    n_v = max(cell["ch1"]) - min(cell["ch1"]) + 1
    l_u = n_u * module.pitch_u.values  # x
    l_v = n_v * module.pitch_v.values  # y
    l_w = 2 * module.module_t.values  # z
    return (l_u, l_v, l_w)


def extract_rotation_matrix(module):
    rot_matrix = np.matrix(
        [
            [module.rot_xu.values[0], module.rot_xv.values[0], module.rot_xw.values[0]],
            [module.rot_yu.values[0], module.rot_yv.values[0], module.rot_yw.values[0]],
            [module.rot_zu.values[0], module.rot_zv.values[0], module.rot_zw.values[0]],
        ]
    )
    return rot_matrix, np.linalg.inv(rot_matrix)


def cartesion_to_spherical(x, y, z):
    r3 = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r3)
    return r3, theta, phi


def theta_to_eta(theta):
    return -np.log(np.tan(0.5 * theta))


def get_all_local_angles(hits, cells, detector):
    direction_count_u = cells.groupby(["hit_id"]).ch0.agg(["min", "max"])
    direction_count_v = cells.groupby(["hit_id"]).ch1.agg(["min", "max"])
    nb_u = direction_count_u["max"] - direction_count_u["min"] + 1
    nb_v = direction_count_v["max"] - direction_count_v["min"] + 1

    vols = hits["volume_id"].values
    layers = hits["layer_id"].values
    modules = hits["module_id"].values

    pitch = detector["pixel_size"]
    thickness = detector["thicknesses"]

    pitch_cells = pitch[vols, layers, modules]
    thickness_cells = thickness[vols, layers, modules]

    l_u = nb_u * pitch_cells[:, 0]
    l_v = nb_v * pitch_cells[:, 1]
    l_w = 2 * thickness_cells
    return l_u, l_v, l_w


def get_all_rotated(hits, detector, l_u, l_v, l_w):
    vols = hits["volume_id"].values
    layers = hits["layer_id"].values
    modules = hits["module_id"].values
    rotations = detector["rotations"]
    rotations_hits = rotations[vols, layers, modules]
    u = l_u.values.reshape(-1, 1)
    v = l_v.values.reshape(-1, 1)
    w = l_w.reshape(-1, 1)
    dirs = np.concatenate((u, v, w), axis=1)

    dirs = np.expand_dims(dirs, axis=2)
    vecRot = np.matmul(rotations_hits, dirs).squeeze(2)
    return vecRot


def extract_dir_new(hits, cells, detector):
    l_u, l_v, l_w = get_all_local_angles(hits, cells, detector)
    g_matrix_all = get_all_rotated(hits, detector, l_u, l_v, l_w)
    hit_ids = hits["hit_id"].to_numpy()

    l_u, l_v = l_u.to_numpy(), l_v.to_numpy()

    _, g_theta, g_phi = np.vstack(cartesion_to_spherical(*list(g_matrix_all.T)))
    logging.info("G calc")
    _, l_theta, l_phi = cartesion_to_spherical(l_u, l_v, l_w)
    logging.info("L calc")
    l_eta = theta_to_eta(l_theta)
    g_eta = theta_to_eta(g_theta)

    angles = np.vstack([hit_ids, l_eta, l_phi, l_u, l_v, l_w, g_eta, g_phi]).T
    logging.info("Concated")
    df_angles = pd.DataFrame(
        angles,
        columns=[
            "hit_id",
            "leta",
            "lphi",
            "lx",
            "ly",
            "lz",
            "geta",
            "gphi",
        ],
    )
    logging.info("DF constructed")

    return df_angles


def check_diff(h1, h2, name):
    n1 = h1[name].values
    n2 = h2[name].values
    print(name, max(np.absolute(n1 - n2)))


#############################################
#           FEATURE_AUGMENTATION            #
#############################################


def augment_hit_features(hits, cells, detector_orig, detector_proc):
    cell_stats = get_cell_stats(cells)
    hits["cell_count"] = cell_stats[:, 0]
    hits["cell_val"] = cell_stats[:, 1]

    angles = extract_dir_new(hits, cells, detector_proc)

    return angles


def get_cell_stats(cells):
    hit_cells = cells.groupby(["hit_id"]).value.count().values
    hit_value = cells.groupby(["hit_id"]).value.sum().values
    cell_stats = np.hstack((hit_cells.reshape(-1, 1), hit_value.reshape(-1, 1)))
    cell_stats = cell_stats.astype(np.float32)
    return cell_stats


def add_region_labels(hits, region_labels: dict):
    """
    Label the 6 detector regions (forward-endcap pixel, forward-endcap strip, etc.)
    """

    for region_label, conditions in region_labels.items():
        condition_mask = np.logical_and.reduce(
            [
                hits[condition_column].isin(condition)
                if isinstance(condition, list)
                else hits[condition_column] == condition
                for condition_column, condition in conditions.items()
            ]
        )
        hits.loc[condition_mask, "region"] = region_label

    assert (
        hits.region.isna()
    ).sum() == 0, "There are hits that do not belong to any region!"

    return hits


###########################################
#           CELL INFO LOADING             #
###########################################


def add_cell_info(truth, cells, detector_dims):
    cell_stats = get_cell_stats(cells)
    truth["cell_count"] = cell_stats[:, 0]
    truth["cell_val"] = cell_stats[:, 1]

    angles = extract_dir_new(truth, cells, detector_dims)
    truth = pd.merge(truth, angles, on="hit_id")

    return truth


#############################################
#               DETECTOR UTILS              #
#############################################


def load_detector(detector_path):
    detector_df = pd.read_csv(detector_path)
    detector_dims = preprocess_detector(detector_df)
    return detector_df, detector_dims


def preprocess_detector(detector):
    thicknesses = Detector_Thicknesses(detector).get_thicknesses()
    rotations = Detector_Rotations(detector).get_rotations()
    pixel_size = Detector_Pixel_Size(detector).get_pixel_size()
    return dict(thicknesses=thicknesses, rotations=rotations, pixel_size=pixel_size)


def determine_array_size(detector):
    max_v, max_l, max_m = (0, 0, 0)
    unique_vols = detector.volume_id.unique()
    max_v = max(unique_vols) + 1
    for v in unique_vols:
        vol = detector.loc[detector["volume_id"] == v]
        unique_layers = vol.layer_id.unique()
        max_l = max(max_l, max(unique_layers) + 1)
        for l in unique_layers:
            lay = vol.loc[vol["layer_id"] == l]
            unique_modules = lay.module_id.unique()
            max_m = max(max_m, max(unique_modules) + 1)
    return max_v, max_l, max_m


class Detector_Rotations(object):
    def __init__(self, detector):
        self.detector = detector
        self.max_v, self.max_l, self.max_m = determine_array_size(detector)

    def get_rotations(self):
        self._init_rotation_array()
        self._extract_all_rotations()
        return self.rot

    def _init_rotation_array(self):
        self.rot = np.zeros((self.max_v, self.max_l, self.max_m, 3, 3))

    def _extract_all_rotations(self):
        for i, r in self.detector.iterrows():
            v, l, m = tuple(map(int, (r.volume_id, r.layer_id, r.module_id)))
            rot = self._extract_rotation_matrix(r)
            self.rot[v, l, m] = rot

    def _extract_rotation_matrix(self, mod):
        """
        Extract the rotation matrix from module dataframe
        """
        r = np.matrix(
            [
                [mod.rot_xu.item(), mod.rot_xv.item(), mod.rot_xw.item()],
                [mod.rot_yu.item(), mod.rot_yv.item(), mod.rot_yw.item()],
                [mod.rot_zu.item(), mod.rot_zv.item(), mod.rot_zw.item()],
            ]
        )
        return r


class Detector_Thicknesses(object):
    def __init__(self, detector):
        self.detector = detector
        self.max_v, self.max_l, self.max_m = determine_array_size(detector)

    def get_thicknesses(self):
        self._init_thickness_array()
        self._extract_all_thicknesses()
        return self.all_t

    def _init_thickness_array(self):
        self.all_t = np.zeros((self.max_v, self.max_l, self.max_m))

    def _extract_all_thicknesses(self):
        for i, r in self.detector.iterrows():
            v, l, m = tuple(map(int, (r.volume_id, r.layer_id, r.module_id)))
            self.all_t[v, l, m] = r.module_t


class Detector_Pixel_Size(object):
    def __init__(self, detector):
        self.detector = detector
        self.max_v, self.max_l, self.max_m = determine_array_size(detector)

    def get_pixel_size(self):
        self._init_size_array()
        self._extract_all_size()
        return self.all_s

    def _init_size_array(self):
        self.all_s = np.zeros((self.max_v, self.max_l, self.max_m, 2))

    def _extract_all_size(self):
        for i, r in self.detector.iterrows():
            v, l, m = tuple(map(int, (r.volume_id, r.layer_id, r.module_id)))
            self.all_s[v, l, m, 0] = r.pitch_u
            self.all_s[v, l, m, 1] = r.pitch_v
