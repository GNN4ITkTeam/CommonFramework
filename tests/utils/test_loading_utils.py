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

import os

import pytest
import torch
from torch_geometric.data import Data

from acorn.utils.loading_utils import (
    handle_node_features,
    load_datafiles_in_dir,
    load_pyg,
    pyg_exists,
    save_pyg,
)
from acorn.utils.version_utils import get_pyg_data_keys


@pytest.fixture
def mock_event():
    return Data(x=torch.rand(5, 3))


def test_save_pyg_load_pyg_roundtrip_no_subdir(tmp_path, mock_event):
    path = os.path.join(tmp_path, "event123.pyg")
    save_pyg(mock_event, path)

    assert os.path.exists(path + ".gz")
    assert not os.path.exists(path)

    loaded = load_pyg(path, weights_only=False)
    assert torch.equal(loaded.x, mock_event.x)


def test_save_pyg_with_subdir_shards_by_event_id(tmp_path, mock_event):
    path = os.path.join(tmp_path, "event250.pyg")
    save_pyg(mock_event, path, subdir=100)

    sharded_path = os.path.join(tmp_path, "2", "event250.pyg")
    assert os.path.exists(sharded_path + ".gz")
    assert not os.path.exists(path)
    assert not os.path.exists(path + ".gz")


@pytest.mark.parametrize(
    "event_id,expected_shard",
    [(0, "0"), (99, "0"), (100, "1"), (250, "2")],
)
def test_save_pyg_subdir_groups_events(tmp_path, mock_event, event_id, expected_shard):
    path = os.path.join(tmp_path, f"event{event_id}.pyg")
    save_pyg(mock_event, path, subdir=100)

    sharded_path = os.path.join(tmp_path, expected_shard, f"event{event_id}.pyg")
    assert os.path.exists(sharded_path + ".gz")


def test_save_pyg_subdir_supports_filename_prefixes_and_suffixes(tmp_path, mock_event):
    path = os.path.join(tmp_path, "run001_event250-graph.pyg")
    save_pyg(mock_event, path, subdir=100)

    sharded_path = os.path.join(tmp_path, "2", "run001_event250-graph.pyg")
    assert os.path.exists(sharded_path + ".gz")


def test_save_pyg_subdir_without_event_id_in_filename_raises(tmp_path, mock_event):
    path = os.path.join(tmp_path, "graph.pyg")
    with pytest.raises(ValueError):
        save_pyg(mock_event, path, subdir=100)


def test_pyg_exists_matches_save_pyg_subdir_location(tmp_path, mock_event):
    path = os.path.join(tmp_path, "event250.pyg")

    assert not pyg_exists(path, subdir=100)
    save_pyg(mock_event, path, subdir=100)
    assert pyg_exists(path, subdir=100)
    # Without subdir, the flat (unsharded) path should not be found
    assert not pyg_exists(path)


def test_load_datafiles_in_dir_finds_sharded_files(tmp_path, mock_event):
    for event_id in [0, 99, 100, 250]:
        path = os.path.join(tmp_path, f"event{event_id}.pyg")
        save_pyg(mock_event, path, subdir=100)

    data_files = load_datafiles_in_dir(str(tmp_path))
    assert len(data_files) == 4


@pytest.fixture
def angle_event():
    """An event carrying the angles that node features are derived from.

    `phi` spans a full turn, including the values either side of the +/- pi
    discontinuity that cos/sin exist to remove.
    """
    phi = torch.tensor([-torch.pi, -1.5, 0.0, 1.5, torch.pi - 1e-6])
    return Data(
        hit_r=torch.linspace(100.0, 900.0, 5),
        hit_phi=phi,
        hit_cluster_phi_1=phi.flip(0),
        num_nodes=5,
    )


def test_handle_node_features_derives_cos_and_sin(angle_event):
    phi = angle_event.hit_phi.clone()

    handle_node_features(angle_event, ["hit_r", "hit_cosphi", "hit_sinphi"])

    assert torch.equal(angle_event.hit_cosphi, torch.cos(phi))
    assert torch.equal(angle_event.hit_sinphi, torch.sin(phi))
    torch.testing.assert_close(
        angle_event.hit_cosphi**2 + angle_event.hit_sinphi**2, torch.ones_like(phi)
    )


def test_handle_node_features_does_nothing_when_not_requested(angle_event):
    before = set(get_pyg_data_keys(angle_event))

    handle_node_features(angle_event, ["hit_r", "hit_phi"])

    assert set(get_pyg_data_keys(angle_event)) == before


def test_handle_node_features_derives_any_matching_angle(angle_event):
    """The rule is textual, so config alone decides which angles are converted."""
    cluster_phi = angle_event.hit_cluster_phi_1.clone()

    handle_node_features(angle_event, ["hit_cluster_cosphi_1", "hit_cluster_sinphi_1"])

    assert torch.equal(angle_event.hit_cluster_cosphi_1, torch.cos(cluster_phi))
    assert torch.equal(angle_event.hit_cluster_sinphi_1, torch.sin(cluster_phi))


def test_handle_node_features_keeps_a_value_already_in_the_event(angle_event):
    """A value written by an earlier stage takes precedence over the derivation."""
    angle_event.hit_cosphi = torch.full_like(angle_event.hit_phi, 42.0)

    handle_node_features(angle_event, ["hit_cosphi"])

    assert torch.all(angle_event.hit_cosphi == 42.0)


def test_handle_node_features_raises_when_the_angle_is_missing(angle_event):
    with pytest.raises(ValueError, match="hit_phi_angle_1"):
        handle_node_features(angle_event, ["hit_cosphi_angle_1"])


def test_handle_node_features_must_run_before_scaling(angle_event):
    """Deriving from a scaled angle is wrong, and wrong within a plausible range.

    `scale_features` divides angles by pi, and cos(phi / pi) is not cos(phi) -- but
    it still lands in [-1, 1], so the mistake would not show up as an obviously
    broken feature. Pin the ordering here rather than rely on a comment.
    """
    scaled = Data(hit_phi=angle_event.hit_phi / torch.pi, num_nodes=5)

    handle_node_features(angle_event, ["hit_cosphi"])
    handle_node_features(scaled, ["hit_cosphi"])

    assert (angle_event.hit_cosphi - scaled.hit_cosphi).abs().max() > 0.5
