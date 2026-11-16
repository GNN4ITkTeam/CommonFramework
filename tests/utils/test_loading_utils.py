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

from acorn.stages.edge_classifier.edge_classifier_stage import GraphDataset
from acorn.utils.loading_utils import (
    handle_edge_features,
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


EDGE_FEATURES = [
    "edge_dr",
    "edge_dphi",
    "edge_dz",
    "edge_deta",
    "edge_phislope",
    "edge_rphislope",
]


@pytest.fixture
def edge_event():
    """An event whose edges include one that crosses the +/- pi discontinuity.

    Hits 0 and 1 sit 0.083 rad apart across the discontinuity, so the wrapped
    difference along edge 0 is small while the raw one is close to a full turn.
    Edge 2 joins two hits at the same radius, the case phislope clamps.
    """
    return Data(
        hit_r=torch.tensor([300.0, 310.0, 500.0, 500.0], dtype=torch.float64),
        hit_phi=torch.tensor([3.1, -3.1, 0.2, 0.4], dtype=torch.float64),
        hit_z=torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64),
        hit_eta=torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        num_nodes=4,
    )


def _scaled(event, phi_scale):
    """The event as ``scale_features`` leaves it: r and z in metres, phi / phi_scale."""
    scaled = event.clone()
    scaled.hit_r = scaled.hit_r / 1000.0
    scaled.hit_z = scaled.hit_z / 1000.0
    scaled.hit_phi = scaled.hit_phi / phi_scale
    return scaled


def test_handle_edge_features_does_not_depend_on_phi_scaling(edge_event):
    """Feeding the network cos/sin instead of phi must not change the edge features.

    Without phi among the node features, ``hit_phi`` stays in radians. Before
    ``phi_scale`` existed this inflated edge_dphi by pi and wrapped it at 1 rad,
    corrupting phislope and rphislope with it, all within a plausible range.
    """
    with_phi = _scaled(edge_event, torch.pi)
    with_cos_sin = _scaled(edge_event, 1.0)

    handle_edge_features(with_phi, EDGE_FEATURES, phi_scale=torch.pi)
    handle_edge_features(with_cos_sin, EDGE_FEATURES, phi_scale=1.0)

    for feature in EDGE_FEATURES:
        torch.testing.assert_close(with_cos_sin[feature], with_phi[feature])


def test_handle_edge_features_wraps_dphi_across_pi(edge_event):
    event = _scaled(edge_event, 1.0)

    handle_edge_features(event, ["edge_dphi"], phi_scale=1.0)

    expected = (2 * torch.pi - 6.2) / torch.pi
    torch.testing.assert_close(
        event.edge_dphi[0], torch.tensor(expected, dtype=torch.float64)
    )
    assert event.edge_dphi.abs().max() < 1.0


def test_handle_edge_features_default_assumes_phi_divided_by_pi(edge_event):
    """The default keeps the historical behaviour, for callers that pass nothing."""
    default = _scaled(edge_event, torch.pi)
    explicit = _scaled(edge_event, torch.pi)

    handle_edge_features(default, EDGE_FEATURES)
    handle_edge_features(explicit, EDGE_FEATURES, phi_scale=torch.pi)

    for feature in EDGE_FEATURES:
        assert torch.equal(default[feature], explicit[feature])


@pytest.mark.parametrize(
    "hparams,expected",
    [
        ({"node_features": ["hit_r", "hit_phi"], "node_scales": [1000.0, 3.14]}, 3.14),
        (
            {
                "node_features": ["hit_r", "hit_cosphi", "hit_sinphi"],
                "node_scales": [1000.0, 1.0, 1.0],
            },
            1.0,
        ),
        ({"node_features": ["hit_r", "hit_phi"]}, 1.0),
        ({}, 1.0),
    ],
)
def test_graph_dataset_phi_scale_follows_the_config(hparams, expected):
    """``hit_phi`` is only scaled when it is a node feature and scales are given."""
    dataset = GraphDataset.__new__(GraphDataset)
    dataset.hparams = hparams

    assert dataset.phi_scale() == expected
