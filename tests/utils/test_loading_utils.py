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
    load_datafiles_in_dir,
    load_pyg,
    pyg_exists,
    save_pyg,
)


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
