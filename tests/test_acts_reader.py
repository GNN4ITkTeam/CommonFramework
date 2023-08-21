import sys

sys.path.append("../gnn4itk_cf")

import pytest
from pathlib import Path

import pandas as pd

from gnn4itk_cf.stages.data_reading.models.acts_reader import ActsReader

common_config = {
    "simhit_stem": "hits",
    "detector_path": "test_acts_reader/detectors.csv",
    "data_split": [1, 0, 0],
    "feature_sets": {
        "hit_features": ["hit_id", "x", "y", "z", "r", "phi", "eta", "region"],
        "track_features": ["particle_id", "pt", "radius", "nhits", "particle_eta"],
    },
    "max_workers": 1,
    "region_labels": {
        1: {"volume_id": 16},
        2: {"volume_id": [23, 28]},
        3: {"volume_id": 17},
        4: {"volume_id": [24, 29]},
        5: {"volume_id": 18},
        6: {"volume_id": [25, 30]},
    },
}


@pytest.mark.parametrize("true_hits,tol", [[True, 1.0e-9], [False, 1.0e-3]])
def test_reader_positions(tmp_path, true_hits, tol):
    config = common_config.copy()
    config["use_truth_hits"] = true_hits
    config["stage_dir"] = tmp_path
    config["input_dir"] = "test_acts_reader/exact"

    reader = ActsReader(config)
    reader.convert_to_csv()

    test_particles = pd.read_csv(
        Path(tmp_path) / "trainset/event000000000-particles.csv"
    )
    test_truth = pd.read_csv(Path(tmp_path) / "trainset/event000000000-truth.csv")

    ref_particles = pd.read_csv(
        "test_acts_reader/exact/event000000000-particles_initial.csv"
    )
    ref_truth = pd.read_csv("test_acts_reader/exact/event000000000-hits.csv")
    ref_meas = pd.read_csv("test_acts_reader/exact/event000000000-measurements.csv")

    assert len(ref_truth) == len(ref_meas) == len(test_truth)
    assert len(ref_particles) == len(test_particles)

    assert (ref_truth.geometry_id == test_truth.geometry_id).all()

    assert (
        abs(
            ref_truth[["tx", "ty", "tz"]].to_numpy()
            - test_truth[["x", "y", "z"]].to_numpy()
        )
        < tol
    ).all()


@pytest.mark.parametrize("true_hits", [True, False])
def test_cell_information(tmp_path, true_hits):
    cell_features = [
        "cell_count",
        "cell_val",
        "leta",
        "lphi",
        "lx",
        "ly",
        "lz",
        "geta",
        "gphi",
    ]

    config = common_config.copy()
    config["use_truth_hits"] = true_hits
    config["stage_dir"] = tmp_path
    config["input_dir"] = "test_acts_reader/geometric"
    config["feature_sets"]["hit_features"] += cell_features

    reader = ActsReader(config)
    reader.convert_to_csv()

    test_truth = pd.read_csv(Path(tmp_path) / "trainset/event000000000-truth.csv")

    for feat in cell_features:
        assert feat in test_truth

    test_particles = pd.read_csv(
        Path(tmp_path) / "trainset/event000000000-particles.csv"
    )
    test_truth = pd.read_csv(Path(tmp_path) / "trainset/event000000000-truth.csv")

    ref_meas = pd.read_csv("test_acts_reader/geometric/event000000000-measurements.csv")
    assert len(ref_meas) == len(test_truth)
