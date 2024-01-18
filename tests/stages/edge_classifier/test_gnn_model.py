import sys

sys.path.append("../../acorn")

import yaml
import pytest


def test_model_load():
    """
    Test the model to ensure it is of the right format and loaded correctly. It uses the configuration given in test_gnn_config.yaml.
    """
    from acorn.stages.edge_classifier import InteractionGNN

    # load test_gnn_config.yaml
    with open("stages/edge_classifier/test_gnn_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = InteractionGNN(config)

    assert model is not None


def test_data_load():
    """
    Test the data to ensure it is of the right format and loaded correctly.
    Runs X tests:
    1. Test a correct data load
    2. Test a data load with no testset
    3. Test a data load without enough events
    4. Missing directory
    """
    from acorn.stages.edge_classifier import InteractionGNN

    # load test_gnn_config.yaml
    with open("stages/edge_classifier/test_gnn_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Test 1
    config["data_split"] = [1, 1, 1]
    model = InteractionGNN(config)
    setup_and_test(model)

    # Test 2
    config["data_split"] = [1, 1, 0]
    model = InteractionGNN(config)
    pytest.raises(AssertionError, setup_and_test, model)

    # Test 3
    config["data_split"] = [100, 1, 1]
    model = InteractionGNN(config)

    pytest.raises(AssertionError, model.setup, stage="fit")

    # Test 4
    config["data_split"] = [1, 1, 1]
    config["input_dir"] = "a_missing_directory"
    model = InteractionGNN(config)

    pytest.raises(AssertionError, model.setup, stage="fit")


def setup_and_test(model):
    model.setup(stage="fit")

    assert model.trainset is not None
    assert model.valset is not None
    assert model.testset is not None


def test_construct_weighting():
    """
    TODO
    """

    pass
