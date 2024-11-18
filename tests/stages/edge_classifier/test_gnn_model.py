import sys

sys.path.append("../../acorn")

import pytest
import yaml


def test_model_load():
    from acorn.stages.edge_classifier import InteractionGNN, InteractionGNN2

    _test_model_load("stages/edge_classifier/test_gnn_config.yaml", InteractionGNN)

    _test_model_load("stages/edge_classifier/test_ignn2_config.yaml", InteractionGNN2)


def _test_model_load(config_path, model_class):
    """
    Test the model to ensure it is of the right format and loaded correctly. It uses the configuration given in test_gnn_config.yaml.
    """
    # load test_gnn_config.yaml
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = model_class(config)

    assert model is not None


def test_data_load():
    from acorn.stages.edge_classifier import InteractionGNN, InteractionGNN2

    _test_data_load("stages/edge_classifier/test_gnn_config.yaml", InteractionGNN)

    _test_data_load("stages/edge_classifier/test_ignn2_config.yaml", InteractionGNN2)


def _test_data_load(config_path, model_class):
    """
    Test the data to ensure it is of the right format and loaded correctly.
    Runs X tests:
    1. Test a correct data load
    2. Test a data load with no testset
    3. Test a data load without enough events
    4. Missing directory
    """
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config

    # load test_gnn_config.yaml
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if not config.get("variable_with_prefix"):
        config = add_variable_name_prefix_in_config(config)

    # Test 1
    config["data_split"] = [1, 1, 1]
    model = model_class(config)
    setup_and_test(model)

    # Test 2
    config["data_split"] = [1, 1, 0]
    model = model_class(config)
    pytest.raises(AssertionError, setup_and_test, model)

    # Test 3
    config["data_split"] = [100, 1, 1]
    model = model_class(config)

    pytest.raises(AssertionError, model.setup, stage="fit")

    # Test 4
    config["data_split"] = [1, 1, 1]
    config["input_dir"] = "a_missing_directory"
    model = model_class(config)

    pytest.raises(AssertionError, model.setup, stage="fit")


def setup_and_test(model):
    model.setup(stage="fit")

    assert model.trainset is not None
    assert model.valset is not None
    assert model.testset is not None

    event = model.trainset.get(0)


def test_construct_weighting():
    """
    TODO
    """

    pass
