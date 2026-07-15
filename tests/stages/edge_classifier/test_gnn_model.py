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


def _old_log_metrics_for_test(output, all_truth, target_truth, edge_cut):
    import torch

    from torchmetrics import AUROC

    scores = torch.sigmoid(output)
    preds = scores > edge_cut

    edge_positive = preds.sum().float()
    target_true = target_truth.sum().float()
    target_true_positive = (target_truth.bool() & preds).sum().float()
    all_true_positive = (all_truth.bool() & preds).sum().float()

    total_auc = AUROC(task="binary", thresholds=None)(scores.float(), all_truth.long())

    mask = target_truth | (~all_truth)
    auc = AUROC(task="binary", thresholds=None)(
        scores[mask].float(), all_truth[mask].long()
    )
    true_and_fake_positive = (
        edge_positive - (preds & (~target_truth) & all_truth).sum().float()
    )

    return preds, {
        "eff": target_true_positive / target_true,
        "target_pur": target_true_positive / edge_positive,
        "total_pur": all_true_positive / edge_positive,
        "pur": target_true_positive / true_and_fake_positive,
        "auc": auc,
        "total_auc": total_auc,
    }


def test_epoch_end_metrics_are_pooled_from_hand_computed_counts():
    """
    Check hand-computable validation metrics over two batches.

    Pooled counts:
      edge_positive = 4 + 3 = 7
      target_true = 3 + 2 = 5
      target_true_positive = 2 + 2 = 4
      all_true_positive = 3 + 3 = 6
      true_and_fake_positive = 3 + 2 = 5

    New log_metrics outputs:
      target_eff = 4/5
      target_pur = 4/7
      total_pur = 6/7
      pur = 4/5

    Binned AUROC, matching AUROC(task="binary", thresholds=200):
      auc = 13/20
      total_auc = 19/28

    Old epoch-style means of per-batch log_metrics outputs:
      eff = 5/6
      target_pur = 7/12
      total_pur = 7/8
      pur = 5/6
      auc = 2/3
      total_auc = 11/16
    """
    import torch

    from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage

    model = EdgeClassifierStage({"edge_cut": 0.5})
    logged_metrics = {}

    model.log = lambda *args, **kwargs: None
    model.log_dict = lambda metrics, **kwargs: logged_metrics.update(metrics)
    model.optimizers = lambda: type(
        "OptimizerStub", (), {"param_groups": [{"lr": 0.001}]}
    )()

    output_1 = torch.tensor([2.0, 2.0, 2.0, -2.0, 2.0])
    all_truth_1 = torch.tensor([True, True, False, True, True])
    target_truth_1 = torch.tensor([True, False, False, True, True])

    output_2 = torch.tensor([2.0, -2.0, 2.0, 2.0])
    all_truth_2 = torch.tensor([True, False, True, True])
    target_truth_2 = torch.tensor([True, False, False, True])

    legacy_preds_1, legacy_metrics_1 = _old_log_metrics_for_test(
        output_1, all_truth_1, target_truth_1, edge_cut=0.5
    )
    legacy_preds_2, legacy_metrics_2 = _old_log_metrics_for_test(
        output_2, all_truth_2, target_truth_2, edge_cut=0.5
    )

    preds_1 = model.log_metrics(
        output_1, all_truth_1, target_truth_1, loss=torch.tensor(0.0)
    )
    preds_2 = model.log_metrics(
        output_2, all_truth_2, target_truth_2, loss=torch.tensor(0.0)
    )
    model.on_validation_epoch_end()

    expected_metrics = {
        "target_eff": 4 / 5,
        "target_pur": 4 / 7,
        "total_pur": 6 / 7,
        "pur": 4 / 5,
        "auc": 13 / 20,
        "total_auc": 19 / 28,
    }

    expected_legacy_epoch_metrics = {
        "eff": 5 / 6,
        "target_pur": 7 / 12,
        "total_pur": 7 / 8,
        "pur": 5 / 6,
        "auc": 2 / 3,
        "total_auc": 11 / 16,
    }
    legacy_epoch_metrics = {
        metric: (legacy_metrics_1[metric] + legacy_metrics_2[metric]) / 2
        for metric in expected_legacy_epoch_metrics
    }

    for metric, expected in expected_metrics.items():
        torch.testing.assert_close(
            logged_metrics[metric],
            torch.tensor(expected, device=logged_metrics[metric].device),
        )
    for metric, expected in expected_legacy_epoch_metrics.items():
        torch.testing.assert_close(
            legacy_epoch_metrics[metric],
            torch.tensor(expected, device=legacy_epoch_metrics[metric].device),
        )
        if metric in {
            "target_eff",
            "target_pur",
            "total_pur",
            "pur",
            "auc",
            "total_auc",
        }:
            assert not torch.isclose(
                logged_metrics[metric], legacy_epoch_metrics[metric]
            )

    assert torch.equal(preds_1, torch.tensor([True, True, True, False, True]))
    assert torch.equal(preds_2, torch.tensor([True, False, True, True]))
    assert torch.equal(preds_1, legacy_preds_1)
    assert torch.equal(preds_2, legacy_preds_2)
