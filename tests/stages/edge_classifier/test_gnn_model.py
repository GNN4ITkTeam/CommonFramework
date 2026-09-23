import sys

sys.path.append("../../acorn")

import pytest
import torch
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


@pytest.mark.requires_data
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


IGNN2_TEST_CONFIG = "stages/edge_classifier/test_ignn2_config.yaml"

# Reference values below are CPU values, so these tests pin model and batch to CPU
# regardless of what hardware the suite runs on.
CPU = torch.device("cpu")


def _make_ignn2_batch(config, n_nodes=20, n_edges=40, seed=54321):
    from torch_geometric.data import Data

    generator = torch.Generator(device=CPU).manual_seed(seed)

    def random(*shape):
        return torch.empty(*shape, device=CPU).uniform_(-1.0, 1.0, generator=generator)

    features = {feature: random(n_nodes) for feature in config["node_features"]} | {
        feature: random(n_edges) for feature in config["edge_features"]
    }

    edge_index = torch.randint(
        0, n_nodes, (2, n_edges), generator=generator, device=CPU
    )

    # Truth and weights covering all three branches of loss_function: negative
    # (y == 0, weight != 0), positive target (y == 1, weight > 0) and true
    # background that contributes to neither (y == 1, weight == 0).
    edge_y = torch.zeros(n_edges, device=CPU)
    edge_y[::2] = 1.0
    edge_weights = torch.full((n_edges,), 0.1, device=CPU)
    edge_weights[::2] = 1.0
    edge_weights[::6] = 0.0

    return Data(
        edge_index=edge_index,
        num_nodes=n_nodes,
        edge_y=edge_y,
        edge_weights=edge_weights,
        **features,
    )


def _make_ignn2_model(config, seed=12345):
    from acorn.stages.edge_classifier import InteractionGNN2

    assert config["n_graph_iters"] > 1, "must cover more than one message passing step"

    model = InteractionGNN2(config).to(CPU)

    # Fill parameters deterministically instead of relying on each layer's default
    # init, since that init is not what these tests are meant to pin down.
    generator = torch.Generator(device=CPU).manual_seed(seed)
    with torch.no_grad():
        for _, parameter in sorted(model.named_parameters()):
            parameter.copy_(
                torch.empty_like(parameter).uniform_(-0.5, 0.5, generator=generator)
            )

    model.eval()

    return model


def test_ignn2_forward_output_is_unchanged():
    """
    Pin the numerical output of InteractionGNN2.forward for a fixed model and a
    fixed batch, so that unintended changes to the message passing are caught.
    """
    # Reference output for the batch built by _make_ignn2_batch() and the model
    # built by _make_ignn2_model(). Regenerate only when a change to the model
    # is intended.
    expected_output = torch.tensor(
        [
            -0.5434284, -0.5014151, -0.5240216, -0.5446645,
            -0.5330972, -0.5036340, -0.5199699, -0.5368218,
            -0.5389072, -0.5179747, -0.5435340, -0.5170665,
            -0.5206354, -0.5364789, -0.5306903, -0.5044976,
            -0.5335143, -0.5233094, -0.5441040, -0.5273607,
            -0.5362947, -0.5450839, -0.5138291, -0.5339299,
            -0.5209052, -0.5209703, -0.5523346, -0.5281319,
            -0.5210399, -0.5388852, -0.5335168, -0.5233228,
            -0.5314506, -0.5382625, -0.5191684, -0.5461044,
            -0.4991466, -0.5349424, -0.4993181, -0.5150883,
        ]
    )  # fmt: skip

    with open(IGNN2_TEST_CONFIG, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = _make_ignn2_model(config)
    batch = _make_ignn2_batch(config)
    output = model(batch).detach()

    assert output.shape == (batch.edge_index.shape[1],)
    assert output.device == CPU
    torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-6)


def test_ignn2_backward_step_is_unchanged():
    """
    Pin the loss and the per-parameter gradient norms of a single training step
    of InteractionGNN2, so that unintended changes to the backward pass or to the
    loss are caught as well.
    """
    # [loss, positive_loss, negative_loss], regenerate together with
    # expected_grad_norms only when a change to the model is intended.
    expected_losses = torch.tensor([0.9218263, 0.8599558, 0.0618705])

    # Gradient norms in sorted-by-name order. The trailing zeros are
    # node_network[n_graph_iters - 1]: its x_updated is discarded, so it never trains.
    expected_grad_norms = torch.tensor(
        [
            0.3140384, 0.3909341, 0.4685329, 0.9019760,
            0.0078479, 0.0149439, 0.0088262, 0.0178993,
            0.0166832, 0.0380882, 0.0078102, 0.0257531,
            0.0311653, 0.0460890, 0.0158423, 0.0309345,
            0.0591234, 0.0631964, 0.1167666, 0.2432865,
            0.8003574, 0.8075249, 0.4958490, 1.0061685,
            0.0207137, 0.0515902, 0.0188384, 0.0370879,
            0.0111740, 0.0287145, 0.0296158, 0.0588863,
            0.0252395, 0.0807320, 0.0314904, 0.0614272,
            0.0000000, 0.0000000, 0.0000000, 0.0000000,
        ]
    )  # fmt: skip

    with open(IGNN2_TEST_CONFIG, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = _make_ignn2_model(config)
    batch = _make_ignn2_batch(config)

    model.zero_grad()
    output = model(batch)
    loss, positive_loss, negative_loss = model.loss_function(output, batch)
    loss.backward()

    named_parameters = sorted(model.named_parameters())
    names = [name for name, _ in named_parameters]
    losses = torch.stack([loss, positive_loss, negative_loss]).detach()
    grad_norms = torch.stack(
        [
            torch.zeros((), device=CPU)
            if parameter.grad is None
            else parameter.grad.norm()
            for _, parameter in named_parameters
        ]
    ).detach()

    # The last message step's x_updated is never used, so only that node network
    # is expected to be untrained.
    last_node_network = f"node_network.{config['n_graph_iters'] - 1}."
    untrained = {name for name, norm in zip(names, grad_norms) if norm == 0}
    assert untrained == {name for name in names if name.startswith(last_node_network)}

    assert losses.device == CPU and grad_norms.device == CPU
    torch.testing.assert_close(losses, expected_losses, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_norms, expected_grad_norms, rtol=1e-5, atol=1e-6)


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

    Per-pt-cut target efficiency, pooled over both batches. edge_pt comes from the
    tracks, so edges no track maps onto stay at -1 and fall below every cut:
      edge_pt = [2000, 500, 500, 3000, 1500] and [3000, 500, 1000, 800]
      pt >= 1000: batch 1 edges 0, 3, 4 -> 2 recalled of 3; batch 2 edges 0, 2 -> 1 of 1
                  target_eff_pt1000 = 3/4
      pt >= 2000: batch 1 edges 0, 3    -> 1 recalled of 2; batch 2 edge 0    -> 1 of 1
                  target_eff_pt2000 = 2/3
    """
    import torch

    from torch_geometric.data import Data

    from acorn.stages.edge_classifier.edge_classifier_stage import EdgeClassifierStage

    model = EdgeClassifierStage(
        {"edge_cut": 0.5, "validation_eff_pt_cuts": [1000, 2000]}
    )
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

    # The trailing -1 is a track without a matching edge, which must be ignored.
    batch_1 = Data(
        track_to_edge_map=torch.tensor([0, 1, 2, 3, 4, -1]),
        track_particle_pt=torch.tensor([2000.0, 500.0, 500.0, 3000.0, 1500.0, 5000.0]),
    )
    batch_2 = Data(
        track_to_edge_map=torch.tensor([0, 1, 2, 3]),
        track_particle_pt=torch.tensor([3000.0, 500.0, 1000.0, 800.0]),
    )

    legacy_preds_1, legacy_metrics_1 = _old_log_metrics_for_test(
        output_1, all_truth_1, target_truth_1, edge_cut=0.5
    )
    legacy_preds_2, legacy_metrics_2 = _old_log_metrics_for_test(
        output_2, all_truth_2, target_truth_2, edge_cut=0.5
    )

    preds_1 = model.log_metrics(
        batch_1, output_1, all_truth_1, target_truth_1, loss=torch.tensor(0.0)
    )
    preds_2 = model.log_metrics(
        batch_2, output_2, all_truth_2, target_truth_2, loss=torch.tensor(0.0)
    )
    model.on_validation_epoch_end()

    expected_metrics = {
        "target_eff": 4 / 5,
        "target_pur": 4 / 7,
        "total_pur": 6 / 7,
        "pur": 4 / 5,
        "auc": 13 / 20,
        "total_auc": 19 / 28,
        "target_eff_pt1000": 3 / 4,
        "target_eff_pt2000": 2 / 3,
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
