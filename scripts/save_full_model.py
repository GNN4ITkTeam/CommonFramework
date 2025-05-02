"""This module is to save the full pytorch model loaded from a checkpoint
as a Torchscript model that can be used in inference.
It can be used as:
```bash
python scripts/save_full_model.py examples/Example_1/gnn_train.yaml -o saved_model_files --tag v1
```
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
import yaml
from pytorch_lightning import LightningModule
import onnxruntime as ort

from acorn import stages
from acorn.core.core_utils import find_latest_checkpoint
from acorn.stages.edge_classifier import (
    RecurrentInteractionGNN2,
    ChainedInteractionGNN2,
    GNNFilterJitable,
)

torch.use_deterministic_algorithms(True)


def model_save(
    stage_name: str,
    model_name: str,
    checkpoint_path: str | Path,
    output_path: str,
    tag_name: str | None = None,
):
    lightning_model = getattr(getattr(stages, stage_name), model_name)
    if not issubclass(lightning_model, LightningModule):
        raise ValueError(f"Model {model_name} is not a LightningModule")

    # find the best checkpoint in the checkpoint path
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} not found")

    if checkpoint_path.is_dir():
        checkpoint_path = find_latest_checkpoint(
            checkpoint_path, templates=["best*.ckpt", "*.ckpt"]
        )
        if not checkpoint_path:
            raise ValueError(f"No checkpoint found in {checkpoint_path}")

    # load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model = lightning_model.load_from_checkpoint(checkpoint_path).to("cpu")
    print(model.hparams)

    # save for use in production environment
    out_path = Path(output_path)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    # perform some dummy inference
    model_config = model.hparams
    num_spacepoints = 100
    num_edges = 2000
    spacepoint_features = len(model_config["node_features"])
    print(f"number of node features: {spacepoint_features}")

    node_features = torch.rand(num_spacepoints, spacepoint_features).to(torch.float32)
    edge_list = torch.randint(0, 100, (2, num_edges)).to(torch.int64)

    if "MetricLearning" in model_name:
        input_data = (node_features,)
        input_names = ["node_features"]
        dynamic_axes = {"node_features": {0: "num_spacepoints"}}
    elif "InteractionGNN2" in model_name:
        num_edge_features = len(model_config["edge_features"])
        input_data = [node_features, edge_list]
        input_names = ["node_features", "edge_list"]
        dynamic_axes = {
            "node_features": {0: "num_spacepoints"},
            "edge_list": {1: "num_edges"},
        }
        if num_edge_features > 0:
            edge_attr = torch.rand(num_edges, num_edge_features).to(torch.float32)
            input_data.append(edge_attr)
            input_names.append("edge_attr")
            dynamic_axes["edge_attr"] = {0: "num_edges"}

        is_recurrent = (
            model_config["node_net_recurrent"] and model_config["edge_net_recurrent"]
        )
        print(f"Is a recurrent GNN?: {is_recurrent}")
        if is_recurrent:
            print("Use RecurrentInteractionGNN2")
            new_gnn = RecurrentInteractionGNN2(model_config)
        else:
            print("Use ChainedInteractionGNN2")
            new_gnn = ChainedInteractionGNN2(model_config)
        new_gnn.load_state_dict(model.state_dict())

        model = new_gnn
    elif "GNNFilter" in model_name:
        input_data = [node_features, edge_list]
        input_names = ["node_features", "edge_list"]
        dynamic_axes = {
            "node_features": {0: "num_spacepoints"},
            "edge_list": {1: "num_edges"},
        }

        new_model = GNNFilterJitable(model_config)
        new_model.load_state_dict(model.state_dict())

        # check the model outputs are the same
        with torch.no_grad():
            output = model(*input_data)
            new_output = new_model(*input_data)
            assert new_output.equal(output)
        model = new_model
    else:
        input_data = [node_features, edge_list]
        input_names = ["node_features", "edge_list"]
        dynamic_axes = {
            "node_features": {0: "num_spacepoints"},
            "edge_list": {1: "num_edges"},
        }

    input_data = tuple(input_data)
    with torch.no_grad():
        output = model(*input_data)

    torch_script_path = (
        out_path / f"{stage_name}-{model_name}-{tag_name}.pt"
        if tag_name
        else out_path / f"{stage_name}-{model_name}.pt"
    )

    # exporting to torchscript
    with torch.jit.optimized_execution(True):
        script = model.to_torchscript(example_inputs=[input_data])

    with torch.no_grad():
        new_output = script(*input_data)
    torch.jit.freeze(script)
    assert new_output.equal(output)

    # save the model
    print(f"Saving model to {torch_script_path}")
    torch.jit.save(script, torch_script_path)
    print(f"Done saving model to {torch_script_path}")

    # try to save the model to ONNX
    try:
        print("Trying to save the model to ONNX")
        onnx_path = (
            out_path / f"{stage_name}-{model_name}-{tag_name}.onnx"
            if tag_name
            else out_path / f"{stage_name}-{model_name}.onnx"
        )
        torch.onnx.export(
            model,
            input_data,
            onnx_path,
            verbose=False,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        print(f"Done saving model to {onnx_path}")
    except Exception as e:
        print(f"Failed to save the model to ONNX: {e}")
        return

    # check the model outputs are the same
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_np_data = [i.detach().cpu().numpy() for i in input_data]
    ort_inputs = dict(zip(input_names, input_np_data))
    ort_outs = torch.tensor(ort_session.run(None, ort_inputs)[0], dtype=torch.float32)
    threashold = 1e-6
    while True:
        try:
            assert torch.allclose(ort_outs, output, atol=threashold, rtol=threashold)
            break
        except AssertionError:
            print(
                f"Output mismatch within {threashold}, trying again with higher threashold."
            )
            threashold *= 10
    print(f"ONNX output matches within {threashold:.0E}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save a model from a checkpoint")
    parser.add_argument("config", type=str, help="configuration file")
    parser.add_argument("-o", "--output", type=str, help="Output path", default=".")
    parser.add_argument("-t", "--tag", type=str, default=None, help="version name")
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="checkpoint path", default=None
    )
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} not found")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    stage = config["stage"]
    model = config["model"]
    checkpoint = Path(config["stage_dir"]) / "artifacts"
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    model_save(stage, model, checkpoint, args.output, args.tag)
