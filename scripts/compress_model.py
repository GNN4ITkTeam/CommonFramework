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

"""
This script compresses the NNs based on the applied pruning, identifiable as rows filled with 0s
"""

import torch
from torch import nn
from acorn.core.core_utils import str_to_class
import click
from acorn.core.pruning_utils import LinkedMaskLayerNorm

@click.command()
# Add an optional click argument to specify the checkpoint to use
@click.option("--checkpoint", "-c", default=None, help="Checkpoint to use for training")
@click.option("--align", "-a", default=False, type=bool, help="Align pruned sizes in memory")
def main(checkpoint, align):

    if not checkpoint:
        print("Checkpoint file required!")
        return
    
    config = torch.load(checkpoint, map_location=torch.device("cpu"), weights_only=False)["hyper_parameters"]
    
    stage = config["stage"]
    model = config["model"]

    stage_module_class = str_to_class(stage, model)
    stage_module = stage_module_class.load_from_checkpoint(
        checkpoint_path=checkpoint,
        hparams=config
    )

    print("Number of parameters before", count_parameters(stage_module))
    for name, module in stage_module.named_modules():
        if isinstance(module, nn.Sequential):
            for subname, submodule in module.named_children():
                full_name = f"{name}.{subname}"
                if isinstance(submodule, nn.Linear) and full_name[-1] != "6" and full_name not in ["edge_output_transform.3", "node_network.7.0", "node_network.7.3"]:
                    module = prune_linear(module, int(subname))

    print("Number of parameters after", count_parameters(stage_module))                    

    optimizer, scheduler = stage_module.configure_optimizers()

    new_ckpt = torch.load(checkpoint, weights_only=False)
    filtered_state = filter_optimizer_state(new_ckpt["optimizer_states"][0], stage_module.state_dict()) # configure optimizers before overwriting state dict
    new_ckpt["optimizer_states"] = [filtered_state]

    new_ckpt["state_dict"] = stage_module.state_dict()

    new_ckpt["hyper_parameters"]["mlp_sizes"] = get_hidden_layers(stage_module)

    oldName = checkpoint.split("/")[-1].split(".ckpt")
    name = oldName[0] + "_dense.ckpt"
    torch.save(new_ckpt, name)



def get_hidden_layers(stage_module):
    mlp_sizes={}

    for name, module in stage_module.named_modules():
        if isinstance(module, nn.Sequential):
            mlp_sizes[name] = []
            for subname, submodule in module.named_children():
                if isinstance(submodule, nn.Linear):
                    mlp_sizes[name].append(submodule.weight.size()[0])
                
    return mlp_sizes

def prune_linear_out_features(linear: nn.Linear, keep_idx: torch.Tensor):
    """Return a new Linear layer with selected output features kept."""
    in_features = linear.in_features
    out_features = len(keep_idx)

    new_linear = nn.Linear(in_features, out_features, bias=linear.bias is not None)

    with torch.no_grad():
        new_linear.weight.copy_(linear.weight[keep_idx, :])
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias[keep_idx])

    return new_linear


def prune_linear_in_features(linear: nn.Linear, keep_idx: torch.Tensor):
    """Return a new Linear layer with selected input features kept."""
    in_features = len(keep_idx)
    out_features = linear.out_features

    new_linear = nn.Linear(in_features, out_features, bias=linear.bias is not None)

    with torch.no_grad():
        new_linear.weight.copy_(linear.weight[:, keep_idx])
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias)

    return new_linear


def prune_layernorm(ln, keep_idx):
    if isinstance(ln, LinkedMaskLayerNorm):
        # Just a placeholder for loading the model
        return LinkedMaskLayerNorm(None, "", ln)
    else:
        return nn.LayerNorm(len(keep_idx), eps=ln.eps, elementwise_affine=ln.elementwise_affine)


def prune_linear(model, idx):
    """
    Remove pruned output features in the first Linear layer and fix downstream layers.
    Structure: linear (removing output features) - layer norm (creating new norm with adjusted size)
        - activation (no adjustment needed) - linear (removing input features)
    """

    # Find pruned rows
    keep_idx = torch.where(model[idx].weight.abs().sum(dim=1) != 0)[0]

    # Resize first linear layer = remove the output features
    size_before = list(model[idx].weight.size())
    model[idx] = prune_linear_out_features(model[idx], keep_idx)
    print(f"pruning linear layer {idx} {size_before} -> {list(model[idx].weight.size())}")

    # Resize following LayerNorm
    model[idx+1] = prune_layernorm(model[idx+1], keep_idx)

    # Resize next linear input features
    size_before = list(model[idx+3].weight.size())
    model[idx+3] = prune_linear_in_features(model[idx+3], keep_idx)
    print(f"pruning linear layer {idx+3} {size_before} -> {list(model[idx+3].weight.size())}")

    return model


def filter_optimizer_state(old_state, state_dict):
    new_state = {"state": {}, "param_groups": old_state["param_groups"]}

    # map id to name to identify optimizer for model element
    id_to_name = {}
    current_id = 0
    for group in old_state['param_groups']:
        for p_id in group['params']:
            name = list(state_dict.keys())[current_id]
            id_to_name[p_id] = name
            current_id += 1

    for p_id, states in old_state['state'].items():
        name = id_to_name[p_id]
        weight = state_dict[name]

        # pruning output
        keep_indices = torch.where(weight.abs().sum(dim=tuple(range(1, weight.ndim))) > 0)[0]

        new_states = {}
        for key, tensor in states.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                new_states[key] = tensor[keep_indices]
            else:
                new_states[key] = tensor
        
        keep_indices = torch.where(weight.abs().sum(dim=tuple(range(1, weight.ndim))) > 0)[0]

        new_state[p_id] = new_states

    return new_state


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    main()