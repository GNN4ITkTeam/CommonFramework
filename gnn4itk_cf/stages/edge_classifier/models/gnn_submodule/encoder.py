import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint  # as do_checkpoint
from gnn4itk_cf.utils import make_mlp
from functools import partial
from itertools import product, combinations_with_replacement


def get_string_edge_type(*edge_type):
    return "__".join(edge_type)


class HeteroNodeEncoder(torch.nn.Module):
    """Heterogeneous encoder module. Create heterogeneous node encoders as a torch.nn.ModuleDict instance from a list of region_ids:
    [
        {'id': LIST_OF_REGION_ID_1, 'name': REGION_NAME_1},
        {'id': LIST_OF_REGION_ID_2, 'namE': REGION_NAME_2},
        ...
    ]
    as

    {
        REGION_NAME_1: MODULE_1
        REGION_NAME_2: MODULE_2,
        ...
    }
    Args:
        hparams: A dictionary of hyperparameters.
    """

    def __init__(
        self,
        node_features=["r", "phi", "z"],
        hidden=64,
        nb_node_layer=4,
        hidden_activation="ReLU",
        output_activation=None,
        layernorm=False,
        batchnorm=False,
        input_dropout=0,
        hidden_dropout=0,
        region_ids=[],
        hetero_level=4,
        checkpoint=True,
        **hparams
    ) -> None:
        super().__init__()

        self.hparams = hparams

        self.encoders = torch.nn.ModuleDict()

        self.do_checkpoint = checkpoint

        input_dim = len(node_features)

        encoder = make_mlp(
            input_dim,
            [hidden] * nb_node_layer,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
            batch_norm=batchnorm,
            input_dropout=input_dropout,
            hidden_dropout=hidden_dropout,
        )
        for region_id in region_ids:
            name = region_id["name"]
            if hetero_level < 1:
                self.encoders[name] = encoder
            else:
                self.encoders[name] = make_mlp(
                    input_dim,
                    [hidden] * nb_node_layer,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    layer_norm=layernorm,
                    batch_norm=batchnorm,
                    input_dropout=input_dropout,
                    hidden_dropout=hidden_dropout,
                )

    def forward(self, x_dict: dict):
        for node_type, x_in in x_dict.items():
            network = (
                partial(checkpoint, self.encoders[node_type], use_reentrant=False)
                if self.do_checkpoint
                else self.encoders[node_type]
            )
            x_dict[node_type] = network(x_in.float())
        return x_dict


class HeteroEdgeEncoder(torch.nn.Module):
    """Heterogeneous encoder module. Create heterogeneous node encoders as a torch.nn.ModuleDict instance from a list of region_ids:
    [
        {'id': LIST_OF_REGION_ID_1, 'name': REGION_NAME_1},
        {'id': LIST_OF_REGION_ID_2, 'name': REGION_NAME_2},
        ...
    ]
    as

    {
        REGION_NAME_1__to__REGION_NAME_1: MODULE_1__1
        REGION_NAME_1__to__REGION_NAME_2: MODULE_1__2,
        ...
    }
    Args:
        hparams: A dictionary of hyperparameters.
    """

    def __init__(
        self,
        edge_features=[],
        hidden=64,
        nb_edge_layer=4,
        hidden_activation="ReLU",
        output_activation=None,
        layernorm=False,
        batchnorm=False,
        hidden_dropout=0,
        region_ids=[],
        hetero_level=4,
        checkpoint=True,
        **hparams
    ) -> None:
        super().__init__()
        self.hparams = hparams
        self.region_ids = region_ids
        self.checkpoint = checkpoint
        encoders = {}
        input_dim = 2 * hidden + len(edge_features)
        if hetero_level < 2:
            encoder = make_mlp(
                input_dim,
                [hidden] * nb_edge_layer,
                layer_norm=layernorm,
                batch_norm=batchnorm,
                output_activation=output_activation,
                hidden_activation=hidden_activation,
                hidden_dropout=hidden_dropout,
            )
            for region0, region1 in combinations_with_replacement(self.region_ids, r=2):
                encoders[(region0["name"], "to", region1["name"])] = encoders[
                    (region1["name"], "to", region0["name"])
                ] = encoder
        else:
            for region0, region1 in combinations_with_replacement(self.region_ids, r=2):
                encoder = make_mlp(
                    input_dim,
                    [hidden] * nb_edge_layer,
                    layer_norm=layernorm,
                    batch_norm=batchnorm,
                    output_activation=output_activation,
                    hidden_activation=hidden_activation,
                    hidden_dropout=hidden_dropout,
                )
                encoders[
                    get_string_edge_type(region0["name"], "to", region1["name"])
                ] = encoders[
                    get_string_edge_type(region1["name"], "to", region0["name"])
                ] = encoder
            if self.hparams.get("simplified_edge_conv"):
                for region0, region1 in product(self.region_ids, self.region_ids):
                    if region0["name"] == region1["name"]:
                        continue
                    encoders[
                        get_string_edge_type(region0["name"], "to", region1["name"])
                    ] = encoders[
                        get_string_edge_type(region0["name"], "to", region0["name"])
                    ]
        self.encoders = nn.ModuleDict(encoders)

    def forward(
        self, x_dict: dict, edge_index_dict: dict, edge_dict={}, *args, **kwargs
    ):
        edge_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.shape[1] == 0:
                continue
            src, _, dst = edge_type
            x_in = torch.cat(
                [x_dict[src][edge_index[0]], x_dict[dst][edge_index[1]]], dim=-1
            )
            if edge_type in edge_dict:
                x_in = torch.cat([x_in, edge_dict[edge_type]], dim=-1)
            network = self.encoders[get_string_edge_type(*edge_type)]
            if self.checkpoint:
                network = partial(
                    checkpoint,
                    self.encoders[get_string_edge_type(*edge_type)],
                    use_reentrant=False,
                )
            edge_dict[edge_type] = network(x_in)
        return edge_dict
