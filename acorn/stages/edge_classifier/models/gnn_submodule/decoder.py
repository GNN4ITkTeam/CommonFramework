import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from acorn.utils import make_mlp

from functools import partial
from itertools import product, combinations_with_replacement
from .encoder import get_string_edge_type


class HeteroEdgeDecoder(torch.nn.Module):
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
        track_running_stats=False,
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
        decoders = {}
        if self.hparams["hetero_level"] < 4:
            decoder = make_mlp(
                3 * hidden,
                [hidden] * nb_edge_layer + [1],
                layer_norm=layernorm,
                batch_norm=batchnorm,
                output_activation=None,
                hidden_activation=hidden_activation,
                hidden_dropout=hidden_dropout,
                track_running_stats=track_running_stats,
            )
            for region0, region1 in combinations_with_replacement(self.region_ids, r=2):
                decoders[(region0["name"], "to", region1["name"])] = decoders[
                    (region1["name"], "to", region0["name"])
                ] = decoder
        else:
            for region0, region1 in combinations_with_replacement(self.region_ids, r=2):
                decoder = make_mlp(
                    3 * hidden,
                    [hidden] * nb_edge_layer + [1],
                    layer_norm=layernorm,
                    batch_norm=batchnorm,
                    output_activation=None,
                    hidden_activation=hidden_activation,
                    hidden_dropout=hidden_dropout,
                    track_running_stats=track_running_stats,
                )
                decoders[
                    get_string_edge_type(region0["name"], "to", region1["name"])
                ] = decoders[
                    get_string_edge_type(region1["name"], "to", region0["name"])
                ] = decoder
            if self.hparams.get("simplified_edge_conv"):
                for region0, region1 in product(region_ids, region_ids):
                    if region0["name"] == region1["name"]:
                        continue
                    decoders[
                        get_string_edge_type(region0["name"], "to", region1["name"])
                    ] = decoders[
                        get_string_edge_type(region0["name"], "to", region0["name"])
                    ]
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, x_dict, edge_index_dict, edge_dict, *args, **kwargs):
        output_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            # if no edge exists, create an empty tensor for output
            if edge_index.shape[1] == 0:
                output_dict[edge_type] = torch.empty(0).to(edge_index.device)
                continue
            src, _, dst = edge_type
            x_in = torch.cat(
                [
                    x_dict[src][edge_index[0]],
                    x_dict[dst][edge_index[1]],
                    edge_dict[edge_type],
                ],
                dim=-1,
            )
            network = self.decoders[get_string_edge_type(*edge_type)]
            if self.checkpoint:
                network = partial(checkpoint, network, use_reentrant=False)
            output_dict[edge_type] = network(x_in).squeeze(-1)
        return output_dict
