import warnings
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_geometric.nn import aggr, to_hetero

from gnn4itk_cf.utils import make_mlp

from functools import partial
from itertools import product, combinations_with_replacement
from .encoder import get_string_edge_type

class HeteroEdgeDecoder(torch.nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams
        region_ids = self.hparams['region_ids']
        decoders = {}
        if self.hparams['hetero_level'] < 4:
            decoder = make_mlp(
                3 * hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=None,
                hidden_activation=hparams["hidden_activation"],
                hidden_dropout=hparams.get('hidden_dropout', 0)
            )
            for region0, region1 in combinations_with_replacement(region_ids, r=2):
                decoders[(region0['name'], 'to', region1['name']) ] = decoders[(region1['name'], 'to', region0['name'])] = decoder
        else:
            for region0, region1 in combinations_with_replacement(region_ids, r=2):
                decoder = make_mlp(
                    3 * hparams["hidden"],
                    [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
                    layer_norm=hparams["layernorm"],
                    batch_norm=hparams["batchnorm"],
                    output_activation=None,
                    hidden_activation=hparams["hidden_activation"],
                    hidden_dropout=hparams.get('hidden_dropout', 0)
                )
                decoders[ get_string_edge_type(region0['name'], 'to', region1['name']) ] = decoders[ get_string_edge_type(region1['name'], 'to', region0['name']) ] = decoder
            if self.hparams.get('simplified_edge_conv'):
                for region0, region1 in product(region_ids, region_ids):
                    if region0['name'] == region1['name']: continue
                    decoders[get_string_edge_type(region0['name'], 'to', region1['name'])] = decoders[get_string_edge_type(region0['name'], 'to', region0['name'])]
        self.decoders=nn.ModuleDict(decoders)

    def forward(self, x_dict, edge_index_dict, edge_dict, *args, **kwargs):
        output_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            # if no edge exists, create an empty tensor for output
            if edge_index.shape[1]==0: 
                output_dict[edge_type]=torch.empty(0).to(edge_index.device)
                continue
            src, _, dst = edge_type
            x_in = torch.cat([
                x_dict[src][edge_index[0]],
                x_dict[dst][edge_index[1]],
                edge_dict[edge_type]
            ], dim=-1)
            network = self.decoders[get_string_edge_type(*edge_type)]
            if self.hparams.get('checkpoint'):
                network = partial(checkpoint, network,  use_reentrant=False) 
            output_dict[edge_type] = network(x_in).squeeze(-1)
        return output_dict

class HeteroNodeDecoder(torch.nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams
        region_ids = self.hparams['region_ids']
        decoders = {}
        if self.hparams['hetero_level'] < 4:
            decoder = make_mlp(
                hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=None,
                hidden_activation=hparams["hidden_activation"],
                hidden_dropout=hparams.get('hidden_dropout', 0)
            )
            for region in region_ids:
                decoders[ region['name'] ] = decoder
        else:
            for region in region_ids:
                decoder = make_mlp(
                    hparams["hidden"],
                    [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
                    layer_norm=hparams["layernorm"],
                    batch_norm=hparams["batchnorm"],
                    output_activation=None,
                    hidden_activation=hparams["hidden_activation"],
                    hidden_dropout=hparams.get('hidden_dropout', 0)
                )
                decoders[ region['name'] ] = decoder
        self.decoders=nn.ModuleDict(decoders)

    def forward(self, x_dict):     
        for node_type, x_in in x_dict.items():
            network = partial(checkpoint, self.decoders[node_type], use_reentrant=False) if self.hparams.get('checkpoint') else self.decoders[node_type]
            x_dict[node_type] = network(x_in.float())
        return x_dict