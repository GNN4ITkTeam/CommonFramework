import warnings
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_geometric.nn import aggr, to_hetero

from gnn4itk_cf.utils import make_mlp
from ...edge_classifier_stage import EdgeClassifierStage

from functools import partial
from itertools import product, combinations_with_replacement

def get_string_edge_type(*edge_type):
    return '__'.join(edge_type)

class HeteroNodeEncoder(torch.nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams

        self.encoders = torch.nn.ModuleDict()

        input_dim = len(hparams["node_features"])

        encoder = make_mlp(
            input_dim,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            input_dropout=hparams.get('input_dropout', 0.),
            hidden_dropout=hparams.get('hidden_dropout', 0.)
        )
        for region_id in self.hparams['region_ids']:
            name = region_id['name']
            if self.hparams['hetero_level'] < 1:
                self.encoders[name] = encoder
            else:
                self.encoders[name] = make_mlp(
                    input_dim,
                    [hparams["hidden"]] * hparams["nb_node_layer"],
                    output_activation=hparams["output_activation"],
                    hidden_activation=hparams["hidden_activation"],
                    layer_norm=hparams["layernorm"],
                    batch_norm=hparams["batchnorm"],
                    input_dropout=hparams.get('input_dropout', 0.),
                    hidden_dropout=hparams.get('hidden_dropout', 0.)
                )
        
    def forward(self, x_dict):     
        for node_type, x_in in x_dict.items():
            network = partial(checkpoint, self.encoders[node_type], use_reentrant=False) if self.hparams.get('checkpoint') else self.encoders[node_type]
            x_dict[node_type] = network(x_in.float())
        return x_dict

class HeteroEdgeEncoder(torch.nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams
        region_ids = self.hparams['region_ids']
        encoders = {}
        input_dim = 2*self.hparams['hidden'] + + len(self.hparams.get('edge_features', []))
        if self.hparams['hetero_level'] < 2:
            encoder = make_mlp(
                input_dim,
                [hparams["hidden"]] * hparams["nb_edge_layer"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                hidden_dropout=hparams.get('hidden_dropout', 0.)
            )
            for region0, region1 in combinations_with_replacement(region_ids, r=2):
                encoders[(region0['name'], 'to', region1['name']) ] = encoders[(region1['name'], 'to', region0['name'])] = encoder
        else:
            for region0, region1 in combinations_with_replacement(region_ids, r=2):
                encoder = make_mlp(
                    input_dim,
                    [hparams["hidden"]] * hparams["nb_edge_layer"],
                    layer_norm=hparams["layernorm"],
                    batch_norm=hparams["batchnorm"],
                    output_activation=hparams["output_activation"],
                    hidden_activation=hparams["hidden_activation"],
                    hidden_dropout=hparams.get('hidden_dropout', 0)
                )
                encoders[ get_string_edge_type(region0['name'], 'to', region1['name']) ] = encoders[ get_string_edge_type(region1['name'], 'to', region0['name']) ] = encoder
            if self.hparams.get('simplified_edge_conv'):
                for region0, region1 in product(region_ids, region_ids):
                    if region0['name'] == region1['name']: continue
                    encoders[get_string_edge_type(region0['name'], 'to', region1['name'])] = encoders[get_string_edge_type(region0['name'], 'to', region0['name'])]
        self.encoders=nn.ModuleDict(encoders)

    def forward(self, x_dict, edge_index_dict, edge_dict={}, *args, **kwargs):
        edge_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.shape[1]==0: continue
            src, _, dst = edge_type
            x_in = torch.cat([
                x_dict[src][edge_index[0]],
                x_dict[dst][edge_index[1]]
            ], dim=-1)
            if edge_type in edge_dict:
                x_in = torch.cat([
                    x_in,
                    edge_dict[edge_type]
                ], dim=-1)
            network = self.encoders[get_string_edge_type(*edge_type)]
            if self.hparams['checkpoint']:
                network = partial(checkpoint, self.encoders[ get_string_edge_type(*edge_type) ],  use_reentrant=False)
            edge_dict[edge_type] = network(x_in)
        return edge_dict