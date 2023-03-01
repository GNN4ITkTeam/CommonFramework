# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add, scatter_mean, scatter_max

from gnn4itk_cf.utils import make_mlp
from ..edge_classifier_stage import EdgeClassifierStage


class InteractionGNN(EdgeClassifierStage):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Define the dataset to be used, if not using the default
        self.save_hyperparameters(hparams)

        self.setup_layer_sizes()  

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )

        # Setup input network
        self.node_encoder = make_mlp(
            len(hparams["node_features"]),
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            self.concatenation_factor * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def aggregation_step(self, num_nodes):

        aggregation_dict = {
            "sum_mean_max": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
                    scatter_mean(x, y, dim=0, dim_size=num_nodes),
                    scatter_add(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "sum_max": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
                    scatter_add(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "mean_max": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
                    scatter_mean(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "mean_sum": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_mean(x, y, dim=0, dim_size=num_nodes),
                    scatter_add(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "sum": lambda x, y, **kwargs: scatter_add(x, y, dim=0, dim_size=num_nodes),
            "mean": lambda x, y, **kwargs: scatter_mean(x, y, dim=0, dim_size=num_nodes),
            "max": lambda x, y, **kwargs: scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
        }

        return aggregation_dict[self.hparams["aggregation"]]

    def message_step(self, x, start, end, e):

        # Compute new node features
        edge_messages = torch.cat(
            [
                self.aggregation_step(x.shape[0])(e, end),
                self.aggregation_step(x.shape[0])(e, start)
            ],
            dim=-1,
        )

        node_inputs = torch.cat([x, edge_messages], dim=-1)

        x_out = self.node_network(node_inputs)

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)

        return x_out, e_out

    def output_step(self, x, start, end, e):

        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)
        classifier_output = self.output_edge_classifier(classifier_inputs).squeeze(-1)

        if "undirected" in self.hparams and self.hparams["undirected"]: # Take mean of outgoing edges and incoming edges
            classifier_output = (classifier_output[:classifier_output.shape[0] // 2] + classifier_output[classifier_output.shape[0] // 2:]) / 2

        return 

    def forward(self, batch, **kwargs):

        x = torch.stack([batch[feature] for feature in self.hparams["node_features"]], dim=-1).float()
        start, end = batch.edge_index
        if "undirected" in self.hparams and self.hparams["undirected"]:
            start, end = torch.cat([start, end]), torch.cat([end, start])

        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder, x)
        e = checkpoint(self.edge_encoder, torch.cat([x[start], x[end]], dim=1))

        # Loop over iterations of edge and node networks
        for _ in range(self.hparams["n_graph_iters"]):
            x, e = checkpoint(self.message_step, x, start, end, e)

        return self.output_step(x, start, end, e)

    def setup_layer_sizes(self):

        if self.hparams["aggregation"] == "sum_mean_max":
            self.concatenation_factor = 7
        elif self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"]:
            self.concatenation_factor = 5
        elif self.hparams["aggregation"] in ["sum", "mean", "max"]:
            self.concatenation_factor = 3
        else:
            raise ValueError("Aggregation type not recognised")

class InteractionGNN2(EdgeClassifierStage):
    """
    Message Passing Neural Network
    """
    def __init__(self, hparams):
        super().__init__(hparams)


                
                
        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )


        # Define the dataset to be used, if not using the default
        self.save_hyperparameters(hparams)

        #self.setup_layer_sizes()


        if hparams["concat"] == True:
            if hparams["in_out_diff_agg"]:
                in_node_net = hparams["hidden"]*4
            else:
                in_node_net = hparams["hidden"]*3
            in_edge_net = hparams["hidden"]*6
        else:
            if hparams["in_out_diff_agg"]:
                in_node_net = hparams["hidden"]*3
            else:
                in_node_net = hparams["hidden"]*2
            in_edge_net = hparams["hidden"]*3
        # node encoder
        self.node_encoder = make_mlp(
            input_size=len(hparams["node_features"]),
            sizes=[hparams["hidden"]]*hparams["n_node_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],)
        # edge encoder
        if "edge_features" in hparams and len(hparams["edge_features"]) != 0:
            self.edge_encoder = make_mlp(
                input_size=len(hparams["edge_features"]),
                sizes=[hparams["hidden"]]*hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],)
        else:
            self.edge_encoder = make_mlp(
                input_size=2*hparams["hidden"],
                sizes=[hparams["hidden"]]*hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],)

        # edge network
        if hparams["edge_net_recurrent"]:
            self.edge_network = make_mlp(
                input_size=in_edge_net,
                sizes=[hparams["hidden"]]*hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],)
        else:
            self.edge_network = nn.ModuleList(
            [make_mlp(
            input_size=in_edge_net,
            sizes=[hparams["hidden"]]*hparams["n_edge_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],)
            for i in range(hparams['n_graph_iters'])])
                # node network
        if hparams["node_net_recurrent"]:
            self.node_network = make_mlp(
                input_size=in_node_net,
                sizes=[hparams["hidden"]]*hparams["n_node_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],)
        else:
            self.node_network = nn.ModuleList(
            [make_mlp(
            input_size=in_node_net,
            sizes=[hparams["hidden"]]*hparams["n_node_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],)
            for i in range(hparams['n_graph_iters'])])
        
        # edge decoder
        self.edge_decoder = make_mlp(
            input_size=hparams["hidden"],
            sizes=[hparams["hidden"]]*hparams["n_edge_decoder_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],)
        # edge output transform layer
        self.edge_output_transform = make_mlp(
            input_size=hparams["hidden"],
            sizes=[hparams["hidden"], 1],
            output_activation=hparams["edge_output_transform_final_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],)

        # dropout layer
        self.dropout = nn.Dropout(p=0.1)
        # hyperparams
        #self.hparams = hparams

    def forward(self, batch):

        x = torch.stack([batch[feature] for feature in self.hparams["node_features"]], dim=-1).float()
        if "edge_features" in self.hparams and len(self.hparams)!=0:
            edge_attr = torch.stack([batch[feature] for feature in self.hparams["edge_features"]], dim=-1).float()
        else:
            edge_attr = None

        x.requires_grad = True
        if edge_attr!= None:
            edge_attr.requires_grad = True

        # Get src and dst
        src, dst = batch.edge_index
        
        # Encode nodes and edges features into latent spaces
        if self.hparams["checkpointing"]:
            x = checkpoint(self.node_encoder, x)
            if edge_attr!=None:
                e = checkpoint(self.edge_encoder, edge_attr) 
            else:
                e = checkpoint(self.edge_encoder, torch.cat([x[src], x[dst]], dim=-1))
        else:
            x = self.node_encoder(x)
            if edge_attr!=None:
                e = self.edge_encoder(edge_attr)
            else:
                e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))
        # Apply dropout
        #x = self.dropout(x)
        #e = self.dropout(e)

        # memorize initial encodings for concatenate in the gnn loop if request
        if self.hparams["concat"] == True:
            input_x = x
            input_e = e
        # Initialize outputs
        outputs = []
        # Loop over gnn layers
        for i in range(self.hparams["n_graph_iters"]):
            if self.hparams["checkpointing"]:
                if self.hparams["concat"] == True:
                    x = checkpoint(self.concat, x, input_x)
                    e = checkpoint(self.concat, e, input_e)
                if self.hparams["node_net_recurrent"] and self.hparams["edge_net_recurrent"]:
                    x, e, out = checkpoint(self.message_step, x, e, src, dst)
                else:
                    x, e, out = checkpoint(self.message_step, x, e, src, dst, i)
            else:
                if self.hparams["concat"] == True:
                    x = torch.cat([x, input_x], dim=-1)
                    e = torch.cat([e, input_e], dim=-1)
                if self.hparams["node_net_recurrent"] and self.hparams["edge_net_recurrent"]:
                    x, e, out = self.message_step(x, e, src, dst)
                else:
                    x, e, out = self.message_step(x, e, src, dst, i)
            outputs.append(out)
        return outputs[-1].squeeze(-1)

    def message_step(self, x, e, src, dst, i=None):
        edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1) # order dst src x ?
        if self.hparams["edge_net_recurrent"]:
            e_updated = self.edge_network(edge_inputs)
        else:
            e_updated = self.edge_network[i](edge_inputs)
        # Update nodes        
        edge_messages_from_src = scatter_add(e_updated, dst, dim=0, dim_size=x.shape[0])
        edge_messages_from_dst = scatter_add(e_updated, src, dim=0, dim_size=x.shape[0])
        if self.hparams["in_out_diff_agg"]:
            node_inputs = torch.cat([edge_messages_from_src, edge_messages_from_dst, x], dim=-1) # to check : the order dst src  x ?
        else: 
            # add message from src and dst ?? # edge_messages = edge_messages_from_src + edge_messages_from_dst
            edge_messages = edge_messages_from_src + edge_messages_from_dst
            node_inputs = torch.cat([edge_messages, x], dim=-1)
        #x_updated = self.dropout(self.node_network[i](node_inputs))
        if self.hparams["node_net_recurrent"]:
            x_updated = self.node_network(node_inputs)
        else:
            x_updated = self.node_network[i](node_inputs)

        return x_updated, e_updated, self.edge_output_transform(self.edge_decoder(e_updated))


    def concat(self, x, y):
        return torch.cat([x, y], dim=-1) 
