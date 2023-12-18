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

from functools import partial
import warnings
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add
from torch_geometric.nn import aggr

from acorn.utils import make_mlp
from ..edge_classifier_stage import EdgeClassifierStage
from .gnn_submodule.encoder import HeteroEdgeEncoder, HeteroNodeEncoder
from .gnn_submodule.updater import (
    HeteroNodeConv,
    HeteroEdgeConv,
    EdgeUpdater,
)
from .gnn_submodule.igcn import InteractionConv, InteractionConv2
from .gnn_submodule.decoder import HeteroEdgeDecoder
from itertools import product, combinations_with_replacement


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

        self.setup_aggregation()

        hparams["node_net_recurrent"] = (
            True
            if "node_net_recurrent" not in hparams
            else hparams["node_net_recurrent"]
        )
        hparams["edge_net_recurrent"] = (
            True
            if "edge_net_recurrent" not in hparams
            else hparams["edge_net_recurrent"]
        )
        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )
        hparams["track_running_stats"] = (
            False
            if "track_running_stats" not in hparams
            else hparams["track_running_stats"]
        )

        # Setup input network
        self.node_encoder = make_mlp(
            len(hparams["node_features"]),
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            track_running_stats=hparams["track_running_stats"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            track_running_stats=hparams["track_running_stats"],
        )

        # The edge network computes new edge features from connected nodes
        if not hparams["edge_net_recurrent"]:
            self.edge_networks = nn.ModuleList(
                [
                    make_mlp(
                        3 * hparams["hidden"],
                        [hparams["hidden"]] * hparams["nb_edge_layer"],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        output_activation=hparams["output_activation"],
                        hidden_activation=hparams["hidden_activation"],
                        track_running_stats=hparams["track_running_stats"],
                    )
                    for _ in range(hparams["n_graph_iters"])
                ]
            )
        else:
            self.edge_network = make_mlp(
                3 * hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_edge_layer"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                track_running_stats=hparams["track_running_stats"],
            )

        # The node network computes new node features
        if not hparams["node_net_recurrent"]:
            self.node_networks = nn.ModuleList(
                [
                    make_mlp(
                        self.network_input_size,
                        [hparams["hidden"]] * hparams["nb_node_layer"],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        output_activation=hparams["output_activation"],
                        hidden_activation=hparams["hidden_activation"],
                        track_running_stats=hparams["track_running_stats"],
                    )
                    for _ in range(hparams["n_graph_iters"])
                ]
            )
        else:
            self.node_network = make_mlp(
                self.network_input_size,
                [hparams["hidden"]] * hparams["nb_node_layer"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                track_running_stats=hparams["track_running_stats"],
            )

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
            track_running_stats=hparams["track_running_stats"],
        )

        self.save_hyperparameters(hparams)

    def message_step(self, x, start, end, e, i):
        # Compute new node features
        edge_messages = torch.cat(
            [
                self.aggregation(e, end, dim_size=x.shape[0]),
                self.aggregation(e, start, dim_size=x.shape[0]),
            ],
            dim=-1,
        )

        node_inputs = torch.cat([x, edge_messages], dim=-1)

        if not self.hparams["node_net_recurrent"]:
            x_out = self.node_networks[i](node_inputs)
        else:
            x_out = self.node_network(node_inputs)

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        if not self.hparams["edge_net_recurrent"]:
            e_out = self.edge_networks[i](edge_inputs)
        else:
            e_out = self.edge_network(edge_inputs)

        return x_out, e_out

    def output_step(self, x, start, end, e):
        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)
        scores = self.output_edge_classifier(classifier_inputs).squeeze(-1)

        if (
            self.hparams.get("undirected")
            and self.hparams.get("dataset_class") != "HeteroGraphDataset"
        ):
            scores = torch.mean(scores.view(2, -1), dim=0)

        return scores

    def forward(self, batch, **kwargs):
        x = torch.stack(
            [batch[feature] for feature in self.hparams["node_features"]], dim=-1
        ).float()
        start, end = batch.edge_index
        if "undirected" in self.hparams and self.hparams["undirected"]:
            start, end = torch.cat([start, end]), torch.cat([end, start])

        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder, x, use_reentrant=False)
        e = checkpoint(
            self.edge_encoder, torch.cat([x[start], x[end]], dim=1), use_reentrant=False
        )

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            x, e = checkpoint(
                self.message_step, x, start, end, e, i, use_reentrant=False
            )

        return self.output_step(x, start, end, e)

    def setup_aggregation(self):
        if "aggregation" not in self.hparams:
            self.hparams["aggregation"] = ["sum"]
            self.network_input_size = 3 * (self.hparams["hidden"])
        elif isinstance(self.hparams["aggregation"], str):
            self.hparams["aggregation"] = [self.hparams["aggregation"]]
            self.network_input_size = 3 * (self.hparams["hidden"])
        elif isinstance(self.hparams["aggregation"], list):
            self.network_input_size = (1 + 2 * len(self.hparams["aggregation"])) * (
                self.hparams["hidden"]
            )
        else:
            raise ValueError("Unknown aggregation type")

        try:
            self.aggregation = aggr.MultiAggregation(
                self.hparams["aggregation"], mode="cat"
            )
        except ValueError:
            raise ValueError(
                "Unknown aggregation type. Did you know that the latest version of"
                " GNN4ITk accepts any list of aggregations? E.g. [sum, mean], [max,"
                " min, std], etc."
            )


class InteractionGNNWithPyG(EdgeClassifierStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Define the dataset to be used, if not using the default

        self.setup_aggregation()

        hparams["node_net_recurrent"] = (
            True
            if "node_net_recurrent" not in hparams
            else hparams["node_net_recurrent"]
        )
        hparams["edge_net_recurrent"] = (
            True
            if "edge_net_recurrent" not in hparams
            else hparams["edge_net_recurrent"]
        )
        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )
        hparams["track_running_stats"] = (
            False
            if "track_running_stats" not in hparams
            else hparams["track_running_stats"]
        )

        # Setup input network
        self.node_encoder = make_mlp(
            len(hparams["node_features"]),
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            track_running_stats=hparams["track_running_stats"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            track_running_stats=hparams["track_running_stats"],
        )

        self.convs = nn.ModuleList([])
        conv = InteractionConv(
            self.network_input_size, aggr=self.hparams["aggregation"], **self.hparams
        )
        for i in range(self.hparams["n_graph_iters"]):
            self.convs.append(
                conv
                if self.hparams.get("recurrent")
                else InteractionConv(
                    self.network_input_size,
                    aggr=self.hparams["aggregation"],
                    **self.hparams
                )
            )

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation="Sigmoid",
            hidden_activation=hparams["hidden_activation"],
            track_running_stats=hparams["track_running_stats"],
        )

        self.save_hyperparameters(hparams)
        self.checkpoint = self.hparams.get("checkpoint")

    def forward(self, batch, **kwargs):
        x = torch.stack(
            [batch[feature] for feature in self.hparams["node_features"]], dim=-1
        ).to(self.dtype)
        edge_index = batch.edge_index

        # if undirected, extend the edge index to include the inverse graph
        if "undirected" in self.hparams and self.hparams["undirected"]:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        start, end = edge_index
        x = (
            checkpoint(self.node_encoder, x, use_reentrant=False)
            if self.checkpoint
            else self.node_encoder(x)
        )
        e = (
            checkpoint(
                self.edge_encoder,
                torch.cat([x[start], x[end]], dim=1),
                use_reentrant=False,
            )
            if self.checkpoint
            else self.edge_encoder(torch.cat([x[start], x[end]], dim=1))
        )

        for i in range(self.hparams["n_graph_iters"]):
            conv = (
                partial(checkpoint, self.convs[i], use_reentrant=False)
                if self.checkpoint
                else self.convs[i]
            )
            x, e = conv(edge_index, x, e)

        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)

        scores = self.output_edge_classifier(classifier_inputs).squeeze(-1)

        if (
            self.hparams.get("undirected")
            and self.hparams.get("dataset_class") != "HeteroGraphDataset"
        ):
            scores = torch.mean(scores.view(2, -1), dim=0)

        return scores

    def setup_aggregation(self):
        if "aggregation" not in self.hparams:
            self.hparams["aggregation"] = ["sum"]
            self.network_input_size = 3 * (self.hparams["hidden"])
        elif isinstance(self.hparams["aggregation"], str):
            self.hparams["aggregation"] = [self.hparams["aggregation"]]
            self.network_input_size = 3 * (self.hparams["hidden"])
        elif isinstance(self.hparams["aggregation"], list):
            self.network_input_size = (1 + 2 * len(self.hparams["aggregation"])) * (
                self.hparams["hidden"]
            )
        else:
            raise ValueError("Unknown aggregation type")


class InteractionGNN2(EdgeClassifierStage):
    """
    Interaction Network (L2IT version).
    Operates on directed graphs.
    Aggregate and reduce (sum) separately incomming and outcoming edges latents.
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_batch_norm"] = hparams.get("output_batch_norm", False)
        hparams["edge_output_transform_final_batch_norm"] = hparams.get(
            "edge_output_transform_final_batch_norm", False
        )
        hparams["edge_output_transform_final_batch_norm"] = hparams.get(
            "edge_output_transform_final_batch_norm", False
        )
        hparams["track_running_stats"] = (
            False
            if "track_running_stats" not in hparams
            else hparams["track_running_stats"]
        )

        # TODO: Add equivalent check and default values for other model parameters ?
        # TODO: Use get() method

        # Define the dataset to be used, if not using the default
        self.save_hyperparameters(hparams)

        # self.setup_layer_sizes()

        if hparams["concat"]:
            if hparams["in_out_diff_agg"]:
                in_node_net = hparams["hidden"] * 4
            else:
                in_node_net = hparams["hidden"] * 3
            in_edge_net = hparams["hidden"] * 6
        else:
            if hparams["in_out_diff_agg"]:
                in_node_net = hparams["hidden"] * 3
            else:
                in_node_net = hparams["hidden"] * 2
            in_edge_net = hparams["hidden"] * 3
        # node encoder
        self.node_encoder = make_mlp(
            input_size=len(hparams["node_features"]),
            sizes=[hparams["hidden"]] * hparams["n_node_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )
        # edge encoder
        if "edge_features" in hparams and len(hparams["edge_features"]) != 0:
            self.edge_encoder = make_mlp(
                input_size=len(hparams["edge_features"]),
                sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_batch_norm=hparams["output_batch_norm"],
                track_running_stats=hparams["track_running_stats"],
            )
        else:
            self.edge_encoder = make_mlp(
                input_size=2 * hparams["hidden"],
                sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_batch_norm=hparams["output_batch_norm"],
                track_running_stats=hparams["track_running_stats"],
            )

        # edge network
        if hparams["edge_net_recurrent"]:
            self.edge_network = make_mlp(
                input_size=in_edge_net,
                sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_batch_norm=hparams["output_batch_norm"],
                track_running_stats=hparams["track_running_stats"],
            )
        else:
            self.edge_network = nn.ModuleList(
                [
                    make_mlp(
                        input_size=in_edge_net,
                        sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
                        output_activation=hparams["output_activation"],
                        hidden_activation=hparams["hidden_activation"],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        output_batch_norm=hparams["output_batch_norm"],
                        track_running_stats=hparams["track_running_stats"],
                    )
                    for i in range(hparams["n_graph_iters"])
                ]
            )
        # node network
        if hparams["node_net_recurrent"]:
            self.node_network = make_mlp(
                input_size=in_node_net,
                sizes=[hparams["hidden"]] * hparams["n_node_net_layers"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_batch_norm=hparams["output_batch_norm"],
                track_running_stats=hparams["track_running_stats"],
            )
        else:
            self.node_network = nn.ModuleList(
                [
                    make_mlp(
                        input_size=in_node_net,
                        sizes=[hparams["hidden"]] * hparams["n_node_net_layers"],
                        output_activation=hparams["output_activation"],
                        hidden_activation=hparams["hidden_activation"],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        output_batch_norm=hparams["output_batch_norm"],
                        track_running_stats=hparams["track_running_stats"],
                    )
                    for i in range(hparams["n_graph_iters"])
                ]
            )

        # edge decoder
        self.edge_decoder = make_mlp(
            input_size=hparams["hidden"],
            sizes=[hparams["hidden"]] * hparams["n_edge_decoder_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )
        # edge output transform layer
        self.edge_output_transform = make_mlp(
            input_size=hparams["hidden"],
            sizes=[hparams["hidden"], 1],
            output_activation=hparams["edge_output_transform_final_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["edge_output_transform_final_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )

        # dropout layer
        self.dropout = nn.Dropout(p=0.1)
        # hyperparams
        # self.hparams = hparams

    def forward(self, batch):
        x = torch.stack(
            [batch[feature] for feature in self.hparams["node_features"]], dim=-1
        ).float()

        # Same features on the 3 channels in the STRIP ENDCAP TODO: Process it in previous stage
        mask = torch.logical_or(batch.region == 2, batch.region == 6).reshape(-1)
        x[mask] = torch.cat([x[mask, 0:4], x[mask, 0:4], x[mask, 0:4]], dim=1)
        # print(x[:, 8:12])

        if "edge_features" in self.hparams and len(self.hparams) != 0:
            edge_attr = torch.stack(
                [batch[feature] for feature in self.hparams["edge_features"]], dim=-1
            ).float()
        else:
            edge_attr = None

        x.requires_grad = True
        if edge_attr is not None:
            edge_attr.requires_grad = True

        # Get src and dst
        src, dst = batch.edge_index

        # Encode nodes and edges features into latent spaces
        if self.hparams["checkpointing"]:
            x = checkpoint(self.node_encoder, x)
            if edge_attr is not None:
                e = checkpoint(self.edge_encoder, edge_attr)
            else:
                e = checkpoint(self.edge_encoder, torch.cat([x[src], x[dst]], dim=-1))
        else:
            x = self.node_encoder(x)
            if edge_attr is not None:
                e = self.edge_encoder(edge_attr)
            else:
                e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))
        # Apply dropout
        # x = self.dropout(x)
        # e = self.dropout(e)

        # memorize initial encodings for concatenate in the gnn loop if request
        if self.hparams["concat"]:
            input_x = x
            input_e = e
        # Initialize outputs
        outputs = []
        # Loop over gnn layers
        for i in range(self.hparams["n_graph_iters"]):
            if self.hparams["checkpointing"]:
                if self.hparams["concat"]:
                    x = checkpoint(self.concat, x, input_x)
                    e = checkpoint(self.concat, e, input_e)
                if (
                    self.hparams["node_net_recurrent"]
                    and self.hparams["edge_net_recurrent"]
                ):
                    x, e, out = checkpoint(self.message_step, x, e, src, dst)
                else:
                    x, e, out = checkpoint(self.message_step, x, e, src, dst, i)
            else:
                if self.hparams["concat"]:
                    x = torch.cat([x, input_x], dim=-1)
                    e = torch.cat([e, input_e], dim=-1)
                if (
                    self.hparams["node_net_recurrent"]
                    and self.hparams["edge_net_recurrent"]
                ):
                    x, e, out = self.message_step(x, e, src, dst)
                else:
                    x, e, out = self.message_step(x, e, src, dst, i)
            outputs.append(out)
        return outputs[-1].squeeze(-1)

    def message_step(self, x, e, src, dst, i=None):
        edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)  # order dst src x ?
        if self.hparams["edge_net_recurrent"]:
            e_updated = self.edge_network(edge_inputs)
        else:
            e_updated = self.edge_network[i](edge_inputs)
        # Update nodes
        edge_messages_from_src = scatter_add(e_updated, dst, dim=0, dim_size=x.shape[0])
        edge_messages_from_dst = scatter_add(e_updated, src, dim=0, dim_size=x.shape[0])
        if self.hparams["in_out_diff_agg"]:
            node_inputs = torch.cat(
                [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
            )  # to check : the order dst src  x ?
        else:
            # add message from src and dst ?? # edge_messages = edge_messages_from_src + edge_messages_from_dst
            edge_messages = edge_messages_from_src + edge_messages_from_dst
            node_inputs = torch.cat([edge_messages, x], dim=-1)
        # x_updated = self.dropout(self.node_network[i](node_inputs))
        if self.hparams["node_net_recurrent"]:
            x_updated = self.node_network(node_inputs)
        else:
            x_updated = self.node_network[i](node_inputs)

        return (
            x_updated,
            e_updated,
            self.edge_output_transform(self.edge_decoder(e_updated)),
        )

    def concat(self, x, y):
        return torch.cat([x, y], dim=-1)


class InteractionGNN2WithPyG(InteractionGNN2):
    def __init__(self, hparams):
        super().__init__(hparams)
        if hparams["concat"]:
            if hparams["in_out_diff_agg"]:
                in_node_net = hparams["hidden"] * 6
            else:
                in_node_net = hparams["hidden"] * 4
            in_edge_net = hparams["hidden"] * 4
        else:
            if hparams["in_out_diff_agg"]:
                in_node_net = hparams["hidden"] * 3
            else:
                in_node_net = hparams["hidden"] * 2
            in_edge_net = hparams["hidden"] * 3
        self.convs = nn.ModuleList([])
        conv = InteractionConv2(in_node_net, in_edge_net, **self.hparams)
        for _ in range(self.hparams["n_graph_iters"]):
            self.convs.append(
                conv
                if self.hparams.get("node_net_recurrent")
                or self.hparams.get("edge_net_recurrent")
                else InteractionConv2(in_node_net, in_edge_net, **self.hparams)
            )
        self.checkpoint = self.hparams.get("checkpoint", False) or self.hparams.get(
            "checkpointing", False
        )

    def forward(self, batch):
        x = torch.stack(
            [batch[feature] for feature in self.hparams["node_features"]], dim=-1
        )

        # Get src and dst
        src, dst = batch.edge_index

        # Encode nodes and edges features into latent spaces
        node_encoder = (
            partial(checkpoint, self.node_encoder, use_reentrant=False)
            if self.checkpoint
            else self.node_encoder
        )

        edge_encoder = (
            partial(checkpoint, self.edge_encoder, use_reentrant=False)
            if self.checkpoint
            else self.edge_encoder
        )

        x = node_encoder(x.to(self.dtype))

        e = (
            torch.stack(
                [batch[feature] for feature in self.hparams["edge_features"]], dim=-1
            ).to(self.dtype)
            if len(self.hparams.get("edge_features", [])) > 0
            else torch.cat([x[src], x[dst]], dim=-1)
        )

        e = edge_encoder(e)

        # memorize initial encodings for concatenate in the gnn loop if request
        input_x = x.clone()
        input_e = e.clone()
        # Initialize outputs
        outputs = []
        # Loop over gnn layers
        for i in range(self.hparams["n_graph_iters"]):
            conv = (
                partial(checkpoint, self.convs[i], use_reentrant=False)
                if self.checkpoint
                else self.convs[i]
            )
            if self.hparams["concat"]:
                x = torch.cat([x, input_x], dim=1)
                e = torch.cat([e, input_e], dim=1)
            x, e = conv(
                edge_index=batch.edge_index,
                x=x,
                e=e,
                in_out_diff_agg=self.hparams.get("in_out_diff_agg"),
            )

            outputs.append(self.edge_output_transform(self.edge_decoder(e)))
        return outputs[-1].squeeze(-1)


class HeteroMixin:
    """
    Mixin methods specifically for Heterogeneous GNN. These include initiation methods to create heterogeneous modules in the network.
    """

    def __init__(self, hparams):
        self.hparams = hparams

    def make_coding_module(self, module, params={}):
        hparams = self.hparams.copy()
        for key, val in params.items():
            hparams[key] = val
        return module(**hparams)

    def make_single_conv(self, msg_passing_class, conv_class, hparams):
        convs = {}
        for region0, region1 in combinations_with_replacement(
            hparams["region_ids"], r=2
        ):
            conv = msg_passing_class(**hparams)
            convs[region0["name"], "to", region1["name"]] = convs[
                (region1["name"], "to", region0["name"])
            ] = conv
        if self.hparams.get("simplified_edge_conv"):
            for region0, region1 in product(
                hparams["region_ids"], hparams["region_ids"]
            ):
                if region0["name"] == region1["name"]:
                    continue
                convs[(region0["name"], "to", region1["name"])] = convs[
                    (region0["name"], "to", region0["name"])
                ]
        return conv_class(convs, aggr=hparams.get("modulewise_aggregation", "sum"))

    def make_updater(self, msg_passing_class, conv_class, params={}):
        hparams = self.hparams.copy()
        for key, val in params.items():
            hparams[key] = val

        module_convs = torch.nn.ModuleList([])
        if hparams.get("recurrent"):
            conv = self.make_single_conv(msg_passing_class, conv_class, hparams)
            for _ in range(hparams["n_graph_iters"]):
                module_convs.append(conv)
        else:
            for _ in range(hparams["n_graph_iters"]):
                conv = self.make_single_conv(msg_passing_class, conv_class, hparams)
                module_convs.append(conv)

        return module_convs


class HeteroInteractionGNN(InteractionGNN, HeteroMixin):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.setup_aggregation()
        node_updater = eval(self.hparams.get("node_updater", "NodeUpdater"))
        self.node_encoder = self.make_coding_module(HeteroNodeEncoder)
        self.edge_encoder = self.make_coding_module(HeteroEdgeEncoder)
        self.node_networks = self.make_updater(node_updater, HeteroNodeConv)
        self.edge_networks = self.make_updater(EdgeUpdater, HeteroEdgeConv)
        self.output_edge_classifier = self.make_coding_module(HeteroEdgeDecoder)

    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """
        preprocess = True
        input_dir = "input_dir"
        if stage in ["fit", "predict"]:
            self.load_data(stage, self.hparams[input_dir], preprocess)
            # self.test_data(stage)
        elif stage == "test":
            if not self.hparams.get("reprocess_classifier"):
                print("Reading data from stage_dir without preprocessing")
                input_dir = "stage_dir"
                preprocess = False
            self.load_data(stage, self.hparams[input_dir], preprocess)

        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss", summary="min")
            self.logger.experiment.define_metric("auc", summary="max")
        except Exception:
            warnings.warn(
                "Failed to define figures of merit, due to logger unavailable"
            )

    def forward(self, batch, **kwargs):
        x_dict = batch.input_node_features_dict
        edge_index_dict = batch.edge_index_dict
        edge_dict = batch.collect("input_edge_features")

        x_input_dict = self.node_encoder(x_dict)
        x_dict = x_input_dict.copy()

        edge_input_dict = self.edge_encoder(
            x_dict, edge_index_dict, edge_dict=edge_dict
        )
        edge_dict = edge_input_dict.copy()

        for node_updater, edge_updater in zip(self.node_networks, self.edge_networks):
            if self.hparams.get("concat_node"):
                x_dict = {
                    name: torch.cat([x_dict[name], x_input], dim=-1)
                    for name, x_input in x_input_dict.items()
                }
            x_dict = node_updater(x_dict, edge_index_dict, edge_dict)
            if self.hparams.get("concat_edge"):
                edge_dict = {
                    name: torch.cat([edge_dict[name], e], dim=-1)
                    for name, e in edge_input_dict.items()
                }
            edge_dict = edge_updater(x_dict, edge_index_dict, edge_dict)
        return self.output_edge_classifier(x_dict, edge_index_dict, edge_dict), x_dict

    def training_step(self, batch, batch_idx):
        eval_dict = self.shared_evaluation(batch, batch_idx)
        loss, pos_loss, neg_loss = (
            eval_dict["loss"],
            eval_dict["pos_loss"],
            eval_dict["neg_loss"],
        )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "train_pos_loss",
            pos_loss,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "train_neg_loss",
            neg_loss,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

        return loss

    def shared_evaluation(self, batch, batch_idx):
        output_dict, _ = self(batch)
        for k, v in output_dict.items():
            batch[k]["output"] = v
        batch = batch.to_homogeneous()

        output = batch.output
        loss, pos_loss, neg_loss = self.loss_function(output, batch)

        all_truth = batch.y.bool()
        target_truth = (batch.weights > 0) & all_truth

        return {
            "loss": loss,
            "all_truth": all_truth,
            "target_truth": target_truth,
            "output": output.detach(),
            "batch": batch,
            "pos_loss": pos_loss,
            "neg_loss": neg_loss,
        }
