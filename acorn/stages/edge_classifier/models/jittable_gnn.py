from typing import Dict, Optional, Tuple
import importlib

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add
from torch_geometric.nn import SAGEConv

from acorn.utils import make_mlp

from ..edge_classifier_stage import EdgeClassifierStage


class InteractionGNNParams:
    """handel GNN paramerters"""

    params: Dict[str, bool]

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
        hparams["track_running_stats"] = hparams.get("track_running_stats", False)

        # define boolean variables used in the forward method.
        # For non-booleans, treat them as attributes of the class
        # because TorchScript requires the dictionary has consistent types.
        self.params = {}
        self.params["checkpointing"] = hparams.get("checkpointing", False)
        self.params["concat"] = hparams.get("concat", True)
        self.n_graph_iters: int = hparams.get("n_graph_iters", 8)
        self.in_out_diff_agg: bool = hparams.get("in_out_diff_agg", True)
        self.ckpting_reentrant: bool = hparams.get("ckpting_reentrant", False)

        if hparams["concat"]:
            if hparams["in_out_diff_agg"]:
                self.in_node_net = hparams["hidden"] * 4
            else:
                self.in_node_net = hparams["hidden"] * 3
            self.in_edge_net = hparams["hidden"] * 6
        else:
            if hparams["in_out_diff_agg"]:
                self.in_node_net = hparams["hidden"] * 3
            else:
                self.in_node_net = hparams["hidden"] * 2
            self.in_edge_net = hparams["hidden"] * 3
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
        if hparams.get("IGNN2", True):
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
            self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        if self.params["checkpointing"] and self.training:
            return self.forward_with_checkpoint(node_features, edge_index, edge_attr)
        else:
            return self.forward_without_checkpoint(node_features, edge_index, edge_attr)


class RecurrentInteractionGNN(InteractionGNNParams, EdgeClassifierStage):
    def __init__(self, hparams):
        hparams["concat"] = hparams.get("concat", False)
        hparams["n_node_net_layers"] = hparams["nb_node_layer"]
        hparams["n_edge_net_layers"] = hparams["nb_edge_layer"]
        hparams["edge_output_transform_final_activation"] = None
        hparams["in_out_diff_agg"] = hparams.get("in_out_diff_agg", True)
        hparams["IGNN2"] = False
        super().__init__(hparams)

        # edge network
        self.edge_network = make_mlp(
            input_size=self.in_edge_net,
            sizes=[hparams["hidden"]] * hparams["nb_edge_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )

        # node network
        self.node_network = make_mlp(
            input_size=self.in_node_net,
            sizes=[hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )

        # edge decoder
        self.output_edge_classifier = make_mlp(
            input_size=3 * hparams["hidden"],
            sizes=[hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            output_activation=hparams["edge_output_transform_final_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )

    @torch.jit.export
    def forward_without_checkpoint(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        src, dst = edge_index[0], edge_index[1]
        x = self.node_encoder(x)
        e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))
        outputs = []
        for i in range(self.n_graph_iters):
            x, e, out = self.recurrent_message_step(x, e, src, dst)
        outputs.append(torch.sigmoid(out))
        return outputs

    def recurrent_message_step(
        self, x: torch.Tensor, e: torch.Tensor, start: torch.Tensor, end: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute new node features
        dst_index = torch.unsqueeze(end, 0).tile(e.shape[1], 1)
        edge_messages_from_src = torch.scatter_add(
            torch.zeros(e.shape[1], x.shape[0], device=dst_index.device),
            1,
            dst_index,
            e.T,
        ).T

        src_index = torch.unsqueeze(start, 0).tile(e.shape[1], 1)
        edge_messages_from_dst = torch.scatter_add(
            torch.zeros(e.shape[1], x.shape[0], device=src_index.device),
            1,
            src_index,
            e.T,
        ).T

        node_inputs = torch.cat(
            [x, edge_messages_from_src, edge_messages_from_dst], dim=-1
        )

        x_out = self.node_network(node_inputs)
        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)
        classifier_inputs = torch.cat([x_out[start], x_out[end], e_out], dim=1)
        scores = self.output_edge_classifier(classifier_inputs)
        return x_out, e_out, scores


class RecurrentInteractionGNN2(InteractionGNNParams, EdgeClassifierStage):
    """
    Interaction Network (L2IT version).
    Operates on directed graphs.
    Aggregate and reduce (sum) separately incomming and outcoming edges latents.
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        # edge network
        self.edge_network = make_mlp(
            input_size=self.in_edge_net,
            sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )

        # node network
        self.node_network = make_mlp(
            input_size=self.in_node_net,
            sizes=[hparams["hidden"]] * hparams["n_node_net_layers"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_batch_norm=hparams["output_batch_norm"],
            track_running_stats=hparams["track_running_stats"],
        )

    @torch.jit.export
    def forward_without_checkpoint(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        src, dst = edge_index[0], edge_index[1]
        x = self.node_encoder(x)
        if edge_attr is not None:
            e = self.edge_encoder(edge_attr)
        else:
            e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))

        input_x = x
        input_e = e
        outputs = []
        for i in range(self.n_graph_iters):
            if self.params["concat"]:
                x = torch.cat([x, input_x], dim=-1)
                e = torch.cat([e, input_e], dim=-1)
            x, e, out = self.recurrent_message_step(x, e, src, dst)
            outputs.append(out)
        return outputs[-1].squeeze(-1)

    @torch.jit.unused
    def forward_with_checkpoint(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        src, dst = edge_index
        x.requires_grad = True

        if edge_attr is not None:
            edge_attr.requires_grad = True

        # Encode nodes and edges features into latent spaces
        x = checkpoint(self.node_encoder, x, use_reentrant=self.ckpting_reentrant)
        if edge_attr is not None:
            e = checkpoint(
                self.edge_encoder, edge_attr, use_reentrant=self.ckpting_reentrant
            )
        else:
            e = checkpoint(
                self.edge_encoder,
                torch.cat([x[src], x[dst]], dim=-1),
                use_reentrant=self.ckpting_reentrant,
            )

        # memorize initial encodings for concatenate in the gnn loop if request
        if self.hparams["concat"]:
            input_x = x
            input_e = e
        # Initialize outputs
        outputs = []
        # Loop over gnn layers
        for _ in range(self.n_graph_iters):
            if self.hparams["concat"]:
                x = checkpoint(
                    self.concat, x, input_x, use_reentrant=self.ckpting_reentrant
                )
                e = checkpoint(
                    self.concat, e, input_e, use_reentrant=self.ckpting_reentrant
                )

            x, e, out = checkpoint(
                self.recurrent_message_step,
                x,
                e,
                src,
                dst,
                use_reentrant=self.ckpting_reentrant,
            )
            outputs.append(out)

        return outputs[-1].squeeze(-1)

    def recurrent_message_step(
        self, x: torch.Tensor, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)
        e_updated = self.edge_network(edge_inputs)

        dst_index = torch.unsqueeze(dst, 0).tile(e_updated.shape[1], 1)
        edge_messages_from_src = torch.scatter_add(
            torch.zeros(e_updated.shape[1], x.shape[0], device=dst_index.device),
            1,
            dst_index,
            e_updated.T,
        ).T

        src_index = torch.unsqueeze(src, 0).tile(e_updated.shape[1], 1)
        edge_messages_from_dst = torch.scatter_add(
            torch.zeros(e_updated.shape[1], x.shape[0], device=src_index.device),
            1,
            src_index,
            e_updated.T,
        ).T

        if self.in_out_diff_agg:
            node_inputs = torch.cat(
                [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
            )  # to check : the order dst src  x ?
        else:
            # add message from src and dst ?? # edge_messages = edge_messages_from_src + edge_messages_from_dst
            edge_messages = edge_messages_from_src + edge_messages_from_dst
            node_inputs = torch.cat([edge_messages, x], dim=-1)

        x_updated = self.node_network(node_inputs)

        return (
            x_updated,
            e_updated,
            self.edge_output_transform(self.edge_decoder(e_updated)),
        )

    def concat(self, x, y):
        return torch.cat([x, y], dim=-1)


class ChainedInteractionGNN2(InteractionGNNParams, EdgeClassifierStage):
    """
    Interaction Network (L2IT version).
    Operates on directed graphs.
    Aggregate and reduce (sum) separately incomming and outcoming edges latents.
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        # edge network
        self.edge_network = nn.ModuleList(
            [
                make_mlp(
                    input_size=self.in_edge_net,
                    sizes=[hparams["hidden"]] * hparams["n_edge_net_layers"],
                    output_activation=hparams["output_activation"],
                    hidden_activation=hparams["hidden_activation"],
                    layer_norm=hparams["layernorm"],
                    batch_norm=hparams["batchnorm"],
                    output_batch_norm=hparams["output_batch_norm"],
                    track_running_stats=hparams["track_running_stats"],
                )
                for _ in range(hparams["n_graph_iters"])
            ]
        )

        # node network
        self.node_network = nn.ModuleList(
            [
                make_mlp(
                    input_size=self.in_node_net,
                    sizes=[hparams["hidden"]] * hparams["n_node_net_layers"],
                    output_activation=hparams["output_activation"],
                    hidden_activation=hparams["hidden_activation"],
                    layer_norm=hparams["layernorm"],
                    batch_norm=hparams["batchnorm"],
                    output_batch_norm=hparams["output_batch_norm"],
                    track_running_stats=hparams["track_running_stats"],
                )
                for _ in range(hparams["n_graph_iters"])
            ]
        )

    @torch.jit.export
    def forward_without_checkpoint(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        src, dst = edge_index[0], edge_index[1]
        x = self.node_encoder(x)
        if edge_attr is not None:
            e = self.edge_encoder(edge_attr)
        else:
            e = self.edge_encoder(torch.cat([x[src], x[dst]], dim=-1))

        input_x = x
        input_e = e
        outputs = []

        for node_net, edge_net in zip(self.node_network, self.edge_network):
            if self.params["concat"]:
                x = torch.cat([x, input_x], dim=-1)
                e = torch.cat([e, input_e], dim=-1)
            edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)
            e_updated = edge_net(edge_inputs)

            dst_index = torch.unsqueeze(dst, 0).tile(e_updated.shape[1], 1)
            edge_messages_from_src = torch.scatter_add(
                torch.zeros(e_updated.shape[1], x.shape[0], device=dst_index.device),
                1,
                dst_index,
                e_updated.T,
            ).T

            src_index = torch.unsqueeze(src, 0).tile(e_updated.shape[1], 1)
            edge_messages_from_dst = torch.scatter_add(
                torch.zeros(e_updated.shape[1], x.shape[0], device=src_index.device),
                1,
                src_index,
                e_updated.T,
            ).T

            if self.in_out_diff_agg:
                node_inputs = torch.cat(
                    [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
                )
            else:
                edge_messages = edge_messages_from_src + edge_messages_from_dst
                node_inputs = torch.cat([edge_messages, x], dim=-1)
            x_updated = node_net(node_inputs)
            outputs.append(self.edge_output_transform(self.edge_decoder(e_updated)))
            x = x_updated
            e = e_updated

        return outputs[-1].squeeze(-1)

    @torch.jit.unused
    def forward_with_checkpoint(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        src, dst = edge_index
        x.requires_grad = True

        if edge_attr is not None:
            edge_attr.requires_grad = True

        # Encode nodes and edges features into latent spaces
        x = checkpoint(self.node_encoder, x, use_reentrant=self.ckpting_reentrant)
        if edge_attr is not None:
            e = checkpoint(
                self.edge_encoder, edge_attr, use_reentrant=self.ckpting_reentrant
            )
        else:
            e = checkpoint(
                self.edge_encoder,
                torch.cat([x[src], x[dst]], dim=-1),
                use_reentrant=self.ckpting_reentrant,
            )

        # memorize initial encodings for concatenate in the gnn loop if request
        if self.hparams["concat"]:
            input_x = x
            input_e = e
        # Initialize outputs
        outputs = []
        # Loop over gnn layers
        for i in range(self.n_graph_iters):
            if self.hparams["concat"]:
                x = checkpoint(
                    self.concat, x, input_x, use_reentrant=self.ckpting_reentrant
                )
                e = checkpoint(
                    self.concat, e, input_e, use_reentrant=self.ckpting_reentrant
                )

            x, e, out = checkpoint(
                self.message_step,
                x,
                e,
                src,
                dst,
                i,
                use_reentrant=self.ckpting_reentrant,
            )
            outputs.append(out)

        return outputs[-1].squeeze(-1)

    def message_step(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        i: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_inputs = torch.cat([e, x[src], x[dst]], dim=-1)  # order dst src x ?
        # Torch.jit does not like the use of edge_network[i]. Error info from the torch.jit is:
        # Expected integer literal for index but got a variable or non-integer.
        # ModuleList/Sequential indexing is only supported with integer literals.
        # For example, 'i = 4; self.layers[i](x)' will fail because i is not a literal.
        # Enumeration is supported, e.g. 'for index, v in enumerate(self): out = v(inp)':
        # So this function is only used in training mode.
        e_updated = self.edge_network[i](edge_inputs)

        # Update nodes
        edge_messages_from_src = scatter_add(e_updated, dst, dim=0, dim_size=x.shape[0])
        edge_messages_from_dst = scatter_add(e_updated, src, dim=0, dim_size=x.shape[0])
        if self.in_out_diff_agg:
            node_inputs = torch.cat(
                [edge_messages_from_src, edge_messages_from_dst, x], dim=-1
            )  # to check : the order dst src  x ?
        else:
            # add message from src and dst ?? # edge_messages = edge_messages_from_src + edge_messages_from_dst
            edge_messages = edge_messages_from_src + edge_messages_from_dst
            node_inputs = torch.cat([edge_messages, x], dim=-1)

        x_updated = self.node_network[i](node_inputs)

        return (
            x_updated,
            e_updated,
            self.edge_output_transform(self.edge_decoder(e_updated)),
        )

    def concat(self, x, y):
        return torch.cat([x, y], dim=-1)


class GCNEncoderJitable(nn.Module):
    def __init__(self, gnn_config) -> None:
        super().__init__()
        self.gnn_config = gnn_config
        self.layers = nn.ModuleList()

        for conf in gnn_config:
            module = importlib.import_module(conf["module_name"])
            layer = getattr(module, conf["class_name"])(**conf["init_kwargs"])
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, edge_list: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_list)
            else:
                x = layer(x)
        return x


class GNNFilterJitable(EdgeClassifierStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["track_running_stats"] = hparams.get("track_running_stats", False)

        self.net = make_mlp(
            hparams["hidden"] * 2,
            [hparams["hidden"] // (2**i) for i in range(hparams["nb_layer"])] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
            track_running_stats=hparams["track_running_stats"],
        )
        self.gnn = GCNEncoderJitable(hparams["gnn_config"])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.gnn(x, edge_index)
        output = self.net(torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1))
        return output.squeeze(-1)
