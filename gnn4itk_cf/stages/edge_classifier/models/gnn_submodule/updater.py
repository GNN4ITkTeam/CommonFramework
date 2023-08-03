from collections import defaultdict
import torch
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_geometric.nn import aggr, HeteroConv, MessagePassing
from torch_geometric.nn.conv.hgt_conv import group

from gnn4itk_cf.utils import make_mlp


def get_aggregation(aggregation):
    """
    Factory dictionary for aggregation depending on the hparams["aggregation"]
    """

    aggregation_dict = {
        "sum": lambda e, end, x: scatter_add(e, end, dim=0, dim_size=x.shape[0]),
        "mean": lambda e, end, x: scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
        "max": lambda e, end, x: scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
        "sum_max": lambda e, end, x: torch.cat(
            [
                scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                scatter_add(e, end, dim=0, dim_size=x.shape[0]),
            ],
            dim=-1,
        ),
        "mean_sum": lambda e, end, x: torch.cat(
            [
                scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
                scatter_add(e, end, dim=0, dim_size=x.shape[0]),
            ],
            dim=-1,
        ),
        "mean_max": lambda e, end, x: torch.cat(
            [
                scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
            ],
            dim=-1,
        ),
    }

    return aggregation_dict[aggregation]


class EdgeUpdater(torch.nn.Module):
    def __init__(
        self,
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
        concat_edge=True,
        **hparams,
    ):
        super().__init__()
        self.hparams = hparams
        concatenation_factor = 3
        self.checkpoint = checkpoint
        if self.hparams.get("concat_edge"):
            concatenation_factor += 1

        # The edge network computes new edge features from connected nodes
        self.network = make_mlp(
            concatenation_factor * hidden,
            [hidden] * nb_edge_layer,
            layer_norm=layernorm,
            batch_norm=batchnorm,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
            hidden_dropout=hidden_dropout,
        )

    def forward(self, x, edge_index, edge, *args, **kwargs):
        src, dst = edge_index
        if isinstance(x, tuple):
            x1, x2 = x
            x_input = torch.cat([x1[src], x2[dst], edge], dim=-1)
        else:
            x_input = torch.cat([x[src], x[dst], edge], dim=-1)
        edge_out = (
            checkpoint(self.network, x_input, use_reentrant=False)
            if self.checkpoint
            else self.network(x_input)
        )
        if self.hparams.get("concat_edge"):
            return edge_out
        else:
            return edge + edge_out


class HeteroEdgeConv(HeteroConv):
    def __init__(self, convs: dict, aggr: str = "sum"):
        super().__init__(convs, aggr)

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        edge_dict=None,
        *args_dict,
        **kwargs_dict,
    ) -> dict:
        out_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.shape[1] == 0:
                continue
            src, rel, dst = edge_type
            str_edge_type = "__".join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append((value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None), value_dict.get(dst, None))

            conv = self.convs[str_edge_type]

            edge = None
            if isinstance(edge_dict, dict):
                edge = edge_dict.get(edge_type)

            if src == dst:
                out = (
                    conv(x_dict[src], edge_index, edge)
                    if edge is not None
                    else conv(x_dict[src], edge_index)
                )
            else:
                out = (
                    conv((x_dict[src], x_dict[dst]), edge_index, edge)
                    if edge is not None
                    else conv((x_dict[src], x_dict[dst]), edge_index)
                )

            out_dict[edge_type] = out

        return out_dict


class NodeUpdater(MessagePassing):
    def __init__(
        self,
        aggr: str = "add",
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
        **hparams,
    ):
        super().__init__(
            aggr, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers
        )

        self.hparams = hparams

        # self.aggr_module = get_aggregation(self.hparams['aggregation'])
        self.setup_aggregation()

        input_size = self.network_input_size

        if self.hparams.get("concat_node"):
            input_size += self.hparams["hidden"]

        # The node network computes new node features
        self.node_network = make_mlp(
            input_size,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            hidden_dropout=hparams.get("hidden_dropout", 0),
        )

    def message(self, edge, src_edge):
        return src_edge, edge

    def aggregate(self, out, x, edge_index, src_edge_index):
        src, dst = edge_index
        if src_edge_index is not None:
            src, _ = src_edge_index
        src_edge, dst_edge = out

        src_message = self.aggregation(src_edge, src, dim_size=x.shape[0])
        dst_message = self.aggregation(dst_edge, dst, dim_size=x.shape[0])

        return torch.cat([src_message, dst_message], dim=-1)

    def update(self, agg_message, x):
        x_in = torch.cat([x, agg_message], dim=-1).float()
        x_out = (
            checkpoint(self.node_network, x_in, use_reentrant=False)
            if self.hparams.get("checkpoint")
            else self.node_network(x_in)
        )
        if self.hparams.get("concat_node"):
            return x_out
        else:
            return x + x_out

    def forward(self, x, edge_index, edge, *args, **kwargs):
        src_edge, src_edge_index = kwargs.get("src_edge"), kwargs.get("src_edge_index")
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src, x_dst = x, x

        x_dst = self.propagate(
            edge_index,
            x=x_dst,
            edge=edge,
            src_edge=src_edge,
            src_edge_index=src_edge_index,
        )

        return x_dst

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


class DirectedNodeUpdater(NodeUpdater):
    def setup_aggregation(self):
        if "aggregation" not in self.hparams:
            self.hparams["aggregation"] = ["sum"]
            self.network_input_size = 2 * (self.hparams["hidden"])
        elif isinstance(self.hparams["aggregation"], str):
            self.hparams["aggregation"] = [self.hparams["aggregation"]]
            self.network_input_size = 2 * (self.hparams["hidden"])
        elif isinstance(self.hparams["aggregation"], list):
            self.network_input_size = (1 + len(self.hparams["aggregation"])) * (
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

    def aggregate(self, out, x, edge_index):
        src, dst = edge_index
        _, edge = out

        dst_message = self.aggregation(edge, dst, dim_size=x.shape[0])

        return dst_message


class HeteroNodeConv(HeteroConv):
    def __init__(self, convs, aggr="sum"):
        super().__init__(convs, aggr)

    def forward(
        self,
        x_dict,
        edge_index_dict,
        edge_dict,
        *args_dict,
        **kwargs_dict,
    ):
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
            *args_dict (optional): Additional forward arguments of invididual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For example, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.shape[1] == 0:
                continue
            src, rel, dst = edge_type
            str_edge_type = "__".join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append((value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None), value_dict.get(dst, None))

            conv = self.convs[str_edge_type]
            x_in = (x_dict[src], x_dict[dst])
            src_edge_index = edge_index_dict.get((dst, rel, src))
            edge = edge_dict[edge_type]
            src_edge = edge_dict.get((dst, rel, src))

            out = conv(
                x_in,
                edge_index,
                edge,
                src_edge_index=src_edge_index,
                src_edge=src_edge,
                *args,
                **kwargs,
            )

            out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict
