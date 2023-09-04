from torch_geometric.nn import MessagePassing
from gnn4itk_cf.utils import make_mlp
import torch


class InteractionConv(MessagePassing):
    def __init__(
        self,
        node_net_input_size,
        hidden=128,
        nb_node_layer=3,
        nb_edge_layer=3,
        hidden_activation="SiLU",
        output_activation="SiLU",
        layernorm=True,
        batchnorm=False,
        aggr="add",
        *,
        aggr_kwargs={},
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
        **kwargs
    ):
        super().__init__(
            aggr,
            aggr_kwargs=aggr_kwargs,
            flow=flow,
            node_dim=node_dim,
            decomposed_layers=decomposed_layers,
            **kwargs
        )

        self.edge_network = make_mlp(
            3 * hidden,
            [hidden] * nb_edge_layer,
            layer_norm=layernorm,
            batch_norm=batchnorm,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
        )
        self.node_network = make_mlp(
            node_net_input_size,
            [hidden] * nb_node_layer,
            layer_norm=layernorm,
            batch_norm=batchnorm,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
        )

    def message(self, e) -> torch.Tensor:
        return e

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        edge_index,
        x,
        ptr=None,
        dim_size=None,
    ) -> torch.Tensor:
        src_message = self.aggr_module(inputs, edge_index[1], dim_size=x.size(0))
        dst_message = self.aggr_module(inputs, edge_index[0], dim_size=x.size(0))
        out = torch.cat([src_message, dst_message], dim=1)
        return out

    def update(self, inputs: torch.Tensor, x) -> torch.Tensor:
        x_in = torch.cat([x, inputs], dim=1)
        out = self.node_network(x_in)
        return out

    def edge_update(self, edge_index, x, e) -> torch.Tensor:
        x_in = torch.cat([x[edge_index[0]], x[edge_index[1]], e], dim=1)
        out = self.edge_network(x_in)
        return out

    def forward(self, edge_index, x, e):
        x = self.propagate(edge_index=edge_index, x=x, e=e)
        e = self.edge_updater(edge_index=edge_index, x=x, e=e)
        return x, e


class InteractionConv2(InteractionConv):
    def __init__(
        self,
        node_net_input_size,
        edge_net_input_size,
        hidden=128,
        n_node_net_layers=3,
        n_edge_net_layers=3,
        hidden_activation="SiLU",
        output_activation="SiLU",
        layernorm=True,
        batchnorm=False,
        aggr="add",
        *,
        aggr_kwargs={},
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
        **kwargs
    ):
        super().__init__(
            node_net_input_size,
            hidden,
            n_node_net_layers,
            n_edge_net_layers,
            hidden_activation,
            output_activation,
            layernorm,
            batchnorm,
            aggr,
            aggr_kwargs=aggr_kwargs,
            flow=flow,
            node_dim=node_dim,
            decomposed_layers=decomposed_layers,
            **kwargs
        )

        self.node_network = make_mlp(
            input_size=node_net_input_size,
            sizes=[hidden] * n_node_net_layers,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
            batch_norm=batchnorm,
        )

        self.edge_network = make_mlp(
            input_size=edge_net_input_size,
            sizes=[hidden] * n_edge_net_layers,
            output_activation=output_activation,
            hidden_activation=hidden_activation,
            layer_norm=layernorm,
            batch_norm=batchnorm,
        )

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        edge_index,
        x,
        in_out_diff_agg,
        ptr=None,
        dim_size=None,
    ) -> torch.Tensor:
        src_message = self.aggr_module(inputs, edge_index[1], dim_size=x.size(0))
        dst_message = self.aggr_module(inputs, edge_index[0], dim_size=x.size(0))

        if in_out_diff_agg:
            return torch.cat([src_message, dst_message], dim=1)
        else:
            return src_message + dst_message

    def forward(self, edge_index, x, e, in_out_diff_agg=True):
        x = self.propagate(
            edge_index=edge_index, x=x, e=e, in_out_diff_agg=in_out_diff_agg
        )
        e = self.edge_updater(edge_index, x=x, e=e)
        return x, e
