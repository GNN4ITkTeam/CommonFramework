import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add

try:
    from frnn import frnn
except ImportError:
    frnn = None
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from typing import Optional

from acorn.utils.ml_utils import make_mlp


def find_neighbors(embedding1, embedding2, r_max=1.0, k_max=10):
    embedding1 = embedding1.detach()[None]
    embedding2 = embedding2.detach()[None]
    if frnn is None:
        raise ImportError("FRNN is required by HGNN!")
    _, idxs, _, _ = frnn.frnn_grid_points(
        points1=embedding1,
        points2=embedding2,
        K=k_max,
        r=r_max,
    )
    return idxs.squeeze(0)


def checkpointing(func, enabled=True):
    def checkpointed_fx(*x):
        if any(y.requires_grad for y in x) and enabled:
            return checkpoint(func, *x, use_reentrant=False)
        else:
            return func(*x)

    return checkpointed_fx


class InteractionGNNCell(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        hidden_activation: Optional[str] = "GELU",
        output_activation: Optional[str] = None,
        dropout: Optional[float] = 0.0,
        checkpoint: Optional[bool] = True,
    ):
        super().__init__()

        # The node network computes new node features
        self.node_network = make_mlp(
            d_model * 2,
            [d_hidden, d_model],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            input_dropout=dropout,
            hidden_dropout=dropout,
        )

        # The edge network computes new edge features
        self.edge_network = make_mlp(
            d_model * 3,
            [d_hidden, d_model],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            input_dropout=dropout,
            hidden_dropout=dropout,
        )

        self.node_norm = nn.BatchNorm1d(d_model * 2, track_running_stats=False)
        self.edge_norm = nn.BatchNorm1d(d_model * 3, track_running_stats=False)

        self.node_update = checkpointing(self.node_update, enabled=checkpoint)
        self.edge_update = checkpointing(self.edge_update, enabled=checkpoint)

    def node_update(self, nodes, edges, graph):
        """
        Calculate node update with checkpointing
        """
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        node_input = torch.cat([nodes, edge_messages], dim=-1)
        node_input = self.node_norm(node_input)
        nodes = self.node_network(node_input) + nodes  # Skip connection

        return nodes

    def edge_update(self, nodes, edges, graph):
        """
        Calculate edge update with checkpointing
        """
        edge_input = torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)
        edge_input = self.edge_norm(edge_input)
        edges = self.edge_network(edge_input) + edges

        return edges

    def forward(self, nodes, edges, graph):
        nodes = self.node_update(nodes, edges, graph)
        edges = self.edge_update(nodes, edges, graph)

        return nodes, edges


class HierarchicalGNNCell(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_edge_layers: Optional[int] = 2,
        hidden_activation: Optional[str] = "GELU",
        output_activation: Optional[str] = None,
        dropout: Optional[float] = 0.0,
        checkpoint: Optional[bool] = True,
    ):
        super().__init__()

        self.node_network = make_mlp(
            d_model * 3,
            [d_hidden, d_model],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            input_dropout=dropout,
            hidden_dropout=dropout,
        )

        self.edge_network = make_mlp(
            d_model * 3,
            [d_hidden, d_model],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            input_dropout=dropout,
            hidden_dropout=dropout,
        )

        self.snode_network = make_mlp(
            d_model * 3,
            [d_hidden, d_model],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            input_dropout=dropout,
            hidden_dropout=dropout,
        )

        self.sedge_network = make_mlp(
            d_model * 3,
            [d_hidden, d_model],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            input_dropout=dropout,
            hidden_dropout=dropout,
        )

        self.node_norm = nn.BatchNorm1d(d_model * 3, track_running_stats=False)
        self.edge_norm = nn.BatchNorm1d(d_model * 3, track_running_stats=False)
        self.snode_norm = nn.BatchNorm1d(d_model * 3, track_running_stats=False)
        self.sedge_norm = nn.BatchNorm1d(d_model * 3, track_running_stats=False)

        self.node_update = checkpointing(self.node_update, enabled=checkpoint)
        self.edge_update = checkpointing(self.edge_update, enabled=checkpoint)
        self.snode_update = checkpointing(self.snode_update, enabled=checkpoint)
        self.sedge_update = checkpointing(self.sedge_update, enabled=checkpoint)

    def node_update(self, nodes, edges, snodes, graph, bgraph, bweights):
        """
        Calculate node updates with checkpointing
        """
        snode_messages = scatter_add(
            bweights * F.normalize(snodes, p=2)[bgraph[1]],
            bgraph[0],
            dim=0,
            dim_size=nodes.shape[0],
        )
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        node_inputs = torch.cat([nodes, edge_messages, snode_messages], dim=-1)
        node_inputs = self.node_norm(node_inputs)
        nodes = self.node_network(node_inputs) + nodes
        return nodes

    def edge_update(self, nodes, edges, graph):
        """
        Calculate edge updates with checkpointing
        """
        edge_inputs = torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)
        edge_inputs = self.edge_norm(edge_inputs)
        edges = self.edge_network(edge_inputs) + edges
        return edges

    def snode_update(self, nodes, snodes, sedges, bgraph, bweights, sgraph, sweights):
        """
        Calculate supernode updates with checkpointing
        """
        node_messages = scatter_add(
            bweights * F.normalize(nodes, p=2)[bgraph[0]],
            bgraph[1],
            dim=0,
            dim_size=snodes.shape[0],
        )
        sedge_messages = scatter_add(
            F.normalize(sedges, p=2) * sweights,
            sgraph[1],
            dim=0,
            dim_size=snodes.shape[0],
        )
        snodes_input = torch.cat([snodes, sedge_messages, node_messages], dim=-1)
        snodes_input = self.snode_norm(snodes_input)
        snodes = self.snode_network(snodes_input) + snodes
        return snodes

    def sedge_update(self, snodes, sedges, sgraph, sweights):
        """
        Calculate superedge updates with checkpointing
        """
        sedges_input = torch.cat([snodes[sgraph[0]], snodes[sgraph[1]], sedges], dim=-1)
        sedges_input = self.sedge_norm(sedges_input)
        sedges = self.sedge_network(sedges_input) + sedges
        return sedges

    def forward(
        self, nodes, edges, snodes, sedges, graph, bgraph, bweights, sgraph, sweights
    ):
        """
        Whereas the message passing in the original/super graphs is implemented by interaction network, the one in between them (bipartite message
        passing is descirbed by weighted graph convolution (vanilla aggregation without attention)
        """

        # Compute new node features
        snodes = self.snode_update(
            nodes, snodes, sedges, bgraph, bweights, sgraph, sweights
        )
        nodes = self.node_update(nodes, edges, snodes, graph, bgraph, bweights)

        # Compute new edge features
        sedges = self.sedge_update(snodes, sedges, sgraph, sweights)
        edges = self.edge_update(nodes, edges, graph)

        return nodes, edges, snodes, sedges


class DynamicGraphConstruction(nn.Module):
    def __init__(
        self,
        k: int,
        weighting_function: Optional[str] = "exp",
        symmetrize: Optional[bool] = False,
        normalize: Optional[bool] = True,
        return_logits: Optional[bool] = False,
    ):
        """
        weighting function is used to turn dot products into weights
        """
        super().__init__()

        self.k = k
        self.symmetrize = symmetrize
        self.normalize = normalize
        self.return_logits = return_logits
        self.gamma = nn.parameter.Parameter(data=torch.zeros(1), requires_grad=True)
        self.beta = nn.parameter.Parameter(data=torch.zeros(1), requires_grad=True)
        self.weighting_function = getattr(torch, weighting_function)
        self.register_buffer("knn_radius", torch.ones(1), persistent=True)

    def forward(self, src_embeddings, dst_embeddings, original_graph=None, offset=0):
        """
        src embeddings: source nodes' embeddings
        dst embeddings: destination nodes' embeddings
        """
        # Construct the Graph
        with torch.no_grad():
            graph_idxs = find_neighbors(
                src_embeddings, dst_embeddings, r_max=self.knn_radius, k_max=self.k
            )
            positive_idxs = graph_idxs >= 0
            ind = (
                torch.arange(graph_idxs.shape[0], device=self.knn_radius.device)
                .unsqueeze(1)
                .expand(graph_idxs.shape)
            )
            graph = torch.stack([ind[positive_idxs], graph_idxs[positive_idxs]], dim=0)
            if original_graph is not None:
                graph = torch.cat([graph, original_graph], dim=1).unique(dim=1)
            if self.symmetrize:
                graph = to_scipy_sparse_matrix(graph.cpu())
                graph, _ = from_scipy_sparse_matrix(graph + graph.T)
                graph = graph.to(self.knn_radius.device)
            if self.training:
                maximum_dist = (
                    (src_embeddings[graph[0]] - dst_embeddings[graph[1]])
                    .square()
                    .sum(-1)
                    .sqrt()
                    .max()
                )
                self.knn_radius = (
                    0.9 * self.knn_radius + 0.11 * maximum_dist
                )  # Keep track of the minimum radius needed to give right number of neighbors

        # Compute bipartite attention
        likelihood = -torch.log(
            (src_embeddings[graph[0]] - dst_embeddings[graph[1]])
            .square()
            .sum(-1)
            .clamp(min=1e-12)
        )

        # regularize to ensure variance of weights
        edge_weights_logits = self.gamma.exp() * (likelihood - offset) + self.beta
        edge_weights = self.weighting_function(edge_weights_logits)

        if self.normalize:
            edge_weights = edge_weights / (
                1e-12 + scatter_add(edge_weights, graph[1], dim=0)[graph[1]]
            )

        edge_weights = edge_weights.unsqueeze(1)

        if self.return_logits:
            return graph, edge_weights, edge_weights_logits

        return graph, edge_weights
