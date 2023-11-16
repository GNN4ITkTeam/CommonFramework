import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from typing import Optional

from .gnn_cells import InteractionGNNCell, HierarchicalGNNCell, DynamicGraphConstruction
from gnn4itk_cf.utils.ml_utils import make_mlp

class InteractionGNNBlock(nn.Module):

    """
    An interaction network for embedding class
    """

    def __init__(
        self,
        d_model: int,
        n_node_features: int,
        n_node_layers: int,
        n_edge_layers: int,
        n_iterations: int,
        hidden_activation: Optional[str] = "GELU",
        output_activation: Optional[str] = None,
        dropout: Optional[float] = 0.,
    ):
        super().__init__()
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                 
        
        # Setup input network
        self.node_encoder = make_mlp(
            n_node_features,
            [d_model]*n_node_layers,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            input_dropout=0.,
            hidden_dropout=dropout,
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * d_model,
            [d_model]*n_edge_layers,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            input_dropout=0.,
            hidden_dropout=dropout,
        )

        # Initialize GNN blocks
        layers = [
            InteractionGNNCell(
                d_model = d_model,
                n_node_layers = n_node_layers,
                n_edge_layers = n_edge_layers,
                hidden_activation = hidden_activation,
                output_activation = output_activation,
                dropout = dropout,
            )
            for _ in range(n_iterations)
        ]
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, node_attr, graph):
        
        node_attr.requires_grad = True
        nodes = self.node_encoder(node_attr)
        edges = self.edge_encoder(torch.cat([nodes[graph[0]], nodes[graph[1]]], dim = 1))
        
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, graph)
        
        return nodes, edges
    
class HierarchicalGNNBlock(nn.Module):

    """
    An hierarchical GNN class
    """

    def __init__(
        self,
        d_model: int,
        emb_size: int,
        n_node_layers: int,
        n_edge_layers: int,
        n_output_layers: int,
        n_iterations: int,
        hidden_activation: Optional[str] = "GELU",
        output_activation: Optional[str] = None,
        dropout: Optional[float] = 0.,
    ):
        super().__init__()
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                
        
        self.snode_encoder = make_mlp(
            d_model + emb_size,
            [d_model]*n_node_layers,
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,
            input_dropout=0.,
            hidden_dropout=dropout,
        )

        
        self.sedge_encoder = make_mlp(
            2 * d_model,
            [d_model]*n_edge_layers,
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,
            input_dropout=0.,
            hidden_dropout=dropout,
        )
        
        self.output_classifier = make_mlp(
            2 * d_model,
            [d_model]*(n_output_layers - 1) + [1],
            hidden_activation=hidden_activation,
            output_activation=None,
            input_dropout=0.,
            hidden_dropout=0.,
        )

        # Initialize GNN blocks
        layers = [
            HierarchicalGNNCell(
                d_model = d_model,
                n_node_layers = n_node_layers,
                n_edge_layers = n_edge_layers,
                hidden_activation = hidden_activation,
                output_activation = output_activation,
                dropout = dropout,
            )
            for _ in range(n_iterations)
        ]
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, nodes, edges, semb, graph, bgraph, bweights, sgraph, sweights):

        # Initialize supernode & edges by aggregating node features. Normalizing with 1-norm to improve training stability
        snodes = scatter_add((F.normalize(nodes, p=1)[bgraph[0]])*bweights, bgraph[1], dim=0, dim_size=semb.shape[0])
        snodes = torch.cat([semb, snodes], dim = -1)
        snodes = self.snode_encoder(snodes)
        sedges = self.sedge_encoder(torch.cat([snodes[sgraph[0]], snodes[sgraph[1]]], dim=1))

        for layer in self.layers:
            nodes, edges, snodes, sedges = layer(
                nodes,
                edges,
                snodes,
                sedges,
                graph,
                bgraph,
                bweights,
                sgraph,
                sweights
            )
            
        logits = self.output_classifier(torch.cat([nodes[bgraph[0]], snodes[bgraph[1]]], dim = 1))
        
        return logits.squeeze(1)


class Pooling(nn.Module):
    def __init__(
        self, 
        d_model: int,
        emb_size: int,
        n_output_layers: int,
        hidden_activation: Optional[str] = "GELU",
        output_activation: Optional[str] = None,
        dropout: Optional[float] = 0.,
        momentum: Optional[float] = 0.95,
        bsparsity: Optional[int] = 5,
        ssparsity: Optional[int] = 10,
        resolution: Optional[float] = 0., 
        min_size: Optional[int] = 5, 
    ):
        super().__init__()
        self.resolution = resolution
        self.min_size = min_size
        self.momentum = momentum
        self.gmm_model = GaussianMixture(2)
        self.node_encoder = make_mlp(
            d_model,
            [d_model] * (n_output_layers - 1) + [emb_size],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            input_dropout=dropout,
            hidden_dropout=dropout,
        )
        self.bgraph_construction = DynamicGraphConstruction(
            bsparsity,
            "exp",
            symmetrize=False,
            normalize=True,
            return_logits=True
        )
        self.sgraph_construction = DynamicGraphConstruction(
            ssparsity,
            "sigmoid",
            symmetrize=True,
            normalize=True,
            return_logits=False
        )
        self.register_buffer("score_cut", torch.tensor([0], dtype = torch.float), persistent = True)
        
    def get_quadratic_coeff(self, weight, mean, var):
        sigma = np.sqrt(var)
        a = -0.5/sigma**2
        b = mean/sigma**2
        c = -0.5*mean**2/sigma**2 - np.log(sigma) + np.log(weight)
        return a, b, c

    def solve_quadratic_eq(self, a, b, c):
        if b**2 > 4*a*c:
            return torch.as_tensor((-b + np.sqrt(b**2 - 4*a*c))/(2*a), dtype = torch.float)
        else:
            return torch.as_tensor(-b/(2*a), dtype = torch.float)

    def determine_cut(self):
        a1, b1, c1 = self.get_quadratic_coeff(self.gmm_model.weights_[0].item(), self.gmm_model.means_[0].item(), self.gmm_model.covariances_[0].item())
        a2, b2, c2 = self.get_quadratic_coeff(self.gmm_model.weights_[1].item(), self.gmm_model.means_[1].item(), self.gmm_model.covariances_[1].item())
        if self.gmm_model.means_[0][0] > self.gmm_model.means_[1][0]:
            return self.solve_quadratic_eq(a1-a2, b1-b2, c1-c2-self.resolution)
        else:
            return self.solve_quadratic_eq(a2-a1, b2-b1, c2-c1-self.resolution)
    
    def get_clusters(self, labels):
        _, inverse, counts = labels.unique(return_inverse = True, return_counts = True)
        valid_mask = counts[inverse] >= self.min_size
        idxs = labels[valid_mask].unique(return_inverse = True)[1]
        return valid_mask, idxs
    
    def forward(self, nodes, graph):
        
        emb = self.node_encoder(nodes)
        emb = F.normalize(emb)
        
        likelihood = - torch.log((emb[graph[0]] - emb[graph[1]]).square().sum(-1).clamp(min=1e-12))
            
        # GMM edge cutting
        self.gmm_model.fit(likelihood.unsqueeze(1).cpu().detach().numpy())
        cut = self.determine_cut()
        
        # Moving Average
        if self.training:
            self.score_cut = self.momentum * self.score_cut + (1 - self.momentum) * cut
    
        # Connected Components
        mask = likelihood >= self.score_cut.to(likelihood.device)
        graph = to_scipy_sparse_matrix(graph[:, mask], num_nodes = nodes.shape[0])
        _, labels = connected_components(graph, directed = False)
        valid_mask, idxs = self.get_clusters(torch.as_tensor(labels, dtype = torch.long, device = emb.device))
        original_bgraph = torch.stack([
            torch.arange(valid_mask.shape[0]).to(valid_mask.device)[valid_mask],
            idxs,
        ], dim = 0)
        
        # Compute centroids
        semb = scatter_add(emb[valid_mask], idxs, dim = 0)
        semb = F.normalize(semb)
        
        # Construct graphs
        bgraph, bweights, logits = self.bgraph_construction(emb, semb, original_bgraph)
        sgraph, sweights = self.sgraph_construction(semb, semb)
    
        return emb, semb, bgraph, bweights, sgraph, sweights, logits