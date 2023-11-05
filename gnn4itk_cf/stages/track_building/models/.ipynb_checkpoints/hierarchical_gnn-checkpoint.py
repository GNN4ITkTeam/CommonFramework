import sys

import torch.nn as nn
import torch
from torch_scatter import scatter_add, scatter_mean
from torch.utils.checkpoint import checkpoint
import numpy as np
import cudf
import cupy as cp
from sklearn.mixture import GaussianMixture
import cugraph
from scipy.optimize import fsolve
import numpy as np

from .bipartite_classification_base import BipartiteClassificationBase

from ..utils import InteractionGNNCell, HierarchicalGNNCell, DynamicGraphConstruction
from gnn4itk_cf.utils import make_mlp

    
class InteractionGNNBlock(nn.Module):

    """
    An interaction network for embedding class
    """

    def __init__(self, hparams, iterations):
        super().__init__()
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                 
        
        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"],
            [hparams["hidden"]]*(hparams["nb_node_layer"]-1) + [hparams["latent"]],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            affine_param=True,
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["spatial_channels"]),
            [hparams["hidden"]]*(hparams["nb_edge_layer"]-1) + [hparams["latent"]],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            affine_param=True,
        )

        # Initialize GNN blocks
        if hparams["share_weight"]:
            cell = InteractionGNNCell(hparams)
            ignn_cells = [
                cell
                for _ in range(iterations)
            ]
        else:
            ignn_cells = [
                InteractionGNNCell(hparams)
                for _ in range(iterations)
            ]
        
        self.ignn_cells = nn.ModuleList(ignn_cells)
        
        # output layers
        self.output_layer = make_mlp(
            hparams["latent"],
            [hparams["hidden"]]*(hparams["output_layers"]-1) + [hparams["emb_dim"]],
            layer_norm=hparams["layernorm"],
            output_activation= None,
            hidden_activation=hparams["hidden_output_activation"],
            affine_param=True,
        )
    
        
        self.hparams = hparams
        
    def forward(self, x, graph):
        
        # x.requires_grad = True
        
        nodes = checkpoint(self.node_encoder, x)
        edges = checkpoint(self.edge_encoder, torch.cat([x[graph[0]], x[graph[1]]], dim=1))
        
        for layer in self.ignn_cells:
            nodes, edges= layer(nodes, edges, graph)
        
        embeddings = self.output_layer(nodes)
        embeddings = nn.functional.normalize(embeddings) 
        
        return embeddings, nodes, edges
    
class HierarchicalGNNBlock(nn.Module):

    """
    An hierarchical GNN class
    """

    def __init__(self, hparams, logging):
        super().__init__()
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                
        
        self.supernode_encoder = make_mlp(
            hparams["latent"],
            [hparams["hidden"]]*(hparams["nb_node_layer"]-1) + [hparams["latent"] - hparams["emb_dim"]],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            affine_param=True,
        )

        # The edge network computes new edge features from connected nodes
        self.superedge_encoder = make_mlp(
            2 * hparams["latent"],
            [hparams["hidden"]]*(hparams["nb_edge_layer"]-1) + [hparams["latent"]],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            affine_param=True,
        )

        # Initialize GNN blocks
        if hparams["share_weight"]:
            cell = HierarchicalGNNCell(hparams)
            hgnn_cells = [
                cell
                for _ in range(hparams["n_hierarchical_graph_iters"])
            ]
        else:
            hgnn_cells = [
                HierarchicalGNNCell(hparams)
                for _ in range(hparams["n_hierarchical_graph_iters"])
            ]
        
        self.hgnn_cells = nn.ModuleList(hgnn_cells)
        
        # output layers
        self.GMM_model = GaussianMixture(2, warm_start = True)
        self.super_graph_construction = DynamicGraphConstruction("sigmoid", hparams)
        self.bipartite_graph_construction = DynamicGraphConstruction("exp", hparams)
        self.register_buffer("score_cut", torch.tensor([float("inf")]))
        
        self.log = logging
        self.hparams = hparams
    
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
        a1, b1, c1 = self.get_quadratic_coeff(self.GMM_model.weights_[0].item(), self.GMM_model.means_[0].item(), self.GMM_model.covariances_[0].item())
        a2, b2, c2 = self.get_quadratic_coeff(self.GMM_model.weights_[1].item(), self.GMM_model.means_[1].item(), self.GMM_model.covariances_[1].item())
        if self.GMM_model.means_[0][0] > self.GMM_model.means_[1][0]:
            return self.solve_quadratic_eq(a1-a2, b1-b2, c1-c2-self.hparams["cluster_granularity"])
        else:
            return self.solve_quadratic_eq(a2-a1, b2-b1, c2-c1-self.hparams["cluster_granularity"])
    
    def get_cluster_labels(self, connected_components, x):
        
        clusters = -torch.ones(len(x), device = x.device).long()
        labels = torch.as_tensor(connected_components["labels"], device = x.device)
        vertex = torch.tensor(connected_components["vertex"], device = x.device) 
        _, inverse, counts = labels.unique(return_inverse = True, return_counts = True)
        mask = counts[inverse] >= self.hparams["min_cluster_size"]
        clusters[vertex[mask]] = labels[mask].unique(return_inverse = True)[1].long()
        
        return clusters
      
    def clustering(self, x, embeddings, graph):
        with torch.no_grad():
            
            # Compute cosine similarity transformed by archypertangent
            likelihood = torch.einsum('ij,ij->i', embeddings[graph[0]], embeddings[graph[1]])
            likelihood = torch.atanh(torch.clamp(likelihood, min=-1+1e-7, max=1-1e-7))
            
            # GMM edge cutting
            self.GMM_model.fit(likelihood[torch.randperm(len(likelihood))[:10000]].unsqueeze(1).cpu().numpy())
            
            # in the case of score cut not initialized, initialize it from the middle point of the two distribution
            if self.score_cut == float("inf"): 
                self.score_cut = torch.tensor([self.GMM_model.means_.mean().item()], device = self.score_cut.device)
    
            cut = self.determine_cut()
            
            # Exponential moving average for score cut
            momentum = 0.95
            if self.training:
                self.score_cut = momentum*self.score_cut + (1-momentum)*cut
        
            if self.training:
                self.log("score_cut", self.score_cut.item())
            
            # Connected Components
            mask = likelihood >= self.score_cut.to(likelihood.device)
            try:
                G = cugraph.Graph()
                df = cudf.DataFrame({"src": cp.asarray(graph[0, mask]),
                                     "dst": cp.asarray(graph[1, mask]),
                                    })            
                G.from_cudf_edgelist(df, source = "src", destination = "dst")
                connected_components = cugraph.components.connected_components(G)
                clusters = self.get_cluster_labels(connected_components, x)
                if clusters.max() <= 2:
                    raise ValueError
            except ValueError:
                # Sometimes all edges got cut, then it will run into value error. In that case just use the original graph
                G = cugraph.Graph()
                df = cudf.DataFrame({"src": cp.asarray(graph[0]),
                                     "dst": cp.asarray(graph[1]),
                                    })            
                G.from_cudf_edgelist(df, source = "src", destination = "dst")
                connected_components = cugraph.components.connected_components(G)
                clusters = self.get_cluster_labels(connected_components, x)
            
            return clusters
        
    def forward(self, x, embeddings, nodes, edges, graph):
   
        # Compute clustering
        clusters = self.clustering(x, embeddings, graph)

        # Compute Centers
        means = scatter_mean(embeddings[clusters >= 0], clusters[clusters >= 0], dim=0, dim_size=clusters.max()+1)
        means = nn.functional.normalize(means)
        
        # Construct Graphs
        super_graph, super_edge_weights = self.super_graph_construction(means, means, sym = True, norm = True, k = self.hparams["supergraph_sparsity"])
        bipartite_graph, bipartite_edge_weights, attention_logits = self.bipartite_graph_construction(embeddings, means, sym = False, norm = True, k = self.hparams["bipartitegraph_sparsity"], logits = True)
        
        attention_logits = attention_logits - attention_logits.mean()
        if self.training:
            self.log("clusters", len(means))
        
        # Initialize supernode & edges by aggregating node features. Normalizing with 1-norm to improve training stability
        supernodes = scatter_add((nn.functional.normalize(nodes, p=1)[bipartite_graph[0]])*bipartite_edge_weights, bipartite_graph[1], dim=0, dim_size=means.shape[0])
        supernodes = torch.cat([means, checkpoint(self.supernode_encoder, supernodes)], dim = -1)
        superedges = checkpoint(self.superedge_encoder, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]]], dim=1))

        for layer in self.hgnn_cells:
            nodes, edges, supernodes, superedges = layer(nodes,
                                                         edges,
                                                         supernodes,
                                                         superedges,
                                                         graph,
                                                         bipartite_graph,
                                                         bipartite_edge_weights,
                                                         super_graph,
                                                         super_edge_weights)
            
        
        return nodes, supernodes, bipartite_graph, attention_logits
    
class HierarchicalGNN(BipartiteClassificationBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams) 
        
        self.ignn_block = InteractionGNNBlock(hparams, hparams["n_interaction_graph_iters"])
        self.hgnn_block = HierarchicalGNNBlock(hparams, self.log)
        
        self.bipartite_output_layer = make_mlp(
            2 * hparams["latent"],
            [hparams["hidden"]]*(hparams["output_layers"]-1) + [1],
            layer_norm=hparams["layernorm"],
            output_activation= None,
            hidden_activation=hparams["hidden_output_activation"],
            affine_param=True,
        )
        
        
    def forward(self, x, graph):
        
        x.requires_grad = True
        
        directed_graph = torch.cat([graph, graph.flip(0)], dim = 1)
        
        intermediate_embeddings, nodes, edges = self.ignn_block(x, directed_graph)
        
        nodes, supernodes, bipartite_graph, attention_logits = self.hgnn_block(x, intermediate_embeddings, nodes, edges, directed_graph) 
        
        bipartite_score_logits = checkpoint(self.bipartite_output_layer, torch.cat([nodes[bipartite_graph[0]], supernodes[bipartite_graph[1]]], dim = 1)).squeeze()
        
        return bipartite_graph, bipartite_score_logits, intermediate_embeddings, attention_logits
