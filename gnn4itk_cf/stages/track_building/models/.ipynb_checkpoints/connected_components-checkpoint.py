# 3rd party imports
import os
import logging
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sps
from tqdm import tqdm
try:
    import cudf
except ImportError:
    logging.warning("cuDF not found, using pandas instead")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local imports
from ..track_building_stage import TrackBuildingStage
from torch_geometric.utils import to_scipy_sparse_matrix

class ConnectedComponents(TrackBuildingStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
        """
        self.hparams = hparams
        self.gpu_available = torch.cuda.is_available()        

    def build_tracks(self, dataset, data_name):
        """
        Given a set of scored graphs, and a score cut, build tracks from graphs by:
        1. Applying the score cut to the graph
        2. Converting the graph to a sparse scipy array
        3. Running connected components on the sparse array
        4. Assigning the connected components labels back to the graph nodes as `labels` attribute
        """

        output_dir = os.path.join(self.hparams["stage_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Saving tracks to {output_dir}")

        for graph in tqdm(dataset):

            # Apply score cut
            edge_mask = graph.scores > self.hparams["score_cut"]

            # Convert to sparse scipy array
            sparse_edges = to_scipy_sparse_matrix(graph.edge_index[:, edge_mask], num_nodes = graph.x.size(0))

            # Run connected components
            _, candidate_labels = sps.csgraph.connected_components(sparse_edges, directed=False, return_labels=True)  
            graph.labels = torch.from_numpy(candidate_labels).long()

            # TODO: Graph name file??
            torch.save(graph, os.path.join(output_dir, f"event{graph.event_id}.pyg"))
