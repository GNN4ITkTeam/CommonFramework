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

# 3rd party imports
import os
import logging
import torch
import scipy.sparse as sps
from tqdm import tqdm


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

            # Get number of nodes
            if hasattr(graph, "num_nodes"):
                num_nodes = graph.num_nodes
            elif hasattr(graph, "x"):
                num_nodes = graph.x.size(0)
            elif hasattr(graph, "x_x"):
                num_nodes = graph.x_x.size(0)
            else:
                num_nodes = graph.edge_index.max().item() + 1

            # Convert to sparse scipy array
            sparse_edges = to_scipy_sparse_matrix(
                graph.edge_index[:, edge_mask], num_nodes=num_nodes
            )

            # Run connected components
            _, candidate_labels = sps.csgraph.connected_components(
                sparse_edges, directed=False, return_labels=True
            )
            graph.labels = torch.from_numpy(candidate_labels).long()
            graph.config.append(self.hparams)

            # TODO: Graph name file??
            torch.save(graph, os.path.join(output_dir, f"event{graph.event_id[0]}.pyg"))
