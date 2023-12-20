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
import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local imports
from ..track_building_stage import TrackBuildingStage
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import remove_isolated_nodes, to_networkx, degree
from torch_geometric.data import Data

from .. import utils


class WeaklyConnectedComponentsAllSimplePath(TrackBuildingStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Model to process labels of track candidate id for nodes in the graph.
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
            # TODO check isolated nodes

            graph.labels = -torch.ones(graph.hit_id.shape).long()

            graph_cc = Data(x=graph.hit_id, edge_index=graph.edge_index)
            # Apply score cut
            edge_mask = graph.scores > self.hparams["score_cut_high"]
            # Get number of nodes
            num_nodes = graph_cc.x.shape[0]
            # Keep only non-masked edges
            graph_cc.edge_index = graph_cc.edge_index[:, edge_mask]
            # Remove isolated nodes
            edge_index, _, mask_x = remove_isolated_nodes(
                graph_cc.edge_index, num_nodes=num_nodes
            )
            graph_cc.edge_index = edge_index
            graph_cc.x = graph_cc.x[mask_x]
            num_nodes = graph_cc.x.shape[0]

            src, dst = graph_cc.edge_index
            in_degree = degree(dst, num_nodes=num_nodes)
            out_degree = degree(src, num_nodes=num_nodes)

            # Convert to sparse scipy array
            sparse_edges = to_scipy_sparse_matrix(
                graph_cc.edge_index, num_nodes=num_nodes
            )
            # Run connected components
            n_cc, candidate_labels = sps.csgraph.connected_components(
                sparse_edges, directed=True, connection="weak", return_labels=True
            )
            # print("Found ", n_cc, "Weakly Connected Components")

            graph.labels[mask_x] = torch.from_numpy(candidate_labels).long()

            #####################################################
            graph_residual = Data(x=graph.hit_id, edge_index=graph.edge_index)
            edge_mask = (graph.scores > self.hparams["score_cut_low"]) & (
                graph.scores < self.hparams["score_cut_high"]
            )
            graph_residual.edge_index = graph_residual.edge_index[:, edge_mask]
            num_nodes = graph_residual.x.shape[0]
            edge_index, _, mask_x = remove_isolated_nodes(
                graph_residual.edge_index, num_nodes=num_nodes
            )
            graph_residual.edge_index = edge_index
            graph_residual.x = graph_residual.x[mask_x]

            graph_residual_nx = to_networkx(graph_residual)
            sources = [
                node
                for node, in_degree in graph_residual_nx.in_degree()
                if in_degree == 0
            ]
            targets = [
                node
                for node, out_degree in graph_residual_nx.out_degree()
                if out_degree == 0
            ]

            labels = {node_idx: -1 for node_idx in graph_residual_nx.nodes}
            label = torch.max(graph.labels) + 1
            for source in sources:
                paths = nx.all_simple_paths(
                    graph_residual_nx, source=source, target=targets, cutoff=25
                )
                for path in paths:
                    if len(path) > 7:
                        # print(path)
                        labels.update({node_idx: label for node_idx in path})
                        label += 1

            graph.labels[mask_x] = torch.tensor(list(labels.values()))

            graph.config.append(self.hparams)

            # TODO: Graph name file??
            # torch.save(graph, os.path.join(output_dir, f"event{graph.event_id[0]}.pyg"))

            # Make a dataframe from pyg graph
            d = utils.load_reconstruction_df(graph)
            # Keep only hit_id associtated to a tracks (label >= 0, not -1)
            d = d[d.track_id >= 0]
            # Make a dataframe of list of hits (one row = one list of hits, ie one track)
            tracks = d.groupby("track_id")["hit_id"].apply(list)
            os.makedirs(
                os.path.join(
                    self.hparams["stage_dir"], os.path.basename(output_dir) + "_tracks"
                ),
                exist_ok=True,
            )
            with open(
                os.path.join(
                    self.hparams["stage_dir"],
                    os.path.basename(output_dir) + "_tracks",
                    f"event{graph.event_id[0]}.txt",
                ),
                "w",
            ) as f:
                f.write(
                    "\n".join(
                        str(t).replace(",", "").replace("[", "").replace("]", "")
                        for t in tracks.values
                    )
                )
