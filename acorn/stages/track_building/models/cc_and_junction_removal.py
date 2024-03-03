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
from tqdm import tqdm
from time import process_time
from scipy.sparse.csgraph import connected_components
import numpy as np

# Local imports
from .cc_and_walk_utils import remove_cycles
from ..track_building_stage import TrackBuildingStage
from torch_geometric.utils import to_scipy_sparse_matrix


class CCandJunctionRemoval(TrackBuildingStage):
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

        for event in tqdm(dataset):
            start_time = process_time()
            event = remove_cycles(event)
            # Initialize the array to track which hit to keep
            to_keep = torch.ones_like(event.hit_id, dtype=torch.bool)
            # random edge removal
            edge_mask = torch.rand(event.edge_index.shape[1]) >= self.hparams.get(
                "random_drop", 0
            )

            # Select good tracks with at least `min_hits` hits
            if self.hparams["score_cut"]:
                # run cc
                chain_edges = event.edge_index[
                    :, (event.scores > self.hparams["score_cut"]) & edge_mask
                ]
                graph = to_scipy_sparse_matrix(
                    chain_edges, num_nodes=event.hit_id.shape[0]
                ).tocsr()
                _, track_id = connected_components(graph, directed=False)

                # remove short tracks
                _, inverse, nhits = np.unique(
                    track_id, return_counts=True, return_inverse=True
                )
                track_id[nhits[inverse] < self.hparams["min_chain_length"]] = -1

                # remove any not simple tracks
                out_hit_id, out_degree = np.unique(
                    chain_edges[0], return_counts=True, return_inverse=False
                )
                in_hit_id, in_degree = np.unique(
                    chain_edges[1], return_counts=True, return_inverse=False
                )
                track_id[np.isin(track_id, track_id[in_hit_id[in_degree > 1]])] = -1
                track_id[np.isin(track_id, track_id[out_hit_id[out_degree > 1]])] = -1

                # store the good tracks and update tracker
                track_id = torch.as_tensor(track_id, dtype=torch.long)
                to_keep[track_id >= 0] = False

            # perform clustering
            if self.hparams["junction_cut"]:
                # Apply the score cut
                junction_edges = event.edge_index[
                    :, (event.scores > self.hparams["junction_cut"]) & edge_mask
                ]
                junction_edges = junction_edges[:, to_keep[junction_edges].all(0)]

                # Compute in and out degrees
                out_hit_id, out_degree = torch.unique(
                    junction_edges[0], return_counts=True, return_inverse=False
                )
                in_hit_id, in_degree = torch.unique(
                    junction_edges[1], return_counts=True, return_inverse=False
                )

                # Masking out the junctions
                mask = torch.isin(
                    junction_edges[0], out_hit_id[out_degree <= 1]
                ) | torch.isin(junction_edges[1], out_hit_id[out_degree <= 1])
                junction_edges = junction_edges[:, mask]

                # build csr graph and run cc
                graph = to_scipy_sparse_matrix(
                    junction_edges, num_nodes=event.x.shape[0]
                )
                _, labels = connected_components(graph, directed=False)
                labels = torch.as_tensor(labels, dtype=torch.long)
                _, inverse, counts = torch.unique(
                    labels, return_counts=True, return_inverse=True
                )
                track_id[to_keep & (counts[inverse] > 1)] = (
                    labels[to_keep & (counts[inverse] > 1)] + track_id.max() + 1
                )

            event.labels = track_id
            event.config.append(self.hparams)
            event.time_taken = process_time() - start_time

            # TODO: Graph name file??
            torch.save(event, os.path.join(output_dir, f"event{event.event_id[0]}.pyg"))
