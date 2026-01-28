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
from scipy.sparse.csgraph import connected_components
from torch_scatter import scatter_max, scatter_min
import numpy as np
import pandas as pd

from torch_geometric.utils import to_scipy_sparse_matrix
from acorn.stages.track_building.models.cc_and_walk_utils import remove_cycles
from acorn.stages.track_building.track_building_stage import TrackBuildingStage


class EdgeMending(TrackBuildingStage):
    def __init__(self, hparams):
        super().__init__(hparams)

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
            event = remove_cycles(event)

            # Initialize the array to track which hit to keep
            to_keep = torch.ones_like(event.hit_id, dtype=torch.bool)
            # random edge removal
            # edge_mask = torch.rand(event.edge_index.shape[1]) >= self.hparams.get(
            #     "random_drop", 0
            # )
            # run cc
            chain_edges = event.edge_index[
                :, (event.edge_scores > self.hparams["score_cut"])
            ]
            graph = to_scipy_sparse_matrix(
                chain_edges, num_nodes=event.hit_id.shape[0]
            ).tocsr()

            _, track_id = connected_components(graph, directed=False)

            # remove any not simple tracks
            out_degree = np.bincount(chain_edges[0], minlength=event.hit_id.shape[0])
            in_degree = np.bincount(chain_edges[1], minlength=event.hit_id.shape[0])

            # find tracks with junctions and short tracks

            tracks_to_remove = np.zeros(track_id.max() + 1, dtype=bool)
            np.bitwise_or.at(tracks_to_remove, track_id[in_degree > 1], 1)
            np.bitwise_or.at(tracks_to_remove, track_id[out_degree > 1], 1)
            nhits = np.bincount(track_id)
            tracks_to_remove |= nhits < self.hparams["min_chain_length"]

            # remove the bad tracks
            track_id[tracks_to_remove[track_id]] = -1

            # store the good tracks and update tracker
            track_id = torch.as_tensor(track_id, dtype=torch.long)

            # to_keep is a list of all nodes in the not-simple tracks. It's a bool with length num hits and is true for not-simple tracks
            to_keep[track_id >= 0] = False

            # Apply the score cut
            junction_edges = event.edge_index[
                :, (event.edge_scores > self.hparams["junction_cut"])
            ]
            junction_score = event.edge_scores[
                (event.edge_scores > self.hparams["junction_cut"])
            ]
            junction_score = junction_score[to_keep[junction_edges].all(0)]
            junction_edges = junction_edges[:, to_keep[junction_edges].all(0)]

            out_degree = np.bincount(junction_edges[0], minlength=event.hit_x.shape[0])
            in_degree = np.bincount(junction_edges[1], minlength=event.hit_x.shape[0])

            x_nodes = np.where((in_degree == 2) & np.equal(in_degree, out_degree))[0]
            in_ind = np.isin(junction_edges[1], x_nodes)
            out_ind = np.isin(junction_edges[0], x_nodes)
            x_chain_edges_out = junction_edges[:, (out_ind)]
            x_chain_edges_in = junction_edges[:, (in_ind)]

            out_order = np.argsort(x_chain_edges_out[0])
            in_order = np.argsort(x_chain_edges_in[1])
            x_out = x_chain_edges_out[:, out_order][0]  # changed
            x_in = x_chain_edges_in[:, in_order][1]  # changed
            in_scores = junction_score[in_ind][in_order]
            out_scores = junction_score[out_ind][out_order]

            in_min = scatter_min(in_scores, x_in)[1]
            in_min = in_min[in_min < x_chain_edges_in.size(1)].numpy()
            out_min = scatter_min(out_scores, x_in)[1]
            out_min = out_min[out_min < x_chain_edges_out.size(1)].numpy()

            # make mended edges
            out_nodes_for_mend = x_chain_edges_out[1][out_order][out_min]
            in_nodes_for_mend = x_chain_edges_in[0][in_order][in_min]
            mend_edges = torch.tensor(np.array([in_nodes_for_mend, out_nodes_for_mend]))

            # Masking out the junctions

            mask = torch.zeros_like(junction_edges[0], dtype=torch.bool)

            in_max = scatter_max(junction_score, junction_edges[1])[1]
            in_max = in_max[in_max < junction_edges.size(1)].numpy()
            out_max = scatter_max(junction_score, junction_edges[0])[1]
            out_max = out_max[out_max < junction_edges.size(1)].numpy()
            mask.index_fill_(0, torch.as_tensor(np.intersect1d(in_max, out_max)), True)

            junction_edges = torch.cat((junction_edges[:, mask], mend_edges), dim=1)
            # build csr graph and run cc
            graph = to_scipy_sparse_matrix(
                junction_edges, num_nodes=event.hit_id.shape[0]
            )
            _, labels = connected_components(graph, directed=False)
            labels = torch.as_tensor(labels, dtype=torch.long)
            nhits = torch.bincount(labels)
            track_id[to_keep & (nhits[labels] > 1)] = (
                labels[to_keep & (nhits[labels] > 1)] + track_id.max() + 1
            )

            event.labels = track_id
            event.config.append(self.hparams)
            # torch.save(event, os.path.join(output_dir, f"event{event.event_id[0]}.pyg"))

            # Make a dataframe from pyg graph
            # d = utils.load_reconstruction_df(event)
            d = pd.DataFrame({"hit_id": event.hit_id, "track_id": event.labels})

            # include distance from origin to sort hits
            # d["r2"] = (event.hit_r**2 + event.hit_z**2).cpu().numpy()
            # Keep only hit_id associtated to a tracks (label >= 0, not -1), sort by track_id and r2
            d = d[d.track_id >= 0]
            # Make a dataframe of list of hits (one row = one list of hits, ie one track)
            tracks = d.groupby("track_id")["hit_id"].apply(list)

            if self.hparams.get("save_tracks", True):
                self.save_tracks(event, tracks, output_dir)

            if self.hparams.get("save_graph", True):
                graph = self.save_graph(event, output_dir)
