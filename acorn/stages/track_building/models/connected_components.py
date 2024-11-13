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

import logging
import os
from functools import partial
from time import process_time

import scipy.sparse as sps
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from acorn.stages.track_building import utils
from acorn.stages.track_building.track_building_stage import (
    GraphDataset,
    TrackBuildingStage,
)
from acorn.utils.loading_utils import remove_variable_name_prefix_in_pyg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local imports
from torch_geometric.utils import to_scipy_sparse_matrix


class ConnectedComponents(TrackBuildingStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the PyModuleMap - a python implementation of the Triplet Module Map.
        """
        self.hparams = hparams
        self.gpu_available = torch.cuda.is_available()
        self.on_truth_particle_edges = self.hparams.get(
            "on_truth_particle_edges", False
        )
        self.on_true_graph_edges = self.hparams.get("on_true_graph_edges", False)

        assert not all(
            [self.on_truth_particle_edges, self.on_true_graph_edges]
        ), "Cannot process both truth and true graph, choose one"
        if self.on_truth_particle_edges:
            self.log.info("Using truth particle edges as input graph for track reco")
        if self.on_true_graph_edges:
            self.log.info("Using true graph edges in the input graph for track reco")

    def _build_event(
        self,
        graph: Data,
        on_truth_particle_edges: bool = False,
        on_true_graph_edges: bool = False,
        score_cut: float = 0.5,
    ):
        """
        Build track candidates for 1 event.
        graph: input graph, must have track_edges, edge_index, scores
        on_truth_particle_edges: whether or not build tracks using truth graph (track_edges)
        on_true_graph_edges : whether or not to build tracks using true edges in the graph (y)
        score_cut: only used if both on_truth_particle_edges and on_true_graph_edges are set to False, a score cut to get edges prediction
        """
        # get edge list based on input selection
        start_time = process_time()
        if on_truth_particle_edges:
            edges = graph.track_edges
        elif on_true_graph_edges:
            edges = graph.edge_index[:, graph.y]
        else:
            edge_mask = graph.edge_scores > score_cut
            edges = graph.edge_index[:, edge_mask]

        # Get number of nodes
        if hasattr(graph, "hit_x"):
            num_nodes = graph.hit_x.size(0)
        else:
            num_nodes = graph.edge_index.max().item() + 1

        # remove isolated nodes
        edges, _, mask = remove_isolated_nodes(edges, num_nodes=num_nodes)

        # Convert to sparse scipy array
        sparse_edges = to_scipy_sparse_matrix(edges, num_nodes=mask.sum().item())

        # Run connected components
        _, candidate_labels = sps.csgraph.connected_components(
            sparse_edges, directed=False, return_labels=True
        )

        # get labels. isolated nodes get -1 as label
        labels = (torch.ones(num_nodes) * -1).long()
        labels[mask] = torch.from_numpy(candidate_labels).long()
        graph.edge_track_labels = labels
        graph.time_taken = process_time() - start_time

        return graph

    def _build_and_save(
        self,
        graph: Data,
        output_dir: str,
        on_truth_particle_edges: bool = False,
        on_true_graph_edges: bool = False,
        score_cut: float = 0.5,
    ):
        """
        Build and save to disk
        """
        graph = self._build_event(
            graph, on_truth_particle_edges, on_true_graph_edges, score_cut
        )
        graph.config.append(self.hparams)
        # Make a dataframe from pyg graph
        d = utils.load_reconstruction_df(graph)
        # include distance from origin to sort hits
        d["r2"] = (graph.hit_r**2 + graph.hit_z**2).cpu().numpy()
        # Keep only hit_id associtated to a tracks (label >= 0, not -1), sort by track_id and r2
        d = d[d.track_id >= 0].sort_values(["track_id", "r2"])
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
        if not self.hparams.get("variable_with_prefix"):
            graph = remove_variable_name_prefix_in_pyg(graph)
        torch.save(graph, os.path.join(output_dir, f"event{graph.event_id[0]}.pyg"))

    def build_tracks(self, dataset: GraphDataset, data_name: str):
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

        max_workers = self.hparams.get("max_workers", 1)
        build_func = partial(
            self._build_and_save,
            output_dir=output_dir,
            on_truth_particle_edges=self.on_truth_particle_edges,
            on_true_graph_edges=self.on_true_graph_edges,
            score_cut=self.hparams.get("score_cut", 0.5),
        )

        if max_workers != 1:
            process_map(
                build_func,
                dataset,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Reconstructing tracks for {data_name} data",
            )
        else:
            for event in tqdm(
                dataset, desc=f"Reconstructing tracks for {data_name} data"
            ):
                build_func(event)
