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
from tqdm.contrib.concurrent import process_map
from functools import partial

from . import cc_and_walk_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local imports
from ..track_building_stage import TrackBuildingStage


class CCandWalk(TrackBuildingStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the CCandWalk - chaining of connected components and walkthrough algorithm (à la L2IT) 
        """
        self.hparams = hparams
        self.gpu_available = torch.cuda.is_available()
        self.cc_only = self.hparams.get("cc_only", False)

    def build_tracks(self, dataset, data_name):
        """
        Explain here the algorithm
        """

        self.log.info("Using CCandWalk method to reconstruct the tracks")

        output_dir = os.path.join(self.hparams["stage_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Saving tracks to {output_dir}")

        max_workers = (
            self.hparams["max_workers"] if "max_workers" in self.hparams else None
        )
        if max_workers != 1:
            process_map(
                partial(self._build_tracks_one_evt, output_dir=output_dir),
                dataset,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Reconstructing tracks for {data_name} data",
            )
        else:
            for event in tqdm(
                dataset, desc=f"Reconstructing tracks for {data_name} data"
            ):
                self._build_tracks_one_evt(event, output_dir=output_dir)

    def _build_tracks_one_evt(self, graph, output_dir):
        """
        Build tracks for one event from connected components + walkthrough
        """
        os.sched_setaffinity(0, range(1000))

        all_trks = dict()

        if self.hparams.get("on_true_graph", False):
            score_name = "y"
            threshold = 0
        else:
            score_name = "scores"
            threshold = self.hparams["score_cut_cc"]

        # Remove cycles by pointing all edges outwards (necessary for topo-sort)
        R = graph.r**2 + graph.z**2
        graph = cc_and_walk_utils.remove_cycles(graph)

        # remove low-scoring edges
        G = cc_and_walk_utils.filter_graph(graph, score_name, threshold)
        # topological sort is really needed only if we run the wrangler afterwards
        G = cc_and_walk_utils.topological_sort_graph(G)

        # Simple paths from connected components
        all_trks["cc"] = cc_and_walk_utils.get_simple_path(G)

        # Walkthrough on remaining non simple paths (ie with branching)
        if not self.cc_only:
            all_trks["walk"] = cc_and_walk_utils.walk_through(
                G,
                score_name,
                self.hparams["score_cut_walk"]["min"],
                self.hparams["score_cut_walk"]["add"],
            )

        # Add tracks labels to the graph
        cc_and_walk_utils.add_track_labels(graph, all_trks)

        torch.save(graph, os.path.join(output_dir, f"event{graph.event_id[0]}.pyg"))
