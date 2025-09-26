# Copyright (C) 2023-2025 CERN for the benefit of the ATLAS collaboration

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


# Local imports
from ..track_building_stage import TrackBuildingStage
from .. import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyMMGEdgeLayerConnector(TrackBuildingStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialize the PYMMGEdgeLayerConnector model, a Python wrapper built on PyMMG (https://gitlab.cern.ch/gnn4itkteam/pymodulemapgraph) 
        to interface with ModuleMapGraph (https://gitlab.cern.ch/gnn4itkteam/ModuleMapGraph).
        """
        try:
            from pymmg import EdgeLayerConnector
        except ImportError:
            self.log.error("Failed to import EdgeLayerConnector from pymmg. Please ensure pymmg is installed.")
            raise

        if torch.cuda.is_available():
            self.device = "cuda"
            self.log.info("Using CUDA for graph construction")
        else:
            raise RuntimeError("CUDA runtime not available. MMG currently requires an NVIDIA GPU with CUDA.")
        self.hparams = hparams

        self.connector = EdgeLayerConnector(
            device=int(hparams.get("gpu_core", 0)),
            nb_blocks=int(hparams.get("gpu_nb_blocks", 512)),
            weights_cut=float(hparams.get("edge_layer_connector_weights_cut", 0.01)),
            min_hits=int(hparams.get("edge_layer_connector_min_hits", 4)),
        )

    def build_tracks(self, dataset, data_name):

        self.log.info("Using PyMMG_EdgeLayerConnector method to reconstruct the tracks")

        output_dir = os.path.join(self.hparams["stage_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Saving tracks to {output_dir}")

        for event in tqdm(dataset, desc=f"Reconstructing tracks for {data_name} data"):
            self._build_tracks_one_evt(event, output_dir=output_dir)

    def _build_tracks_one_evt(self, event, output_dir):
        """
        Build tracks for one event with the PyMMG EdgeLayerConnector
        """
        os.sched_setaffinity(0, range(1000))
        if self.hparams.get("on_true_graph", False):
            event.edge_scores = event.edge_y.float()

        # Build tracks with PyMMG EdgeLayerConnector.
        # event.hit_id, event.edge_index and event.edge_scores are required and will be used in self.connector to build tracks.
        _tracks = self.connector.build_tracks_one_evt(event)

        if self.hparams.get("save_tracks", True):
            self.save_tracks(event, _tracks, output_dir)

        if self.hparams.get("save_graph", True):
            event = self.save_graph(event, output_dir)
