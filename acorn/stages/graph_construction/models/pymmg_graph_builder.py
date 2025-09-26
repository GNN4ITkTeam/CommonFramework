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

# Local imports
from ..graph_construction_stage import GraphConstructionStage
from . import utils
from acorn.utils.loading_utils import remove_variable_name_prefix_in_pyg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyMMGGraphBuilder(GraphConstructionStage):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialize the PYMMGGraphBuilder model, a Python wrapper built on PyMMG (https://gitlab.cern.ch/gnn4itkteam/pymodulemapgraph) 
        to interface with ModuleMapGraph (https://gitlab.cern.ch/gnn4itkteam/ModuleMapGraph).
        """
        try:
            from pymmg import GraphBuilder
        except ImportError:
            self.log.error("Failed to import GraphBuilder from pymmg. Please ensure pymmg is installed.")
            raise

        if torch.cuda.is_available():
            self.device = "cuda"
            self.log.info("Using CUDA for graph construction")
        else:
            raise RuntimeError("CUDA runtime not available. MMG currently requires an NVIDIA GPU with CUDA.")
        self.hparams = hparams

        # Logging config
        self.log = logging.getLogger("ModuleMapGraph")
        log_level = self.hparams["log_level"].upper() if "log_level" in self.hparams else "WARNING"

        if log_level == "WARNING":
            self.log.setLevel(logging.WARNING)
        elif log_level == "INFO":
            self.log.setLevel(logging.INFO)
        elif log_level == "DEBUG":
            self.log.setLevel(logging.DEBUG)
        else:
            raise ValueError(f"Unknown logging level {log_level}")

        self._graph_builder = GraphBuilder(self.hparams["module_map_pattern_path"])

    def to(self, device):
        return self

    def build_graphs(self, dataset, data_name):
        """
        Build the graphs for the data.
        """

        output_dir = os.path.join(self.hparams["stage_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)
        self.log.info(f"Building graphs for {data_name}")

        for graph in tqdm(dataset):
            if graph is None:
                continue
            if os.path.exists(os.path.join(output_dir, f"event{graph.event_id}.pyg")):
                print(f"Graph {graph.event_id} already exists, skipping...")
                continue

            graph = self.build_graph(graph)
            if not self.hparams.get("variable_with_prefix"):
                graph = remove_variable_name_prefix_in_pyg(graph)
            torch.save(graph, os.path.join(output_dir, f"event{graph.event_id}.pyg"))

    def build_graph(self, graph):

        graph.edge_index = self._graph_builder.build_edge_index(
            hit_id=graph.hit_id,
            hit_module_id=graph.hit_module_id,
            hit_x=graph.hit_x,
            hit_y=graph.hit_y,
            hit_z=graph.hit_z,
            nb_hits=graph.hit_id.shape[0],
        )

        y, truth_map = utils.graph_intersection(
            graph.edge_index.to(device),
            graph.track_edges.to(device),
            return_y_pred=True,
            return_truth_to_pred=True,
        )
        graph.edge_y = y.cpu()
        graph.track_to_edge_map = truth_map.cpu()

        return graph
