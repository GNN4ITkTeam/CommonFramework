"""
This class represents the entire logic of the graph construction stage. In particular, it
1. Loads events from the Athena-dumped csv files
2. Processes them into PyG Data objects with the specificied structure (see docs)
3. Runs the training of the metric learning or module map
4. Can run inference to build graphs
5. Can run evaluation to plot/print the performance of the graph construction

TODO: Update structure with the latest Gravnet base class
"""

import sys
sys.path.append("../")
import os
import warnings
import re

from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import Dataset
import torch
import numpy as np
import pandas as pd
from atlasify import atlasify
import matplotlib.pyplot as plt
from tqdm import tqdm

from gnn4itk_cf.utils import run_data_tests, get_ratio

class GraphConstructionStage:
    def __init__(self):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        
        self.trainset, self.valset, self.testset = None, None, None
        self.use_pyg = False # You can set this to True if you want to use the PyG LightningModule
        self.use_csv = False # You can set this to True if you want to use CSV Reading
        self.dataset_class = EventDataset

    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        if stage in ["fit", "predict"]:
            self.load_data(self.hparams["input_dir"])
            
            if self.use_pyg and not self.use_csv:
                self.test_data()
        
        elif stage == "test":
            self.load_data(self.hparams["stage_dir"])

        
    def load_data(self, input_dir):
        """
        Load in the data for training, validation and testing.
        """

        for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
            dataset = self.dataset_class(input_dir, data_name, data_num, use_pyg = self.use_pyg, use_csv = self.use_csv, hparams = self.hparams)
            setattr(self, data_name, dataset)
        
        print(f"Loaded {len(self.trainset)} training events, {len(self.valset)} validation events and {len(self.testset)} testing events")

    def test_data(self):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = ["x", "track_edges"]
        optional_features = ["pid", "n_hits", "primary", "pdg_id", "ghost", "shared", "module_id", "region_id", "hit_id"]

        run_data_tests([self.trainset, self.valset, self.testset], required_features, optional_features)

        # TODO: Add test for the building of input data
        # assert self.trainset[0].x.shape[1] == self.hparams["spatial_channels"], "Input dimension does not match the data"

        # TODO: Add test for the building of truth data

    @classmethod
    def infer(cls, config):
        """ 
        The gateway for the inference stage. This class method is called from the infer_stage.py script.
        """
        if isinstance(cls, LightningModule):
            graph_constructor = cls.load_from_checkpoint(os.path.join(config["input_dir"], "checkpoints", "last.ckpt"))
            graph_constructor.hparams.update(config) # Update the configs used in training with those to be used in inference
        else:
            graph_constructor = cls(config)
    
        graph_constructor.setup(stage="predict")

        for data_name in ["trainset", "valset", "testset"]:
            if hasattr(graph_constructor, data_name):
                graph_constructor.build_graphs(dataset = getattr(graph_constructor, data_name), data_name = data_name)

    def build_graphs(self, dataset, data_name):
        """
        Build the graphs using the trained model. This is the only function that needs to be overwritten by the child class.
        """
        pass
    
    @classmethod
    def evaluate(cls, config):
        """ 
        The gateway for the evaluation stage. This class method is called from the eval_stage.py script.
        """
        
        # Load data from testset directory
        graph_constructor = cls(config)
        graph_constructor.use_csv = False
        graph_constructor.setup(stage="test")

        all_plots = config["plots"]

        # TODO: Handle the list of plots properly
        for plot_function, plot_config in all_plots.items():
            if hasattr(graph_constructor, plot_function):
                getattr(graph_constructor, plot_function)(plot_config, config)
            else:
                print(f"Plot {plot} not implemented")

    def graph_construction_efficiency(self, plot_config, config):
        """
        Plot the graph construction efficiency vs. pT of the edge.
        """
        all_y_truth, all_pt = [], []

        for event in tqdm(self.testset):
            if "target_tracks" in config:
                self.apply_target_conditions(event, config["target_tracks"])
            else:
                event.target_mask = torch.ones(event.truth_map.shape[0], dtype = torch.bool)

            all_y_truth.append(event.truth_map[event.target_mask] >= 0)
            all_pt.append(event.pt[event.target_mask])

        #  TODO: Handle different pT units!
        all_pt = torch.cat(all_pt).cpu().numpy()
        all_y_truth = torch.cat(all_y_truth).cpu().numpy()

        # Get the edgewise efficiency
        # Build a histogram of true pTs, and a histogram of true-positive pTs
        pt_min, pt_max = 1, 50
        if "pt_units" in plot_config and plot_config["pt_units"] == "MeV":
            pt_min, pt_max = pt_min * 1000, pt_max * 1000
        pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)

        true_pt_hist, _ = np.histogram(all_pt, bins = pt_bins)
        true_pos_pt_hist, _ = np.histogram(all_pt[all_y_truth], bins = pt_bins)

        # Divide the two histograms to get the edgewise efficiency
        eff, err = get_ratio(true_pos_pt_hist,  true_pt_hist)
        xvals = (pt_bins[1:] + pt_bins[:-1]) / 2
        xerrs = (pt_bins[1:] - pt_bins[:-1]) / 2

        # Plot the edgewise efficiency
        pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(xvals, eff, xerr=xerrs, yerr=err, fmt='o', color='black', label='Efficiency')
        ax.set_xlabel(f'$p_T [{pt_units}]$', ha='right', x=0.95, fontsize=14)
        ax.set_ylabel(plot_config["title"], ha='right', y=0.95, fontsize=14)
        ax.set_xscale('log')

        # Save the plot
        atlasify(atlas="Internal", 
            subtext=r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
            r"$p_T > 1$GeV, $|\eta < 4$" + "\n"
            r"Mean graph size: " + f"{np.mean([event.edge_index.shape[1] for event in self.testset]):.2f}")
        fig.savefig(os.path.join(config["stage_dir"], "edgewise_efficiency.png"))

    def apply_target_conditions(self, event, target_tracks):
        """
        Apply the target conditions to the event. This is used for the evaluation stage.
        Target_tracks is a list of dictionaries, each of which contains the conditions to be applied to the event.
        """
        passing_tracks = torch.ones(event.truth_map.shape[0], dtype = torch.bool)

        for key, values in target_tracks.items():
            if isinstance(values, list):
                passing_tracks = passing_tracks * (values[0] <= event[key].float()) * (event[key].float() <= values[1])
            else:
                passing_tracks = passing_tracks * (event[key] == values)

        event.target_mask = passing_tracks

class EventDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(self, input_dir, data_name, num_events, use_pyg=False, use_csv=False, preload=False, hparams=None, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        
        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.loaded = False
        self.use_pyg = use_pyg
        self.use_csv = use_csv
        
        self.evt_ids = self.find_evt_ids()
        
    def len(self):
        return len(self.evt_ids)

    def get(self, idx):
        """
        Handles the iteration through the dataset. Depending on how dataset is configured for PyG and CSV file loading,
        will return either PyG events (automatically batched by PyG), CSV files (not batched/concatenated), or both (not batched/concatenated).
        Assumes files are saved correctly from EventReader as: event000...-truth.csv, -particles.csv and -graph.pyg
        """        

        graph = None
        particles = None
        hits = None

        if self.use_pyg:
            # Check if file event{self.evt_ids[idx]}-graph.pyg exists
            if os.path.exists(os.path.join(self.input_dir, self.data_name, f"event{self.evt_ids[idx]}-graph.pyg")):
                graph = torch.load(os.path.join(self.input_dir, self.data_name, f"event{self.evt_ids[idx]}-graph.pyg"))
            else:
                graph = torch.load(os.path.join(self.input_dir, self.data_name, f"event{self.evt_ids[idx]}.pyg"))
            if not self.use_csv:
                return graph

        if self.use_csv:
            particles = pd.read_csv(os.path.join(self.input_dir, self.data_name, f"event{self.evt_ids[idx]}-particles.csv"))
            hits = pd.read_csv(os.path.join(self.input_dir, self.data_name, f"event{self.evt_ids[idx]}-truth.csv"))
            if not self.use_pyg:
                return particles, hits

        return graph, particles, hits

    def find_evt_ids(self):
        """
        Returns a list of all event ids, which are the numbers in filenames that end in .csv and .pyg
        """

        all_files = os.listdir(os.path.join(self.input_dir, self.data_name))
        all_files = [f for f in all_files if f.endswith(".csv") or f.endswith(".pyg")]
        all_event_ids = sorted(list({re.findall("[0-9]+", file)[-1] for file in all_files}))
        if self.num_events is not None:
            all_event_ids = all_event_ids[:self.num_events]

        # Check that events are present for the requested filetypes
        if self.use_csv:
            all_event_ids = [evt_id for evt_id in all_event_ids if (f"event{evt_id}-truth.csv" in all_files) and (f"event{evt_id}-particles.csv" in all_files)]
        if self.use_pyg:
            all_event_ids = [evt_id for evt_id in all_event_ids if f"event{evt_id}-graph.pyg" or f"event{evt_id}.pyg" in all_files]

        return all_event_ids

    def construct_truth(self, event):
        """
        Construct the truth and weighting labels for the model training.
        """
        # TODO: Implement!
        pass