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

import os
import warnings
from itertools import product
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, HeteroData
from torch_geometric.loader import DataLoader
import random

# import roc auc
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from tqdm import tqdm
from atlasify import atlasify
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from gnn4itk_cf.stages.track_building.utils import rearrange_by_distance
from gnn4itk_cf.utils.mapping_utils import get_directed_prediction
from gnn4itk_cf.utils.plotting_utils import plot_efficiency_rz
from gnn4itk_cf.utils import eval_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from gnn4itk_cf.utils import load_datafiles_in_dir, run_data_tests, handle_weighting, handle_hard_cuts, remap_from_mask, get_ratio, handle_edge_features, get_optimizers, plot_eff_pur_region, get_condition_lambda
from gnn4itk_cf.stages.graph_construction.models.utils import graph_intersection

# TODO: What is this for??
torch.multiprocessing.set_sharing_strategy('file_system')

class EdgeClassifierStage(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        self.to(device)
        self.save_hyperparameters(hparams)

        # Assign hyperparameters
        self.trainset, self.valset, self.testset = None, None, None
        self.hparams['dataset_class'] = "GraphDataset"
        self.dataset_class = eval(self.hparams['dataset_class'])
        
    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """
        preprocess = True
        input_dir = "input_dir"
        if stage in ["fit", "predict"]:
            self.load_data(stage, self.hparams[input_dir], preprocess)
            self.test_data(stage)
        elif stage == "test":
            # during test stage, allow the possibility of 
            if not self.hparams.get("reprocess_classifier"):
                print("Reading data from stage_dir without preprocessing")
                input_dir = "stage_dir"
                preprocess = False
            self.load_data(stage, self.hparams[input_dir], preprocess)

        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")
        except Exception:
            warnings.warn("Failed to define figures of merit, due to logger unavailable")
            
    def load_data(self, stage, input_dir, preprocess=True):
        """
        Load in the data for training, validation and testing.
        """

        # if stage == "fit":
        for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
            if data_num > 0:
                dataset = self.dataset_class(input_dir, data_name, data_num, stage, self.hparams, preprocess=preprocess)
                setattr(self, data_name, dataset)

    def test_data(self, stage):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = ["x", "edge_index", "track_edges", "truth_map", "y"]
        optional_features = ["particle_id", "nhits", "primary", "pdgId", "ghost", "shared", "module_id", "region", "hit_id", "pt"]

        run_data_tests([self.trainset, self.valset, self.testset], required_features, optional_features)

    def train_dataloader(self):
        if self.trainset is None:
            return None
        num_workers = 16 if ("num_workers" not in self.hparams or self.hparams["num_workers"] is None) else self.hparams["num_workers"][0]
        return DataLoader(
            self.trainset, batch_size=1, num_workers=num_workers, shuffle=False
        )

    def val_dataloader(self):
        if self.valset is None:
            return None
        num_workers = 16 if ("num_workers" not in self.hparams or self.hparams["num_workers"] is None) else self.hparams["num_workers"][1]
        return DataLoader(
            self.valset, batch_size=1, num_workers=num_workers
        )

    def test_dataloader(self):
        if self.testset is None:
            return None
        num_workers = 16 if ("num_workers" not in self.hparams or self.hparams["num_workers"] is None) else self.hparams["num_workers"][2]
        return DataLoader(
            self.testset, batch_size=1, num_workers=num_workers
        )

    def predict_dataloader(self):

        self.datasets = []
        dataloaders = []
        for i, (data_name, data_num) in enumerate(zip(["trainset", "valset", "testset"], self.hparams["data_split"])):
            if data_num > 0:
                dataset = self.dataset_class(self.hparams["input_dir"], data_name, data_num, "predict", self.hparams)
                self.datasets.append(dataset)
                num_workers = 16 if ("num_workers" not in self.hparams or self.hparams["num_workers"] is None) else self.hparams["num_workers"][i]
                dataloaders.append(DataLoader(dataset, batch_size=1, num_workers=num_workers))
        return dataloaders

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizers(self.parameters(), self.hparams)
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        output = self(batch)
        loss = self.loss_function(output, batch)

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        return loss

    def on_train_epoch_start(self):
        if self.trainset is not None:
            random.shuffle(self.trainset.input_paths)

    def loss_function(self, output, batch):
        """
        Applies the loss function to the output of the model and the truth labels.
        To balance the positive and negative contribution, simply take the means of each separately.
        Any further fine tuning to the balance of true target, true background and fake can be handled
        with the `weighting` config option.
        """

        assert hasattr(batch, "y"), "The batch does not have a truth label. Please ensure the batch has a `y` attribute."
        assert hasattr(batch, "weights"), "The batch does not have a weighting label. Please ensure the batch weighting is handled in preprocessing."

        if ("loss_balancing" in self.hparams) and self.hparams["loss_balancing"]:
            print("loss_balancing")
            negative_mask = ((batch.y == 0) & (batch.weights != 0)) | (batch.weights < 0)
            negative_loss = F.binary_cross_entropy_with_logits(
                output[negative_mask], torch.zeros_like(output[negative_mask]), weight=batch.weights[negative_mask].abs()
            )
            positive_mask = (batch.y == 1) & (batch.weights > 0)
            positive_loss = F.binary_cross_entropy_with_logits(
                output[positive_mask], torch.ones_like(output[positive_mask]), weight=batch.weights[positive_mask].abs()
            )
            return positive_loss + negative_loss
        else:
            loss = F.binary_cross_entropy_with_logits(
                output, batch.y.float(), weight=batch.weights.abs()
            )
        return loss


    def shared_evaluation(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss_function(output, batch)
        batch.output = output.detach() 

        all_truth = batch.y.bool()
        target_truth = (batch.weights > 0) & all_truth
        
        return {"loss": loss.detach(), "all_truth": all_truth, "target_truth": target_truth, "output": output.detach(), 'batch': batch}

    def validation_step(self, batch, batch_idx):
        
        output_dict = self.shared_evaluation(batch, batch_idx)
        self.log_metrics(output_dict['output'], output_dict['all_truth'], output_dict['target_truth'], output_dict['loss'])
        self.log("val_loss", output_dict['loss'], on_step=False, on_epoch=True, batch_size=1, sync_dist=True)        

    def test_step(self, batch, batch_idx):

        return self.shared_evaluation(batch, batch_idx)

    def log_metrics(self, output, all_truth, target_truth, loss):

        preds = torch.sigmoid(output) > self.hparams["edge_cut"]

        # Positives
        edge_positive = preds.sum().float()

        # Signal true & signal tp
        target_true = target_truth.sum().float()
        target_true_positive = (target_truth.bool() & preds).sum().float()
        all_true_positive = (all_truth.bool() & preds).sum().float()

        fake_positive = edge_positive.item() - all_true_positive.item() 
        # Masked Positives
        target_and_fake_edge_positive = target_true_positive + fake_positive

        # add torch.sigmoid(output).float() to convert to float in case training is done with 16-bit precision
        target_auc = roc_auc_score(
            target_truth.bool().cpu().detach(), torch.sigmoid(output).float().cpu().detach()
        )
        true_and_fake_positive = edge_positive - (preds & (~ target_truth) & all_truth).sum().float()

        # Eff, pur, auc
        target_eff = target_true_positive / target_true
        target_pur = target_true_positive / edge_positive
        target_pur_vs_fake = target_true_positive / target_and_fake_edge_positive
        total_pur = all_true_positive / edge_positive
        purity = target_true_positive / true_and_fake_positive
        current_lr = self.optimizers().param_groups[0]["lr"]

        self.log_dict(
            {
                "current_lr": current_lr,
                "eff": target_eff,
                "target_pur": target_pur,
                "target_pur_vs_fake": target_pur_vs_fake,
                "total_pur": total_pur,
                "pur": purity,
                "auc": target_auc,
            },  # type: ignore
            sync_dist=True,
            batch_size=1, 
            on_epoch=True, 
            on_step=False
        )

        return preds

    def on_train_start(self):
        self.trainer.strategy.optimizers = [self.trainer.lr_scheduler_configs[0].scheduler.optimizer]
    
    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        
        # warm up lr
        if (self.hparams["warmup"] is not None) and (self.trainer.current_epoch < self.hparams["warmup"]):
            lr_scale = min(1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]
        
        # after reaching minimum learning rate, stop LR decay
        for pg in optimizer.param_groups:
            pg['lr'] = max(pg['lr'], self.hparams.get('min_lr', 0.00005))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This function handles the prediction of each graph. It is called in the `infer.py` script.
        It can be overwritted in your custom stage, but it should implement three simple steps:
        1. Run an edge-scoring model on the input graph
        2. Add the scored edges to the graph, as `scores` attribute
        3. Append the stage config to the `config` attribute of the graph
        """

        dataset = self.datasets[dataloader_idx]
        if os.path.exists(os.path.join(self.hparams["stage_dir"], dataset.data_name , f"event{batch.event_id}.pyg")):
            return
        eval_dict = self.shared_evaluation(batch, batch_idx)
        output = eval_dict['output']
        batch = eval_dict['batch']
        self.save_edge_scores(batch, output, dataset)

    def save_edge_scores(self, event, output, dataset):
        event.scores = torch.sigmoid(output)
        event = dataset.unscale_features(event)

        event.config.append(self.hparams)
        event.truth_map = graph_intersection(event.edge_index, event.track_edges, return_y_pred=False, return_y_truth=False, return_truth_to_pred=True)

        datatype = dataset.data_name
        os.makedirs(os.path.join(self.hparams["stage_dir"], datatype), exist_ok=True)
        torch.save(event.cpu(), os.path.join(self.hparams["stage_dir"], datatype, f"event{event.event_id[0]}.pyg"))

    @classmethod
    def evaluate(cls, config, checkpoint=None):
        """ 
        The gateway for the evaluation stage. This class method is called from the eval_stage.py script.
        """

        # Load data from testset directory
        graph_constructor = cls(config)
        if checkpoint is not None:
            print(f'Restoring model from {checkpoint}')
            graph_constructor = cls.load_from_checkpoint(checkpoint, hparams=config).to(device)
        graph_constructor.setup(stage="test")

        all_plots = config["plots"]

        # TODO: Handle the list of plots properly
        for plot_function, plot_config in all_plots.items():
            if hasattr(eval_utils, plot_function):
                getattr(eval_utils, plot_function)(graph_constructor, plot_config, config)
            else:
                print(f"Plot {plot_function} not implemented")

    def apply_score_cut(self, event, score_cut):
        """
        Apply a score cut to the event. This is used for the evaluation stage.
        """
        passing_edges_mask = event.scores >= score_cut
        edge_index = event.edge_index
        if self.hparams['undirected']:
            # treat graph level first
            edge_index = rearrange_by_distance(event, event.edge_index)
            get_directed_prediction(event, passing_edges_mask, edge_index)

            # treat track level, simply drop the later half of all track-level features
            track_edges = rearrange_by_distance(event, event.track_edges)
            num_track_edges = track_edges.shape[1]
            event.track_edges = track_edges.T.view(2, -1, 2)[0].T
            # print(track_edges.T.view(2, -1, 2))
            for key in event.keys:
                if isinstance(event[key], torch.Tensor) and ((event[key].shape[0] == num_track_edges)):
                    event[key] = event[key].view(2, -1)[0]
            # hard code to choose tight passing edge masks as the default edge mask to compute truth map for now for backward compatibility
            event.truth_map_loose = graph_intersection(event.edge_index[:, event.passing_edge_mask_loose], event.track_edges, return_y_pred=False, return_truth_to_pred=True)
            event.truth_map_tight = graph_intersection(event.edge_index[:, event.passing_edge_mask_tight], event.track_edges, return_y_pred=False, return_truth_to_pred=True)
            passing_edges_mask = event.passing_edge_mask_tight
                    
        event.graph_truth_map = graph_intersection(event.edge_index, event.track_edges, return_y_pred=False, return_y_truth=False, return_truth_to_pred=True)
        event.truth_map = graph_intersection(event.edge_index[:, passing_edges_mask], event.track_edges, return_y_pred=False, return_truth_to_pred=True)
        event.pred = passing_edges_mask

    def apply_target_conditions(self, event, target_tracks):
        """
        Apply the target conditions to the event. This is used for the evaluation stage.
        Target_tracks is a list of dictionaries, each of which contains the conditions to be applied to the event.
        """
        passing_tracks = torch.ones(event.truth_map.shape[0], dtype = torch.bool).to(device)

        for condition_key, condition_val in target_tracks.items():
            condition_lambda = get_condition_lambda(condition_key, condition_val)
            passing_tracks = passing_tracks * condition_lambda(event)

        event.target_mask = passing_tracks
    
        
class GraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(self, input_dir, data_name = None, num_events = None, stage="fit", hparams={}, transform=None, pre_transform=None, pre_filter=None, preprocess=True):
        super().__init__(input_dir, transform, pre_transform, pre_filter)

        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.stage = stage
        self.preprocess = preprocess
        
        self.input_paths = load_datafiles_in_dir(self.input_dir, self.data_name, self.num_events)
        self.input_paths.sort()  # We sort here for reproducibility

    def len(self):
        return len(self.input_paths)

    def get(self, idx):

        event_path = self.input_paths[idx]
        event = torch.load(event_path, map_location=torch.device("cpu"))
        if not self.preprocess:
            return event
        event = self.preprocess_event(event)

        # return (event, event_path) if self.stage == "predict" else event
        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """
        # print(event)
        if self.hparams.get('undirected'):
            event = self.to_undirected(event)
        event = self.apply_hard_cuts(event)
        event = self.construct_weighting(event)
        event = self.handle_edge_list(event)
        event = self.scale_features(event)
        if self.hparams.get('edge_features')!=None:
            event = self.add_edge_features(event) # scaling must be done before adding features
        return event
        
    def apply_hard_cuts(self, event):
        """
        Apply hard cuts to the event. This is implemented by
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """

        if self.hparams is not None and "hard_cuts" in self.hparams.keys() and self.hparams["hard_cuts"]:
            assert isinstance(self.hparams["hard_cuts"], dict), "Hard cuts must be a dictionary"
            handle_hard_cuts(event, self.hparams["hard_cuts"])
        
        return event

    def construct_weighting(self, event):
        """
        Construct the weighting for the event
        """

        assert event.y.shape[0] == event.edge_index.shape[1], f"Input graph has {event.edge_index.shape[1]} edges, but {event.y.shape[0]} truth labels"

        if self.hparams is not None and "weighting" in self.hparams.keys():
            assert isinstance(self.hparams["weighting"], list) & isinstance(self.hparams["weighting"][0], dict), "Weighting must be a list of dictionaries"
            event.weights = handle_weighting(event, self.hparams["weighting"])
        else:
            event.weights = torch.ones_like(event.y, dtype=torch.float32)
        
        return event
            
    def handle_edge_list(self, event):

        if "input_cut" in self.hparams.keys() and self.hparams["input_cut"] and "scores" in event.keys:
            # Apply a score cut to the event
            self.apply_score_cut(event, self.hparams["input_cut"])

        # if "undirected" in self.hparams.keys() and self.hparams["undirected"]:
        #     # Flip event.edge_index and concat together
        #     self.to_undirected(event)
        return event
            
    
    def to_undirected(self, event):
        """
        Add the reverse of the edge_index to the event. This then requires all edge features to be duplicated.
        Additionally, the truth map must be duplicated.
        """

        num_edges = event.edge_index.shape[1]
        # Flip event.edge_index and concat together
        event.edge_index = torch.cat([event.edge_index, event.edge_index.flip(0)], dim=1)
        # event.edge_index, unique_edge_indices = torch.unique(event.edge_index, dim=1, return_inverse=True)
        num_track_edges = event.track_edges.shape[1]
        event.track_edges = torch.cat([event.track_edges, event.track_edges.flip(0)], dim=1)

        # Concat all edge-like features together
        for key in event.keys:
            if key=='truth_map': continue
            if isinstance(event[key], torch.Tensor) and ((event[key].shape[0] == num_edges)):
                event[key] = torch.cat([event[key], event[key]], dim=0)
                # event[key] = torch.zeros_like(event.edge_index[0], dtype=event[key].dtype).scatter(0, unique_edge_indices, event[key])
            
            # Concat track-like features for evaluation
            elif isinstance(event[key], torch.Tensor) and ((event[key].shape[0] == num_track_edges)):
                event[key] = torch.cat([event[key], event[key]], dim=0)    

        # handle truth_map separately
        truth_map = event.truth_map.clone()
        truth_map[truth_map >=0 ] = truth_map[truth_map >= 0] + num_edges
        event.truth_map = torch.cat([event.truth_map, truth_map], dim=0)
        return event

    def add_edge_features(self, event):
        if "edge_features" in self.hparams.keys():
            assert isinstance(self.hparams["edge_features"], list), "Edge features must be a list of strings"
            handle_edge_features(event, self.hparams["edge_features"])
        return event

    def scale_features(self, event):
        """
        Handle feature scaling for the event
        """

        if self.hparams is not None and "node_scales" in self.hparams.keys() and "node_features" in self.hparams.keys():
            assert isinstance(self.hparams["node_scales"], list), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys, f"Feature {feature} not found in event"
                event[feature] = event[feature] / self.hparams["node_scales"][i]
        
        return event
 
    def unscale_features(self, event):
        """
        Unscale features when doing prediction
        """

        if self.hparams is not None and "node_scales" in self.hparams.keys() and "node_features" in self.hparams.keys():
            assert isinstance(self.hparams["node_scales"], list), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys, f"Feature {feature} not found in event"
                event[feature] = event[feature] * self.hparams["node_scales"][i]
        return event

    def apply_score_cut(self, event, score_cut):
        """
        Apply a score cut to the event. This is used for the evaluation stage.
        """
        passing_edges_mask = event.scores >= score_cut
        num_edges = event.edge_index.shape[1]
        for key in event.keys:
            if isinstance(event[key], torch.Tensor) and event[key].shape and (event[key].shape[0] == num_edges or event[key].shape[-1] == num_edges):
                event[key] = event[key][..., passing_edges_mask]

        remap_from_mask(event, passing_edges_mask)
        return event
    
    def get_y_node(self, event):
        y_node = torch.zeros(event.z.size(0))
        y_node[event.track_edges.view(-1)] = 1
        event.y_node=y_node
        return event

class HeteroGraphMixin:
    def __init__(self) -> None:
        self.hparams = {}
    
    def get_node_type(self, event):
        assert 'region_ids' in self.hparams.keys() and isinstance(self.hparams['region_ids'], list ), "To create a heterogeneous graph, must define region id"
        region = event.region
        node_type = torch.zeros_like(region, dtype=torch.int64)
        node_type_name = []

        for idx, region_id in enumerate(self.hparams['region_ids']):
            mask = torch.isin(region, torch.tensor(region_id['id']))
            node_type[mask] = idx
            node_type_name.append(region_id['name'])
        
        event.node_type = node_type
        event.node_type_name = node_type_name
        return event

    def get_edge_type(self, event):
        assert 'node_type' in event, "event must have node_type. Run it through self.get_node_type"
        assert 'region_ids' in self.hparams.keys() and isinstance(self.hparams['region_ids'], list ), "To create a heterogeneous graph, must define region id"
        edge_index = event.edge_index
        node_type = event.node_type
        edge_type = torch.zeros_like(edge_index[0], dtype=torch.int64)
        edge_type_name = []

        for id, link in enumerate(product(node_type.unique(), node_type.unique())):
            src, dst = link
            mask = ((node_type[edge_index[0]] == src) & (node_type[edge_index[1]] == dst))
            edge_type[mask] = id
            edge_type_name.append(
                (self.hparams['region_ids'][src]['name'], 'to', self.hparams['region_ids'][dst]['name'])
            )

        event.edge_type = edge_type
        event.edge_type_name = edge_type_name
        return event
    
    def convert_heterogeneous(self, event):
        assert 'node_type' in event and 'edge_type' in event and 'node_type_name' in event and 'edge_type_name' in event, "Must run through both self.get_node_type and self.get_edge_type"
        return event.to_heterogeneous(node_type=event.node_type, edge_type=event.edge_type, node_type_names=event.node_type_name, edge_type_names=event.edge_type_name)
    
    def get_input_data(self, event):
        event.input_node_features = torch.stack([event[feature] for feature in self.hparams["node_features"]], dim=-1).float()
        edge_feature_list = self.hparams.get('edge_features', [])
        if len(edge_feature_list) > 0:
            event.input_edge_features = torch.stack([event[feature] for feature in edge_feature_list], dim=-1).float()
        return event

class HeteroGraphDataset(GraphDataset, HeteroGraphMixin):
    def __init__(self, input_dir, data_name=None, num_events=None, stage="fit", hparams={}, transform=None, pre_transform=None, pre_filter=None, preprocess=True):
        super().__init__(input_dir, data_name, num_events, stage, hparams, transform, pre_transform, pre_filter, preprocess)
    
    def preprocess_event(self, event):
        event = super().preprocess_event(event)
        event = Data(**event.to_dict())
        event = self.get_input_data(event)
        event = self.get_node_type(event)
        event = self.get_edge_type(event)
        event = self.convert_heterogeneous(event)
        return event
         
class DirectedHeteroGraphDataset(GraphDataset, HeteroGraphMixin):

    def __init__(self, input_dir, data_name=None, num_events=None, stage="fit", hparams={}, transform=None, pre_transform=None, pre_filter=None, preprocess=True):
        super().__init__(input_dir, data_name, num_events, stage, hparams, transform, pre_transform, pre_filter, preprocess)

    def handle_direction(self, event):
        # get the distance squared
        r2 = event.r**2 + event.z**2
        
        inward_edge_mask = r2[event.edge_index[0]] > r2[event.edge_index[1]]
        inward_track_edge_mask = r2[event.track_edges[0]] > r2[event.track_edges[1]]
            
        event.edge_index[:, inward_edge_mask] = event.edge_index[:, inward_edge_mask].flip(0)
        event.track_edges[:, inward_track_edge_mask] = event.track_edges[:, inward_track_edge_mask].flip(0)
        
        return event

    def preprocess_event(self, event):
        self.hparams['undirected']=False
        event = super().preprocess_event(event)
        event = Data(**event.to_dict())
        event = self.handle_direction(event)
        event = self.get_input_data(event)
        event = self.get_node_type(event)
        event = self.get_edge_type(event)
        event = self.convert_heterogeneous(event)
        return event
        

class HeteroGraphDatasetWithNode(GraphDataset, HeteroGraphMixin):
    def __init__(self, input_dir, data_name=None, num_events=None, stage="fit", hparams={}, transform=None, pre_transform=None, pre_filter=None, preprocess=True):
        super().__init__(input_dir, data_name, num_events, stage, hparams, transform, pre_transform, pre_filter, preprocess)

    def preprocess_event(self, event):
        event = super().preprocess_event(event)
        event = Data(**event.to_dict())
        event = self.get_y_node(event)
        event = self.get_input_data(event)
        event = self.get_node_type(event)
        event = self.get_edge_type(event)
        event = self.convert_heterogeneous(event)
        return event
