import os
import warnings
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

# import roc auc
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from tqdm import tqdm
from atlasify import atlasify
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from gnn4itk_cf.utils import load_datafiles_in_dir, run_data_tests, handle_weighting, handle_hard_cuts, remap_from_mask, get_ratio, get_optimizers

class EdgeClassifierStage(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        self.save_hyperparameters(hparams)

        # Assign hyperparameters
        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = GraphDataset
        
    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        if stage in ["fit", "predict"]:
            self.load_data(stage, self.hparams["input_dir"])
            self.test_data(stage)
        elif stage == "test":
            self.load_data(stage, self.hparams["stage_dir"])

        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")
        except Exception:
            warnings.warn("Failed to define figures of merit, due to logger unavailable")
            
    def load_data(self, stage, input_dir):
        """
        Load in the data for training, validation and testing.
        """

        # if stage == "fit":
        for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
            if data_num > 0:
                dataset = self.dataset_class(input_dir, data_name, data_num, stage, self.hparams)
                setattr(self, data_name, dataset)

    def test_data(self, stage):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = ["x", "edge_index", "track_edges", "truth_map", "y"]
        optional_features = ["particle_id", "nhits", "primary", "pdgId", "ghost", "shared", "module_id", "region", "hit_id", "pt"]

        run_data_tests([self.trainset, self.valset, self.testset], required_features, optional_features)

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(
                self.trainset, batch_size=1, num_workers=16
            )  
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(
                self.valset, batch_size=1, num_workers=0
            )  
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(
                self.testset, batch_size=1, num_workers=0
            )  
        else:
            return None

    def predict_dataloader(self):

        datasets = []
        for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
            if data_num > 0:
                dataset = self.dataset_class(self.hparams["input_dir"], data_name, data_num, "predict", self.hparams)
                datasets.append(dataset) 
        return datasets

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizers(self.parameters(), self.hparams)
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):
        
        output = self(batch)
        loss = self.loss_function(output, batch)     

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def loss_function(self, output, batch):
        """
        Applies the loss function to the output of the model and the truth labels.
        To balance the positive and negative contribution, simply take the means of each separately.
        Any further fine tuning to the balance of true target, true background and fake can be handled 
        with the `weighting` config option.
        """

        assert hasattr(batch, "y"), "The batch does not have a truth label. Please ensure the batch has a `y` attribute."
        assert hasattr(batch, "weights"), "The batch does not have a weighting label. Please ensure the batch weighting is handled in preprocessing."
        
        negative_mask = ((batch.y == 0) & (batch.weights != 0)) | (batch.weights < 0) 
        
        negative_loss = F.binary_cross_entropy_with_logits(
            output[negative_mask], torch.zeros_like(output[negative_mask]), weight=batch.weights[negative_mask].abs()
        )

        positive_mask = (batch.y == 1) & (batch.weights > 0)
        positive_loss = F.binary_cross_entropy_with_logits(
            output[positive_mask], torch.ones_like(output[positive_mask]), weight=batch.weights[positive_mask].abs()
        )

        return positive_loss + negative_loss

    def shared_evaluation(self, batch, batch_idx):
        
        output = self(batch)
        loss = self.loss_function(output, batch)   

        all_truth = batch.y.bool()
        target_truth = (batch.weights > 0) & all_truth
        
        return {"loss": loss, "all_truth": all_truth, "target_truth": target_truth, "output": output}

    def validation_step(self, batch, batch_idx):

        return self.shared_evaluation(batch, batch_idx)

    def test_step(self, batch, batch_idx):

        return self.shared_evaluation(batch, batch_idx)

    def validation_epoch_end(self, outputs):

        if len(outputs) > 0:
            loss = torch.stack([x["loss"] for x in outputs]).mean()
            output = torch.cat([x["output"] for x in outputs])
            all_truth = torch.cat([x["all_truth"] for x in outputs])
            target_truth = torch.cat([x["target_truth"] for x in outputs])

            self.log_metrics(output, all_truth, target_truth, loss)

    def log_metrics(self, output, all_truth, target_truth, loss):

        preds = torch.sigmoid(output) > self.hparams["edge_cut"]

        # Positives
        edge_positive = preds.sum().float()

        # Signal true & signal tp
        target_true = target_truth.sum().float()
        target_true_positive = (target_truth.bool() & preds).sum().float()
        all_true_positive = (all_truth.bool() & preds).sum().float()
        target_auc = roc_auc_score(
            target_truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach()
        )

        # Eff, pur, auc
        target_eff = target_true_positive / target_true
        target_pur = target_true_positive / edge_positive
        total_pur = all_true_positive / edge_positive
        current_lr = self.optimizers().param_groups[0]["lr"]

        self.log_dict(
            {
                "val_loss": loss,
                "current_lr": current_lr,
                "eff": target_eff,
                "target_pur": target_pur,
                "total_pur": total_pur,
                "auc": target_auc,
            },  # type: ignore
            sync_dist=True,
        )

        return preds

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        # warm up lr
        if (self.hparams["warmup"] is not None) and (self.trainer.current_epoch < self.hparams["warmup"]):
            lr_scale = min(1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This function handles the prediction of each graph. It is called in the `infer.py` script.
        It can be overwritted in your custom stage, but it should implement three simple steps:
        1. Run an edge-scoring model on the input graph
        2. Add the scored edges to the graph, as `scores` attribute
        3. Append the stage config to the `config` attribute of the graph
        """
            
        output = self(batch)
        dataset = self.predict_dataloader()[dataloader_idx]
        self.save_edge_scores(batch, output, dataset)

    def save_edge_scores(self, event, output, dataset):

        event.scores = torch.sigmoid(output)
        if "undirected" in self.hparams and self.hparams["undirected"]:
            self.remove_duplicated_edges(event)

        dataset.unscale_features(event)

        event.config.append(self.hparams)

        datatype = dataset.data_name    
        os.makedirs(os.path.join(self.hparams["stage_dir"], datatype), exist_ok=True)
        torch.save(event.cpu(), os.path.join(self.hparams["stage_dir"], datatype, f"event{event.event_id}.pyg"))

    def remove_duplicate_edges(self, event):
        """
        Remove duplicate edges, since we only need an undirected graph. Randomly flip the remaining edges to remove
        any training biases downstream
        """

        event.edge_index[:, event.edge_index[0] > event.edge_index[1]] = event.edge_index[:, event.edge_index[0] > event.edge_index[1]].flip(0)
        event.edge_index, edge_inverse = event.edge_index.unique(return_inverse=True, dim=-1)
        event.y = torch.zeros_like(event.edge_index[0], dtype=event.y.dtype).scatter(0, edge_inverse, event.y)
        event.scores = torch.zeros_like(event.edge_index[0], dtype=event.scores.dtype).scatter(0, edge_inverse, event.scores)
        event.truth_map[event.truth_map >= 0] = edge_inverse[event.truth_map[event.truth_map >= 0]]
        event.truth_map = event.truth_map[:event.track_edges.shape[1]]

        random_flip = torch.randint(2, (event.edge_index.shape[1],), dtype=torch.bool)
        event.edge_index[:, random_flip] = event.edge_index[:, random_flip].flip(0)

    @classmethod
    def evaluate(cls, config):
        """ 
        The gateway for the evaluation stage. This class method is called from the eval_stage.py script.
        """

        # Load data from testset directory
        graph_constructor = cls(config)
        graph_constructor.setup(stage="test")

        all_plots = config["plots"]
        
        # TODO: Handle the list of plots properly
        for plot_function, plot_config in all_plots.items():
            if hasattr(graph_constructor, plot_function):
                getattr(graph_constructor, plot_function)(plot_config, config)
            else:
                print(f"Plot {plot_function} not implemented")

    def graph_scoring_efficiency(self, plot_config, config):
        """
        Plot the graph construction efficiency vs. pT of the edge.
        """
        all_y_truth, all_pt  = [], []

        for event in tqdm(self.testset):

            # Need to apply score cut and remap the truth_map 
            if "score_cut" in config:
                self.apply_score_cut(event, config["score_cut"])
            if "target_tracks" in config:
                self.apply_target_conditions(event, config["target_tracks"])
            else:
                event.target_mask = torch.ones(event.truth_map.shape[0], dtype = torch.bool)
            all_y_truth.append(event.truth_map[event.target_mask] >= 0)
            all_pt.append(event.pt[event.target_mask])

        all_pt = torch.cat(all_pt).cpu().numpy()
        all_y_truth = torch.cat(all_y_truth).cpu().numpy()

        # Get the edgewise efficiency
        # Build a histogram of true pTs, and a histogram of true-positive pTs
        pt_min, pt_max = 1, 50
        if "pt_units" in plot_config and plot_config["pt_units"] == "MeV":
            pt_min, pt_max = pt_min * 1000, pt_max * 1000
        pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)

        true_pt_hist, true_bins = np.histogram(all_pt, bins = pt_bins)
        true_pos_pt_hist, _ = np.histogram(all_pt[all_y_truth], bins = pt_bins)

        # Divide the two histograms to get the edgewise efficiency
        eff, err = get_ratio(true_pos_pt_hist,  true_pt_hist)
        xvals = (true_bins[1:] + true_bins[:-1]) / 2
        xerrs = (true_bins[1:] - true_bins[:-1]) / 2

        # Plot the edgewise efficiency
        pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(xvals, eff, xerr=xerrs, yerr=err, fmt='o', color='black', label='Efficiency')
        ax.set_xlabel(f'$p_T [{pt_units}]$', ha='right', x=0.95, fontsize=14)
        ax.set_ylabel(plot_config["title"], ha='right', y=0.95, fontsize=14)
        ax.set_xscale('log')

        # Save the plot
        atlasify("Internal", 
         r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
         r"$p_T > 1$GeV, $|\eta < 4$")
        fig.savefig(os.path.join(config["stage_dir"], "edgewise_efficiency.png"))

    def graph_roc_curve(self, plot_config, config):
        """
        Plot the ROC curve for the graph construction efficiency.
        """
        all_y_truth, all_scores  = [], []

        for event in tqdm(self.testset):
                
            # Need to apply score cut and remap the truth_map 
            if "weights" in event.keys:
                target_y = event.weights.bool() & event.y.bool()
            else:
                target_y = event.y.bool()

            all_y_truth.append(target_y)
            all_scores.append(event.scores)

        all_scores = torch.cat(all_scores).cpu().numpy()
        all_y_truth = torch.cat(all_y_truth).cpu().numpy()

        # Get the ROC curve
        fpr, tpr, _ = roc_curve(all_y_truth, all_scores)
        auc_score = auc(fpr, tpr)

        # Plot the ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='black', label='ROC curve')
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random classifier')
        ax.set_xlabel('False Positive Rate', ha='right', x=0.95, fontsize=14)
        ax.set_ylabel('True Positive Rate', ha='right', y=0.95, fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='lower right', fontsize=14)
        ax.text(0.95, 0.20, f"AUC: {auc_score:.3f}", ha='right', va='bottom', transform=ax.transAxes, fontsize=14)

        # Save the plot
        atlasify("Internal", 
         f"{plot_config['title']} \n"
         r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
         r"$p_T > 1$GeV, $|\eta < 4$")
        fig.savefig(os.path.join(config["stage_dir"], "roc_curve.png"))

    def apply_score_cut(self, event, score_cut):
        """
        Apply a score cut to the event. This is used for the evaluation stage.
        """
        passing_edges_mask = event.scores >= score_cut
        remap_from_mask(event, passing_edges_mask)

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

class GraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(self, input_dir, data_name = None, num_events = None, stage="fit", hparams=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        
        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.stage = stage
        
        self.input_paths = load_datafiles_in_dir(self.input_dir, self.data_name, self.num_events)
        self.input_paths.sort() # We sort here for reproducibility
        
    def len(self):
        return len(self.input_paths)

    def get(self, idx):

        event_path = self.input_paths[idx]
        event = torch.load(event_path, map_location=torch.device("cpu"))
        self.preprocess_event(event)

        # return (event, event_path) if self.stage == "predict" else event
        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """
        
        self.apply_hard_cuts(event)
        self.construct_weighting(event)
        self.handle_edge_list(event)
        self.scale_features(event)
        
    def apply_hard_cuts(self, event):
        """
        Apply hard cuts to the event. This is implemented by 
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """
        
        if self.hparams is not None and "hard_cuts" in self.hparams.keys() and self.hparams["hard_cuts"]:
            assert isinstance(self.hparams["hard_cuts"], dict), "Hard cuts must be a dictionary"
            handle_hard_cuts(event, self.hparams["hard_cuts"])

    def construct_weighting(self, event):
        """
        Construct the weighting for the event
        """
        
        assert event.y.shape[0] == event.edge_index.shape[1], f"Input graph has {event.edge_index.shape[1]} edges, but {event.y.shape[0]} truth labels"

        if self.hparams is not None and "weighting" in self.hparams.keys():
            assert isinstance(self.hparams["weighting"], list) & isinstance(self.hparams["weighting"][0], dict), "Weighting must be a list of dictionaries"
            handle_weighting(event, self.hparams["weighting"])
        else:
            event.weights = torch.ones_like(event.y, dtype=torch.float32)
            
    def handle_edge_list(self, event):

        if "input_cut" in self.hparams.keys() and self.hparams["input_cut"] and "scores" in event.keys:
            # Apply a score cut to the event
            self.apply_score_cut(event, self.hparams["input_cut"])

        if "undirected" in self.hparams.keys() and self.hparams["undirected"]:
            # Flip event.edge_index and concat together
            event.edge_index = torch.cat([event.edge_index, event.edge_index.flip(0)], dim=1)
            event.y = torch.cat([event.y, event.y], dim=0)
            event.weights = torch.cat([event.weights, event.weights], dim=0)

            # Remove duplicate edges
            event.edge_index, unique_edge_indices = torch.unique(event.edge_index, dim=1, return_inverse=True)
            event.y = torch.zeros_like(event.edge_index[0], dtype=event.y.dtype).scatter(0, unique_edge_indices, event.y)
            event.weights = torch.zeros_like(event.edge_index[0], dtype=event.weights.dtype).scatter(0, unique_edge_indices, event.weights)

    def scale_features(self, event):
        """
        Handle feature scaling for the event
        """
        
        if self.hparams is not None and "node_scales" in self.hparams.keys() and "node_features" in self.hparams.keys():
            assert isinstance(self.hparams["node_scales"], list), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys, f"Feature {feature} not found in event"
                event[feature] = event[feature] / self.hparams["node_scales"][i]

    def unscale_features(self, event):
        """
        Unscale features when doing prediction
        """
        
        if self.hparams is not None and "node_scales" in self.hparams.keys() and "node_features" in self.hparams.keys():
            assert isinstance(self.hparams["node_scales"], list), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys, f"Feature {feature} not found in event"
                event[feature] = event[feature] * self.hparams["node_scales"][i]

    def apply_score_cut(self, event, score_cut):
        """
        Apply a score cut to the event. This is used for the evaluation stage.
        """
        passing_edges_mask = event.scores >= score_cut
        num_edges = event.edge_index.shape[1]
        for key in event.keys:
            if (event[key].shape[0] == num_edges) or (event["key"].shape[1] == num_edges):
                event[key] = event[key][..., passing_edges_mask]

        remap_from_mask(event, passing_edges_mask)