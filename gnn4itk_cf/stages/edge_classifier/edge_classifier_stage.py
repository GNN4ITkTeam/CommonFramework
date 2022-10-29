import sys
sys.path.append("../")
import os
import warnings

from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ..utils import str_to_class, load_datafiles_in_dir, run_data_tests, construct_event_truth

class EdgeClassifierStage:
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Assign hyperparameters
        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = LargeGraphDataset

    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        self.load_data()
        self.test_data()

    def load_data(self):
        """
        Load in the data for training, validation and testing.
        """

        for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
            if data_num > 0:
                dataset = self.dataset_class(self.hparams["input_dir"], data_name, data_num, self.hparams)
                setattr(self, data_name, dataset)

    def test_data(self):
        """
        Test the data to ensure it is of the right format and loaded correctly.
        """
        required_features = ["x", "edge_index", "track_edges"]
        optional_features = ["pid", "n_hits", "primary", "pdg_id", "ghost", "shared", "module_id", "region_id", "hit_id"]

        run_data_tests(self.trainset, self.valset, self.testset, required_features, optional_features)

    @classmethod
    def infer(cls, config):
        """ 
        The gateway for the inference stage. This class method is scalled from the infer_stage.py script.
        """
        if isinstance(cls, LightningModule):
            edge_classifier = cls.load_from_checkpoint(os.path.join(config["input_dir"], "checkpoints", "last.ckpt")).to(device)
            edge_classifier.hparams.update(config) # Update the configs used in training with those to be used in inference
        else:
            edge_classifier = cls(config)

        edge_classifier.setup(stage="predict")

        for data_name in ["trainset", "valset", "testset"]:
            if hasattr(edge_classifier, data_name):
                edge_classifier.score_edges(dataset = getattr(edge_classifier, data_name), data_name = data_name)

    def score_edges(self, dataset, data_name):
        """
        Score the edges in the graph using the a model. Can be overwritten by the child class.
        """
        for batch in dataset:
            batch = batch.to(device)
            output = self(batch)
            batch, output = batch.cpu(), output.cpu()
            self.save_edge_scores(batch, output, data_name)

    def save_edge_scores(self, graph, scores, data_name):

        graph.scores = scores
        torch.save(graph, os.path.join(self.hparams["stage_dir"], data_name, f"event{graph.event_id}.pyg"))

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(
                self.trainset, batch_size=1, num_workers=4
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

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):
        
        self.process_edges(batch)  

        output = self(batch)

        weights = self.process_weights(batch, output)
        loss = self.loss_function(output, batch, weights)     

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def process_edges(self, batch):
        """
        TODO
        """    
        pass

    def process_weights(self, batch, output):
        """
        TODO
        """
        return None

    def loss_function(self, output, batch, weights):
        """
        Handles the weights, which is ToBeImplemented
        """
        return F.binary_cross_entropy(output, batch.y)

    def log_metrics(self, output, sample_indices, batch, loss, log):

        preds = torch.sigmoid(output) > self.hparams["edge_cut"]

        # Positives
        edge_positive = preds.sum().float()

        # Signal true & signal tp
        sig_truth = batch[self.hparams["truth_key"]][sample_indices]
        sig_true = sig_truth.sum().float()
        sig_true_positive = (sig_truth.bool() & preds).sum().float()
        sig_auc = roc_auc_score(
            sig_truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach()
        )

        # Total true & total tp
        tot_truth = (batch.y_pid.bool() | batch.y.bool())[sample_indices]
        tot_true = tot_truth.sum().float()
        tot_true_positive = (tot_truth.bool() & preds).sum().float()
        tot_auc = roc_auc_score(
            tot_truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach()
        )

        # Eff, pur, auc
        sig_eff = sig_true_positive / sig_true
        sig_pur = sig_true_positive / edge_positive
        tot_eff = tot_true_positive / tot_true
        tot_pur = tot_true_positive / edge_positive

        # Combined metrics
        double_auc = sig_auc * tot_auc
        custom_f1 = 2 * sig_eff * tot_pur / (sig_eff + tot_pur)
        sig_fake_ratio = sig_true_positive / (edge_positive - tot_true_positive)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {
                    "val_loss": loss,
                    "current_lr": current_lr,
                    "sig_eff": sig_eff,
                    "sig_pur": sig_pur,
                    "sig_auc": sig_auc,
                    "tot_eff": tot_eff,
                    "tot_pur": tot_pur,
                    "tot_auc": tot_auc,
                    "double_auc": double_auc,
                    "custom_f1": custom_f1,
                    "sig_fake_ratio": sig_fake_ratio,
                },
                sync_dist=True,
            )

        return preds

    def shared_evaluation(self, batch, batch_idx, log=True):

        truth = batch[self.hparams["truth_key"]]
        
        if ("train_purity" in self.hparams.keys()) and (
            self.hparams["train_purity"] > 0
        ):
            edge_sample, truth_sample, sample_indices = purity_sample(
                truth, batch.edge_index, self.hparams["train_purity"]
            )
        else:
            edge_sample, truth_sample, sample_indices = batch.edge_index, truth, torch.arange(batch.edge_index.shape[1])
            
        edge_sample, truth_sample, sample_indices = self.handle_directed(batch, edge_sample, truth_sample, sample_indices)

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~truth_sample.bool()).sum() / truth_sample.sum())
        )
        
        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample).squeeze()

        positive_loss = F.binary_cross_entropy_with_logits(
            output[truth_sample], torch.ones(truth_sample.sum()).to(self.device)
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            output[~truth_sample], torch.zeros((~truth_sample).sum()).to(self.device)
        )

        loss = positive_loss*weight + negative_loss

        preds = self.log_metrics(output, sample_indices, batch, loss, log)

        return {"loss": loss, "preds": preds, "score": torch.sigmoid(output)}

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx)

        return outputs["loss"]

    def test_step(self, batch, batch_idx):

        return self.shared_evaluation(batch, batch_idx, log=False)

class LargeGraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(self, input_dir, data_name, num_events, hparams=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        
        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        
        self.input_paths = load_datafiles_in_dir(self.input_dir, self.data_name, self.num_events)
        
    def len(self):
        return len(self.input_paths)

    def get(self, idx):

        event = torch.load(self.input_paths[idx], map_location=torch.device("cpu"))
        event = self.construct_truth(event)
                
        return event

    def construct_truth(self, event):
        """
        Construct the truth and weighting labels for the model training.
        """
        
        return construct_event_truth(event, self.hparams)