import warnings
import torch
import copy
import os
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add, scatter_mean, scatter_max

from gnn4itk_cf.utils import make_mlp
from ..edge_classifier_stage import EdgeClassifierStage


class Filter(EdgeClassifierStage):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        self.save_hyperparameters(hparams)

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )

        # Setup input network
        # Construct the MLP architecture
        self.net = make_mlp(
            len(hparams["node_features"])*2,
            [hparams["hidden"] // (2**i) for i in range(hparams["nb_layer"])] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def forward(self, batch):

        x = torch.stack([batch[feature] for feature in self.hparams["node_features"]], dim=-1).float()
        output = self.net(torch.cat([x[batch.edge_index[0]], x[batch.edge_index[1]]], dim=-1))
        return output.squeeze(-1)

    def training_step(self, batch, batch_idx):
        
        if self.hparams["ratio"] not in [0, None]:
            with torch.no_grad():
                no_grad_output = self.memory_robust_eval(batch)
                batch = self.subsample(batch, torch.sigmoid(no_grad_output), self.hparams["ratio"])

        output = self(batch)
        loss = self.loss_function(output, batch)     

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def shared_evaluation(self, batch, batch_idx):
        
        output = self.memory_robust_eval(batch)
        loss = self.loss_function(output, batch)   

        all_truth = batch.y.bool()
        target_truth = (batch.weights > 0) & all_truth
        
        return {"loss": loss, "all_truth": all_truth, "target_truth": target_truth, "output": output}

    def subsample(self, batch, scores, ratio):
        """
        Samples all the true signal edges, and a number of fake edges equal to the number of true signal edges times the ratio.
        Then combines those edges and shuffles them.
        """
        sample_signal_true = torch.where(batch.y.bool() & (batch.weights>0))[0]
        num_signal_true = sample_signal_true.shape[0]
        sample_hard_negatives, sample_easy_negatives = self.get_negatives(batch, scores, num_signal_true, ratio)        

        sample_combined = torch.cat([sample_signal_true, sample_hard_negatives, sample_easy_negatives])
        sample_combined = sample_combined[torch.randperm(sample_combined.shape[0])]
        batch.edge_index = batch.edge_index[:, sample_combined]
        batch.y = batch.y[sample_combined]
        batch.weights = batch.weights[sample_combined]
        
        return batch

    def get_negatives(self, batch, scores, num_true, ratio):
        """
        Samples a number of 'hard' and 'easy' negatives, where hard negatives are those with a score above the edge_cut, and easy negatives are those with a score below the edge_cut.
        The number of hard and easy negatives is equal to the number of true signal edges times the ratio.
        """
        negative_mask = ((batch.y == 0) & (batch.weights != 0)) | (batch.weights < 0)
        sample_negatives = torch.where( negative_mask )[0]
        sample_hard_negatives = torch.where(negative_mask)[0][scores[negative_mask] > self.hparams["edge_cut"]]
        sample_easy_negatives = torch.where(negative_mask)[0][scores[negative_mask] <= self.hparams["edge_cut"]]

        # Handle where there are no hard negatives
        if sample_hard_negatives.shape[0] == 0:
            sample_hard_negatives = sample_negatives[torch.randint(sample_negatives.shape[0], (num_true*ratio,))]
        else:
            sample_hard_negatives = sample_hard_negatives[torch.randint(sample_hard_negatives.shape[0], (num_true*ratio,))]
        # Handle where there are no easy negatives
        if sample_easy_negatives.shape[0] == 0:
            sample_easy_negatives = sample_negatives[torch.randint(sample_negatives.shape[0], (num_true*ratio,))]
        else:
            sample_easy_negatives = sample_easy_negatives[torch.randint(sample_easy_negatives.shape[0], (num_true*ratio,))]

        return sample_hard_negatives, sample_easy_negatives

    def memory_robust_eval_v2(self, batch):
        """
        A version of the memory_robust_eval that isn't recurrent - uses a simple loop to avoid memory problems
        """

        # Evaluate and combine the two smaller batches, by storing edges temporarily
        all_edges = batch.edge_index
        evals = []
        attempts = 0

        while not evals:
            chunk_size = all_edges.shape[1] // 2**attempts
            attempts += 1

            try:
                for i in range(0, all_edges.shape[1], chunk_size):
                    batch.edge_index = all_edges[:, i:i+chunk_size]
                    evals.append(self(batch))
            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise e
                warnings.warn("Splitting batch due to memory error")
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
        # Reset the batch edges
        batch.edge_index = all_edges

        return torch.cat(evals)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):            
        dataset = self.datasets[dataloader_idx]

        if os.path.exists(os.path.join(self.hparams["stage_dir"], dataset.data_name , f"event{batch.event_id}.pyg")):
            return
        
        output = self.memory_robust_eval_v2(batch)
        
        
        self.save_edge_scores(batch, output, dataset)