# system imports
import sys

# 3rd party imports
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
import numpy as np
from torch.utils.data import random_split
import torch.nn as nn
from torch_scatter import scatter_min
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import pandas as pd
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
sys.path.append("../..")
from ..track_building_stage import TrackBuildingStage
from ..utils import evaluate_labelled_graph, FRNN_graph

class BipartiteClassificationBase(LightningModule, TrackBuildingStage):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module
        """
        LightningModule.__init__(self)
        TrackBuildingStage.__init__(self)
        self.save_hyperparameters(hparams)
        TrackBuildingStage.setup(self, stage = "hgnn")
        
    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=16, shuffle = True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=16)
        else:
            return None

    def test_dataloader(self, cheat = False):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=16)
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
                    gamma=self.hparams["factor"]
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler
    
    @property
    def loss_schedule(self):
        if hasattr(self.trainer, "current_epoch"):
            loss_schedule = 1 - np.sin(self.trainer.current_epoch/2/self.hparams["emb_epoch"]*np.pi) if self.trainer.current_epoch < self.hparams["emb_epoch"] else 0
        else:
            loss_schedule = 0
        return loss_schedule

    @property
    def logit_schedule(self):
        if hasattr(self.trainer, "current_epoch"):
            logit_schedule = 1 - np.sin(self.trainer.current_epoch/2/self.hparams["logit_epoch"]*np.pi) if self.trainer.current_epoch < self.hparams["logit_epoch"] else 0
        else:
            logit_schedule = 0
        return logit_schedule
    
    def get_emb_weight(self, batch, graph, y):
        """
        Calculate weights and balancing positive and negative samples
        Per edge weight is defined as the sum of pt weights of the two ends
        """
        weights = batch.weights.clone()
        weights[y] = weights[y]/(2 * weights[y].sum() + 1e-6)
        weights[~y] = weights[~y]/(2 * weights[~y].sum() + 1e-6)

        return weights 
    
    def get_asgmt_weight(self, batch, bipartite_graph, row_match, col_match, y):
        """
        Calculate weights and balancing positive and negative samples
        Assignment weight is defined as the maximum weight of the supernode's particle and the hit
        """
        supernode_weights = torch.zeros(bipartite_graph[1].max() + 1, device = self.device).float()
        supernode_weights[col_match] = batch.particle_weights[row_match]
        
        weights = torch.maximum(batch.hit_weights[bipartite_graph[0]], supernode_weights[bipartite_graph[1]])
        
        weights[y] = weights[y]/(2 * weights[y].sum() + 1e-6)
        weights[~y] = weights[~y]/(2 * weights[~y].sum() + 1e-6)
        
        return weights
    
    def get_hinge_distance(self, batch, embeddings, graph, y):
        """
        Calculate hinge and Euclidean distance, 1e-12 is added to avoid sigular derivative
        """
        
        hinge = torch.ones(len(y), device = self.device).long()
        hinge[~y] = -1
        
        dist = ((embeddings[graph[0]] - embeddings[graph[1]]).square().sum(-1)+1e-12).sqrt()
        
        return hinge, dist
    
    def get_bipartite_loss(self, bipartite_scores, bipartite_graph, batch):
        """
        Perform particle-track matching by minimum-weight bipartite matching
        """
        # Matching particle and tracks
        with torch.no_grad():
            # a set of virtual track candidates are added to ensure the existence of full matching
            # PID_CLUSTER_MAPPING is defined as the sum of scores of the hits of a specific particle to be assigned to a specific track
            pid_cluster_mapping = csr_matrix(
                (torch.cat([bipartite_scores, 1e-12*torch.ones(batch.pid.max()+1, device = self.device)], dim = 0).cpu().numpy(),
                (
                    torch.cat([batch.pid[bipartite_graph[0]], torch.arange(batch.pid.max()+1, device = self.device)], dim = 0).cpu().numpy(),
                    torch.cat([bipartite_graph[1], torch.arange(bipartite_graph[1].max()+1,
                                                                bipartite_graph[1].max()+batch.pid.max()+2, device = self.device)], dim = 0).cpu().numpy()
                )
                ),
                shape=(batch.pid.max()+1, bipartite_graph[1].max()+batch.pid.max()+2)
            )
            row_match, col_match = min_weight_full_bipartite_matching(pid_cluster_mapping, maximize=True)
            row_match, col_match = torch.tensor(row_match, device = self.device).long(), torch.tensor(col_match, device = self.device).long()
            noise_mask = (batch.original_pid[row_match] != 0) & (col_match < bipartite_graph[1].max()+1) # filter out noise and virtual tracks
            row_match, col_match = row_match[noise_mask], col_match[noise_mask]

            matched_particles = torch.tensor([False]*(batch.pid.max()+1), device = self.device)
            matched_particles[row_match] = True
            matched_hits = matched_particles[batch.pid[bipartite_graph[0]]]
            pid_assignments = torch.zeros((batch.pid.max()+1), device = self.device).long()
            pid_assignments[row_match] = col_match
            truth = torch.tensor([False]*len(bipartite_scores), device = self.device) 
            truth[matched_hits] = (pid_assignments[batch.pid[bipartite_graph[0]][matched_hits]] == bipartite_graph[1][matched_hits])
        
        # Compute bipartite loss
        asgmt_loss = torch.nn.functional.binary_cross_entropy(bipartite_scores, truth.float(), reduction='none')
        asgmt_loss = torch.dot(asgmt_loss, self.get_asgmt_weight(batch, bipartite_graph, row_match, col_match, truth)) # weight by pT
        
        return asgmt_loss, truth
        
    
    def training_step(self, batch, batch_idx):
        
        bipartite_graph, bipartite_score_logits, intermediate_embeddings, attention_logits = self(batch.input, batch.graph)
        bipartite_scores = torch.sigmoid(bipartite_score_logits + self.logit_schedule*attention_logits)
        
        # Compute embedding loss of edges using PID truth (whenever two ends of an edge have the same PID then define as true otherwise false)
        y_pid = batch.pid[batch.graph[0]] == batch.pid[batch.graph[1]]
        weights = self.get_emb_weight(batch, batch.graph, y_pid)
        hinge, dist = self.get_hinge_distance(batch, intermediate_embeddings, batch.graph, y_pid)

        emb_loss = nn.functional.hinge_embedding_loss(dist/self.hparams["train_r"], hinge, margin=1, reduction='none').square()
        emb_loss = torch.dot(emb_loss, weights)
        
        asgmt_loss, _ = self.get_bipartite_loss(bipartite_scores, bipartite_graph, batch)
        
        # Compute final loss using loss weight scheduling (sine scheduling)
        loss = (self.loss_schedule * emb_loss) + ((1-self.loss_schedule)*asgmt_loss)
        
        self.log_dict(
            {
                "training_loss": loss,
                "embedding_loss": emb_loss,
                "assignment_loss": asgmt_loss
                
            }
        )
        
        return loss


    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        bipartite_graph, bipartite_score_logits, intermediate_embeddings, attention_logits = self(batch.input, batch.graph)
        bipartite_scores = torch.sigmoid(bipartite_score_logits + self.logit_schedule*attention_logits)
        
        # Compute embedding loss

        y_pid = batch.pid[batch.graph[0]] == batch.pid[batch.graph[1]]
        weights = self.get_emb_weight(batch, batch.graph, y_pid)
        hinge, dist = self.get_hinge_distance(batch, intermediate_embeddings, batch.graph, y_pid)

        emb_loss = nn.functional.hinge_embedding_loss(dist/self.hparams["train_r"], hinge, margin=1, reduction='none').square()
        emb_loss = torch.dot(emb_loss, weights)

        asgmt_loss, truth = self.get_bipartite_loss(bipartite_scores, bipartite_graph, batch)

        loss = (self.loss_schedule * emb_loss) + ((1-self.loss_schedule)*asgmt_loss)
        
        # Compute Tracking Efficiency using not modified data to avoid miscalculation from removing isolated hits.
        bipartite_graph[0] = batch.inverse_mask[bipartite_graph[0]]
        cut_bipartite_graph = bipartite_graph[:, bipartite_scores >= self.hparams["score_cut"]]
        
        self.label_graph(batch, cut_bipartite_graph)
        cut_df = evaluate_labelled_graph(batch.cpu(), self.hparams["tracking"], matching_fraction=self.hparams["majority_cut"], matching_style="ATLAS", min_track_length=5, min_particle_length=5)
        
        # Compute best performance based on this bipartite graph
        best_bipartite_graph = bipartite_graph[:, truth]
        self.label_graph(batch, best_bipartite_graph)
        truth_df = evaluate_labelled_graph(batch.cpu(), self.hparams["tracking"], matching_fraction=self.hparams["majority_cut"], matching_style="ATLAS", min_track_length=5, min_particle_length=5)
        
        
        if log:
            self.log_dict(
                {
                    "val_loss": loss,
                    "val_embedding_loss": emb_loss,
                    "val_assignment_loss": asgmt_loss,
                    "track_eff": ((cut_df["is_reconstructable"] & cut_df["is_reconstructed"] & cut_df["is_signal"]).sum()
                                  / (cut_df["is_reconstructable"] & cut_df["is_signal"]).sum()),
                    "best_track_eff": ((truth_df["is_reconstructable"] & truth_df["is_reconstructed"] & truth_df["is_signal"]).sum()
                                       / (truth_df["is_reconstructable"] & truth_df["is_signal"]).sum())
                }
            )
        
        return bipartite_graph, loss

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs[1]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs[1]
    
    def label_graph(self, graph, bipartite_graph):
        
        bipartite_graph = bipartite_graph.cpu().numpy()
            
        # drop duplicates, TODO: support multiple assignment
        df = pd.DataFrame({"hit_id":bipartite_graph[0], "track_id":bipartite_graph[1]})
        df.drop_duplicates(subset="hit_id")
        labels = np.array([-1]*graph.x.shape[0], dtype = np.int64)
        labels[df["hit_id"]] = df["track_id"]

        graph.labels = torch.as_tensor(labels)
    
    def build_tracks(self, dataset, data_name):
        """
        Given a set of scored graphs, and a score cut, build tracks from graphs by:
        1. Applying the score cut to the graph
        2. Converting the graph to a sparse scipy array
        3. Running connected components on the sparse array
        4. Assigning the connected components labels back to the graph nodes as `labels` attribute
        """

        for graph in tqdm(dataset):
            graph = graph.to(self.device)
            bipartite_graph, bipartite_scores, *_ = self(graph.input, graph.graph)
            bipartite_scores = torch.sigmoid(bipartite_scores)

            # Apply score cut
            bipartite_graph = bipartite_graph[:, bipartite_scores >= self.hparams["score_cut"]]
            self.label_graph(graph, bipartite_graph)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()