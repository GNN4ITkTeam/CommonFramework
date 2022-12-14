# 3rd party imports
from ..graph_construction_stage import GraphConstructionStage
import torch.nn.functional as F

from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset
import torch

import logging

# Local imports
from .utils import make_mlp, build_edges, graph_intersection
from ..utils import handle_weighting, handle_hard_node_cuts, build_signal_edges
from gnn4itk_cf.utils import load_datafiles_in_dir

class MetricLearning(GraphConstructionStage, LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        in_channels = len(hparams["node_features"])

        self.network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.dataset_class = GraphDataset
        self.use_pyg = True
        self.save_hyperparameters(hparams)       

    def forward(self, x):

        x_out = self.network(x)
        return F.normalize(x_out)

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(
                self.trainset, batch_size=1, num_workers=0
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

    def get_input_data(self, batch):

        input_data = torch.stack([batch[feature] for feature in self.hparams["node_features"]], dim=-1).float()
        input_data[input_data != input_data] = 0 # Replace NaNs with 0s

        return input_data

    def get_query_points(self, batch, spatial):

        query_indices = torch.arange(len(spatial), device=self.device)
        query_indices = query_indices[torch.randperm(len(query_indices))][
            : self.hparams["points_per_batch"]
        ]
        query = spatial[query_indices]       

        return query_indices, query

    def append_hnm_pairs(self, e_spatial, query, query_indices, spatial, r_train=None, knn=None):
        if r_train is None:
            r_train = self.hparams["r_train"]
        if knn is None:
            knn = self.hparams["knn"]
        
        knn_edges: torch.Tensor = build_edges(
            query=query,
            database=spatial,
            indices=query_indices,
            r_max=r_train,
            k_max=knn,
            backend="FRNN",
            return_indices=False
        )

        e_spatial = torch.cat([e_spatial, knn_edges], dim=-1)

        return e_spatial

    def append_random_pairs(self, e_spatial, query_indices, spatial):
        n_random = int(self.hparams["randomisation"] * len(query_indices))
        indices_src = torch.randint(
            0, len(query_indices), (n_random,), device=self.device
        )
        indices_dest = torch.randint(0, len(spatial), (n_random,), device=self.device)
        random_pairs = torch.stack([query_indices[indices_src], indices_dest])

        e_spatial = torch.cat(
            [e_spatial, random_pairs],
            dim=-1,
        )
        return e_spatial

    def get_hinge_distance(self, spatial, e_spatial, y_cluster, weights=None):

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatial[e_spatial[1]]
        neighbors = spatial[e_spatial[0]]

        try: # This can be resource intensive, so we chunk it if it fails
            d = torch.sum((reference - neighbors) ** 2, dim=-1)
        except RuntimeError:
            d = [torch.sum((ref - nei) ** 2, dim=-1) for ref, nei in zip(reference.chunk(10), neighbors.chunk(10))]
            d = torch.cat(d)
        
        if weights is not None:
            d = d * weights

        return hinge, d

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        training_edges, embedding = self.get_training_edges(batch)
        self.apply_embedding(batch, embedding, training_edges)

        training_edges, truth, truth_map, true_edges = self.get_truth(batch, training_edges)
        weights = self.get_training_weights(batch, training_edges, truth, true_edges, truth_map)

        loss = self.loss_function(embedding, training_edges, truth, weights)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def get_training_edges(self, batch):
        
        # Instantiate empty prediction edge list
        training_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch)

        with torch.no_grad():
            embedding = self(input_data)

        query_indices, query = self.get_query_points(batch, embedding)
        
        # Append Hard Negative Mining (hnm) with KNN graph
        training_edges = self.append_hnm_pairs(training_edges, query, query_indices, embedding)

        # Append random edges pairs (rp) for stability
        training_edges = self.append_random_pairs(training_edges, query_indices, embedding)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        if "undirected" in self.hparams and self.hparams["undirected"]:
            true_edges = torch.cat(
                [batch.track_edges, batch.track_edges.flip(0)], dim=-1
            )
        else:
            true_edges = batch.track_edges
        training_edges = torch.cat(
            [training_edges, true_edges], dim=-1,
        )

        return training_edges, embedding

    def get_truth(self, batch, pred_edges):

        # Calculate truth from intersection between Prediction graph and Truth graph
        if "undirected" in self.hparams and self.hparams["undirected"]:
            true_edges = torch.cat(
                [batch.track_edges, batch.track_edges.flip(0)], dim=-1
            )
        else:
            true_edges = batch.track_edges

        truth, truth_map = graph_intersection(pred_edges, true_edges, return_y_pred=True, return_truth_to_pred=True)

        return pred_edges, truth, truth_map, true_edges

    def get_training_weights(self, batch, training_edges, truth, true_edges, truth_map):

        return handle_weighting(batch, training_edges, truth, self.hparams["weighting"], true_edges, truth_map)

    def apply_embedding(self, batch, embedding_inplace=None, training_edges=None):

        # Apply embedding to input data
        input_data = self.get_input_data(batch)
        if embedding_inplace is None or training_edges is None:
            return self(input_data)

        included_hits = training_edges.unique().long()
        print(embedding_inplace.shape)
        print(input_data.shape)
        embedding_inplace[included_hits] = self(input_data[included_hits])

    def loss_function(self, embedding, pred_edges, truth, weights=None):
        hinge, d = self.get_hinge_distance(embedding, pred_edges, truth, weights)

        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        loss = negative_loss + self.hparams["weight"] * positive_loss

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num):

        embedding = self.apply_embedding(batch)

        # Build whole KNN graph
        pred_edges = build_edges(
            embedding, embedding, indices=None, r_max=knn_radius, k_max=knn_num
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        pred_edges, truth, truth_map, true_edges = self.get_truth(batch, pred_edges)

        hinge, d = self.get_hinge_distance(
            embedding, pred_edges.to(self.device), truth
        )

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"]**2, reduction="mean"
        )

        cluster_true = true_edges.shape[1]
        cluster_true_positive = truth.sum()
        cluster_positive = pred_edges.shape[1]

        eff = cluster_true_positive / cluster_true
        pur = cluster_true_positive / cluster_positive

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {"val_loss": loss, "eff": eff, "pur": pur, "current_lr": current_lr}
        )
        logging.info(f"Efficiency: {eff}")
        logging.info(f"Purity: {pur}")
        logging.info(batch.event_id)

        return {
            "loss": loss,
            "distances": d,
            "preds": embedding,
            "truth": truth,
            "truth_graph": true_edges,
        }

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        knn_val = 500 if "knn_val" not in self.hparams else self.hparams["knn_val"]
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_train"], knn_val
        )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        return self.shared_evaluation(batch, batch_idx, self.hparams["r_train"], 1000)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        logging.info(f"Optimizer step for batch {batch_idx}")
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        logging.info(f"Optimizer step done for batch {batch_idx}")

class GraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(self, input_dir, data_name = None, num_events = None, stage="fit", hparams=None, transform=None, pre_transform=None, pre_filter=None, **kwargs):
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
        # print(event)
        self.preprocess_event(event)

        # return (event, event_path) if self.stage == "predict" else event
        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """
        
        self.apply_hard_cuts(event)
        # print(event)
        self.build_signal_edges(event)
        # print(event)
        
    def apply_hard_cuts(self, event):
        """
        Apply hard cuts to the event. This is implemented by 
        1. Finding which true edges are from tracks that pass the hard cut.
        2. Pruning the input graph to only include nodes that are connected to these edges.
        """
        
        if self.hparams is not None and "hard_cuts" in self.hparams.keys() and self.hparams["hard_cuts"]:
            assert isinstance(self.hparams["hard_cuts"], dict), "Hard cuts must be a dictionary"
            handle_hard_node_cuts(event, self.hparams["hard_cuts"])

    def build_signal_edges(self, event):
        """
        Build signal edges for the event. This is implemented by finding which true edges are from tracks that pass the signal cut.
        """
        
        if self.hparams is not None and "weighting" in self.hparams.keys() and self.hparams["weighting"]:
            build_signal_edges(event, self.hparams["weighting"])
            
    def handle_edge_list(self, event):
        """
        TODO 
        """ 
        pass
