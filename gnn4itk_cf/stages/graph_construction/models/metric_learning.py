import os
import logging

# 3rd party imports
from ..graph_construction_stage import GraphConstructionStage
import torch.nn.functional as F

from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset
import torch

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
        layer_norm = hparams["layer_norm"] if "layer_norm" in hparams else True

        self.network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=layer_norm,
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

    def predict_dataloader(self):

        datasets = []
        for data_name, data_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"]):
            if data_num > 0:
                dataset = self.dataset_class(self.hparams["input_dir"], data_name, data_num, "predict", self.hparams)
                datasets.append(dataset) 
        return datasets

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

        input_data = torch.stack([batch["x_" + feature] for feature in self.hparams["node_features"]], dim=-1).float()
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
        
        knn_edges = build_edges(
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

    def append_signal_edges(self, batch, edges):

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        if "undirected" in self.hparams and self.hparams["undirected"]:
            true_edges = torch.cat(
                [batch.track_edges, batch.track_edges.flip(0)], dim=-1
            )
        else:
            true_edges = batch.track_edges

        # Append the signal edges
        signal_true_edges = build_signal_edges(
            batch,
            self.hparams["weighting"],
            true_edges
        )

        # print("Adding signal edges:", signal_true_edges.shape[1], "from true edges:", true_edges.shape[1])

        edges = torch.cat(
            [edges, signal_true_edges], dim=-1,
        )

        # print("weighting:", self.hparams["weighting"])

        return edges

    def get_distances(self, embedding, pred_edges):

        reference = embedding[pred_edges[1]]
        neighbors = embedding[pred_edges[0]]

        try: # This can be resource intensive, so we chunk it if it fails
            d = torch.sum((reference - neighbors) ** 2, dim=-1)
        except RuntimeError:
            d = [torch.sum((ref - nei) ** 2, dim=-1) for ref, nei in zip(reference.chunk(10), neighbors.chunk(10))]
            d = torch.cat(d)
        
        return d

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        batch.edge_index, embedding = self.get_training_edges(batch)
        self.apply_embedding(batch, embedding, batch.edge_index)

        batch.edge_index, batch.y, truth_map, true_edges = self.get_truth(batch, batch.edge_index)
        weights = self.get_weights(batch, true_edges, truth_map)

        loss = self.loss_function(batch, embedding, weights)

        self.log("train_loss", loss, batch_size=1)

        return loss

    def get_training_edges(self, batch):
        
        # Instantiate empty prediction edge list
        training_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        with torch.no_grad():
            embedding = self.apply_embedding(batch)

        query_indices, query = self.get_query_points(batch, embedding)
        
        # Append Hard Negative Mining (hnm) with KNN graph
        training_edges = self.append_hnm_pairs(training_edges, query, query_indices, embedding)

        # Append random edges pairs (rp) for stability
        training_edges = self.append_random_pairs(training_edges, query_indices, embedding)

        # Append true signal edges
        training_edges = self.append_signal_edges(batch, training_edges)

        return training_edges, embedding

    def get_truth(self, batch, pred_edges):

        # Calculate truth from intersection between Prediction graph and Truth graph
        if "undirected" in self.hparams and self.hparams["undirected"]:
            true_edges = torch.cat(
                [batch.track_edges, batch.track_edges.flip(0)], dim=-1
            )
        else:
            true_edges = batch.track_edges

        pred_edges, truth, truth_map = graph_intersection(pred_edges, true_edges, return_y_pred=True, return_truth_to_pred=True, unique_pred=False)

        return pred_edges, truth, truth_map, true_edges

    def get_weights(self, batch, true_edges, truth_map):

        return handle_weighting(batch, self.hparams["weighting"], true_edges=true_edges, truth_map=truth_map)

    def apply_embedding(self, batch, embedding_inplace=None, training_edges=None):

        # Apply embedding to input data
        input_data = self.get_input_data(batch)
        if embedding_inplace is None or training_edges is None:
            return self(input_data)

        included_hits = training_edges.unique().long()
        embedding_inplace[included_hits] = self(input_data[included_hits])

    def loss_function(self, batch, embedding, weights=None, pred_edges=None, truth=None):
        
        if pred_edges is None:
            assert "edge_index" in batch.keys, "Must provide pred_edges if not in batch"
            pred_edges = batch.edge_index

        if truth is None:
            assert "y" in batch.keys, "Must provide truth if not in batch"
            truth = batch.y

        if weights is None:
            weights = torch.ones_like(truth)

        d = self.get_distances(embedding, pred_edges)

        return self.weighted_hinge_loss(truth, d, weights)

    def weighted_hinge_loss(self, truth, d, weights):
        """
        Calculates the weighted hinge loss

        Given a set of edges, we partition into signal true (truth=1, weight>0), background true (truth=1, weight=0 or weight<0) and false (truth=0). 
        The choice of weights for each set (as specified in the weighting config) defines how these are treated. Weights of 0 are simply masked from the loss function.
        Weights below 0 are treated as false, such that background true edges can be treated as false edges. The same behavior is used in calculating metrics.

        Args:
            truth (``torch.tensor``, required): The truth tensor of composed of 0s and 1s, of shape (E,)
            d (``torch.tensor``, required): The distance tensor between nodes at edges[0] and edges[1] of shape (E,)
            weights (``torch.tensor``, required): The weight tensor of shape (E,)
        Returns:
            ``torch.tensor`` The weighted hinge loss mean as a tensor
        """
        
        negative_mask = ((truth == 0) & (weights != 0)) | (weights < 0) 
        # print("Negatives:", negative_mask.shape, (truth==0).sum(), (weights <0).sum(), (weights != 0).sum())

        # Handle negative loss, but don't reduce vector
        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[negative_mask],
            torch.ones_like(d[negative_mask])*-1,
            margin=self.hparams["margin"]**2,
            reduction= "none",
        )

        # Now reduce the vector with non-zero weights
        negative_loss = torch.mean(negative_loss * weights[negative_mask].abs())

        positive_mask = (truth == 1) & (weights > 0)
        # print("Positives:", positive_mask.shape, (truth==1).sum(), (weights > 0).sum())

        # Handle positive loss, but don't reduce vector
        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[positive_mask],
            torch.ones_like(d[positive_mask]),
            margin=self.hparams["margin"]**2,
            reduction= "none",
        )

        # Now reduce the vector with non-zero weights
        positive_loss = torch.mean(positive_loss * weights[positive_mask].abs())

        return negative_loss + positive_loss

    def shared_evaluation(self, batch, knn_radius, knn_num):

        embedding = self.apply_embedding(batch)

        # Build whole KNN graph
        batch.edge_index = build_edges(
            query=embedding, database=embedding, indices=None, r_max=knn_radius, k_max=knn_num, backend="FRNN"
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        batch.edge_index, batch.y, batch.truth_map, true_edges = self.get_truth(batch, batch.edge_index)

        weights = self.get_weights(batch, true_edges, batch.truth_map)

        d = self.get_distances(
            embedding, batch.edge_index
        )

        loss = self.weighted_hinge_loss(batch.y, d, weights)

        if hasattr(self, "trainer") and self.trainer.state.stage in ["train", "validate"]:
            self.log_metrics(batch, loss, batch.edge_index, true_edges, batch.y, weights)

        return {
            "loss": loss,
            "distances": d,
            "preds": embedding,
            "truth_graph": true_edges,
        }

    def log_metrics(self, batch, loss, pred_edges, true_edges, truth, weights):

        signal_true_edges = build_signal_edges(batch, self.hparams["weighting"], true_edges)
        true_pred_edges = pred_edges[:, truth == 1]
        signal_true_pred_edges = pred_edges[:, (truth == 1) & (weights > 0)]

        total_eff = true_pred_edges.shape[1] / true_edges.shape[1]
        signal_eff = signal_true_pred_edges.shape[1] / signal_true_edges.shape[1]
        total_pur = true_pred_edges.shape[1] / pred_edges.shape[1]
        signal_pur = signal_true_pred_edges.shape[1] / pred_edges.shape[1]
        f1 = 2 * (signal_eff * signal_pur) / (signal_eff + signal_pur)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {"val_loss": loss, "lr": current_lr, "total_eff": total_eff, "total_pur": total_pur, "signal_eff": signal_eff, "signal_pur": signal_pur, "f1": f1},
            batch_size=1
        )

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        # check global step
        if self.trainer.global_step == 0:
            self.define_figures_of_merit()
            
        knn_val = 500 if "knn_val" not in self.hparams else self.hparams["knn_val"]
        outputs = self.shared_evaluation(
            batch, self.hparams["r_train"], knn_val
        )

        return outputs["loss"]

    def define_figures_of_merit(self):

        # Check if self has logger key
        try:
            self.logger.experiment.define_metric("val_loss" , summary="min")
            if "metric_to_monitor" in self.hparams and "metric_mode" in self.hparams:
                self.logger.experiment.define_metric(self.hparams["metric_to_monitor"], summary=self.hparams["metric_mode"])

        except Exception as e:
            raise e
            warnings.warn("Failed to define figures of merit, due to logger unavailable")

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        return self.shared_evaluation(batch, self.hparams["r_train"], 1000)

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        This function handles the prediction of each graph. It is called in the `infer.py` script.
        It can be overwritted in your custom stage, but it should implement three simple steps:
        1. Run an embedding model on the input graph
        2. Build graph and save as batch.edge_index
        3. Run the truth calculation and save as batch.y and batch.truth_map
        """
            
        knn_infer = 500 if "knn_infer" not in self.hparams else self.hparams["knn_infer"]
        self.shared_evaluation(batch, self.hparams["r_infer"], knn_infer)
        
        if self.hparams["undirected"]:
            self.remove_duplicate_edges(batch)

        datatype = self.predict_dataloader()[dataloader_idx].data_name

        self.build_graphs(batch, datatype)

    def build_graphs(self, event, datatype):

        event.config.append(self.hparams)
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
        event.truth_map[event.truth_map >= 0] = edge_inverse[event.truth_map[event.truth_map >= 0]]
        event.truth_map = event.truth_map[:event.track_edges.shape[1]]

        random_flip = torch.randint(2, (event.edge_index.shape[1],), dtype=torch.bool)
        event.edge_index[:, random_flip] = event.edge_index[:, random_flip].flip(0)


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
        self.preprocess_event(event)

        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """
        
        self.clean_node_features(event)
        self.apply_hard_cuts(event)
        # self.remove_split_cluster_truth(event) TODO: Should handle this at some point
        
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
            
    def clean_node_features(self, event):
        """
        Ensure that node features abide by the correct convention. That is, they begin with "x_".
        """

        if "x" in event.keys:
            event.num_nodes = event.x.shape[0]

        for feature in event.keys:
            if event.is_node_attr(feature) and not feature.startswith("x_"):
                event[f"x_{feature}"] = event[feature]
                event[feature] = None

    def handle_edge_list(self, event):
        """
        TODO 
        """ 
        pass
