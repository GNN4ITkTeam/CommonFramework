import warnings
import torch
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add, scatter_mean, scatter_max

from gnn4itk_cf.utils import make_mlp
from ..edge_classifier_stage import EdgeClassifierStage


class InteractionGNN(EdgeClassifierStage):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Define the dataset to be used, if not using the default
        self.save_hyperparameters(hparams)

        self.setup_layer_sizes()

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )

        # Setup input network
        self.node_encoder = make_mlp(
            len(hparams["node_features"]),
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            self.concatenation_factor * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")
        except Exception:
            warnings.warn("Failed to define figures of merit, due to logger unavailable")

    def aggregation_step(self, num_nodes):

        aggregation_dict = {
            "sum_max": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
                    scatter_add(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "mean_max": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
                    scatter_mean(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "mean_sum": lambda x, y, **kwargs: torch.cat(
                [
                    scatter_mean(x, y, dim=0, dim_size=num_nodes),
                    scatter_add(x, y, dim=0, dim_size=num_nodes),
                ],
                dim=-1,
            ),
            "sum": lambda x, y, **kwargs: scatter_add(x, y, dim=0, dim_size=num_nodes),
            "mean": lambda x, y, **kwargs: scatter_mean(x, y, dim=0, dim_size=num_nodes),
            "max": lambda x, y, **kwargs: scatter_max(x, y, dim=0, dim_size=num_nodes)[0],
        }

        return aggregation_dict[self.hparams["aggregation"]]

    def message_step(self, x, start, end, e):

        # Compute new node features
        # edge_messages = self.aggregation_step(x.shape[0])(e, end)
        edge_messages = torch.cat(
            [
                self.aggregation_step(x.shape[0])(e, end),
                self.aggregation_step(x.shape[0])(e, start)
            ],
            dim=-1,
        )

        node_inputs = torch.cat([x, edge_messages], dim=-1)

        x_out = self.node_network(node_inputs)

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)

        return x_out, e_out

    def output_step(self, x, start, end, e):

        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)

        return self.output_edge_classifier(classifier_inputs).squeeze(-1)

    def forward(self, batch):

        x = torch.stack([batch[feature] for feature in self.hparams["node_features"]], dim=-1).float()
        start, end = batch.edge_index

        # Encode the graph features into the hidden space
        x.requires_grad = True
        x = checkpoint(self.node_encoder, x)
        e = checkpoint(self.edge_encoder, torch.cat([x[start], x[end]], dim=1))

        # Loop over iterations of edge and node networks
        for _ in range(self.hparams["n_graph_iters"]):
            x, e = checkpoint(self.message_step, x, start, end, e)

        return self.output_step(x, start, end, e)

    def setup_layer_sizes(self):

        if self.hparams["aggregation"] == "sum_mean_max":
            self.concatenation_factor = 7
        elif self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"]:
            self.concatenation_factor = 5
        elif self.hparams["aggregation"] in ["sum", "mean", "max"]:
            self.concatenation_factor = 3
        else:
            raise ValueError("Aggregation type not recognised")