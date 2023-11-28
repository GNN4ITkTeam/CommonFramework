import torch.nn as nn
import torch
from torch_geometric.nn import knn_graph, radius_graph, aggr
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .utils import make_mlp
from .metric_learning import MetricLearning, GraphDataset


class GravNetMetricLearning(MetricLearning):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Construct architecture
        # -------------------------

        in_channels = len(hparams["node_features"])
        self.network = None

        # Encode input features to hidden features
        self.feature_encoder = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # Construct the GravNet convolution modules
        self.grav_convs = nn.ModuleList(
            [GravConv(hparams) for _ in range(hparams["steps"])]
        )

        # Decode hidden features to output features
        decoder_concat_multiple = (
            hparams["steps"] + 1 if hparams["concat_output"] else 1
        )
        self.decoder_network = make_mlp(
            hparams["emb_hidden"] * decoder_concat_multiple,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )

        self.dataset_class = GraphDataset
        self.use_pyg = True
        self.save_hyperparameters(hparams)

    def forward(self, x):
        # Encode all features
        hidden_features = self.feature_encoder(x)

        # If concating, keep list of all output features
        all_hidden_features = [hidden_features]
        for i, grav_conv in enumerate(self.grav_convs):
            hidden_features, spatial_edges, _, grav_fact = checkpoint(
                grav_conv, hidden_features, self.current_epoch
            )

            self.log_dict(
                {
                    f"nbhood_sizes/nb_size_{i}": spatial_edges.shape[1]
                    / hidden_features.shape[0],
                    f"grav_facts/fact_{i}": grav_fact,
                },
                on_step=False,
                on_epoch=True,
            )

            all_hidden_features.append(hidden_features)

        if self.hparams["concat_output"] and self.hparams["steps"] > 0:
            hidden_features = torch.cat(all_hidden_features, dim=1)

        hidden_features = self.decoder_network(hidden_features)
        return F.normalize(hidden_features)


class GravConv(nn.Module):
    def __init__(self, hparams, input_size=None, output_size=None):
        super().__init__()
        self.hparams = hparams
        self.input_size = hparams["emb_hidden"] if input_size is None else input_size
        self.output_size = hparams["emb_hidden"] if output_size is None else output_size

        if "aggregation" not in hparams:
            hparams["aggregation"] = ["sum"]
            feature_network_input = 2 * (self.input_size + 1)
        elif isinstance(hparams["aggregation"], str):
            hparams["aggregation"] = [hparams["aggregation"]]
            feature_network_input = 2 * (self.input_size + 1)
        elif isinstance(hparams["aggregation"], list):
            feature_network_input = (1 + len(hparams["aggregation"])) * (
                self.input_size + 1
            )
        else:
            raise ValueError("Unknown aggregation type")

        self.grav_aggregation = aggr.MultiAggregation(
            hparams["aggregation"], mode="cat"
        )

        self.feature_network = make_mlp(
            feature_network_input,
            [self.output_size] * hparams["nb_layer"],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        self.spatial_network = make_mlp(
            self.input_size + 1,
            [self.input_size] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # This handles the various r, k, and random edge options
        self.setup_neighborhood_configuration()

    def get_neighbors(self, spatial_features):
        edge_index = torch.empty(
            [2, 0], dtype=torch.int64, device=spatial_features.device
        )

        if self.use_radius:
            radius_edges = radius_graph(
                spatial_features,
                r=self.r,
                max_num_neighbors=self.hparams["max_knn"],
                loop=self.hparams["self_loop"],
            )
            edge_index = torch.cat([edge_index, radius_edges], dim=1)

        if self.use_knn and self.knn > 0:
            k_edges = knn_graph(spatial_features, k=self.knn, loop=True)
            edge_index = torch.cat([edge_index, k_edges], dim=1)

        if self.use_rand_k and self.rand_k > 0:
            random_edges = knn_graph(
                torch.rand(
                    spatial_features.shape[0], 2, device=spatial_features.device
                ),
                k=self.rand_k,
                loop=True,
            )
            edge_index = torch.cat([edge_index, random_edges], dim=1)

        # Remove duplicate edges
        edge_index = torch.unique(edge_index, dim=1)

        return edge_index

    def get_grav_function(self, d):
        grav_weight = self.grav_weight
        grav_function = -grav_weight * d / self.r**2

        return grav_function, grav_weight

    def get_attention_weight(self, spatial_features, edge_index):
        start, end = edge_index
        d = torch.sum((spatial_features[start] - spatial_features[end]) ** 2, dim=-1)
        grav_function, grav_fact = self.get_grav_function(d)

        return torch.exp(grav_function), grav_fact

    def grav_pooling(self, spatial_features, hidden_features):
        edge_index = self.get_neighbors(spatial_features)
        start, end = edge_index
        d_weight, grav_fact = self.get_attention_weight(spatial_features, edge_index)

        if "norm_hidden" in self.hparams and self.hparams["norm_hidden"]:
            hidden_features = F.normalize(hidden_features, p=1, dim=-1)

        # print(hidden_features.shape, hidden_features[start].shape)
        aggregated_features = self.grav_aggregation(
            hidden_features[start] * d_weight.unsqueeze(1),
            end,
            dim=0,
            dim_size=hidden_features.shape[0],
        )

        return aggregated_features, edge_index, grav_fact

    def forward(self, hidden_features, current_epoch):
        self.current_epoch = current_epoch

        hidden_features = torch.cat(
            [hidden_features, hidden_features.mean(dim=1).unsqueeze(-1)], dim=-1
        )
        spatial_features = self.spatial_network(hidden_features)

        if "norm_embedding" in self.hparams and self.hparams["norm_embedding"]:
            spatial_features = F.normalize(spatial_features, p=2, dim=-1)

        aggregated_hidden, edge_index, grav_fact = self.grav_pooling(
            spatial_features, hidden_features
        )
        concatenated_hidden = torch.cat([aggregated_hidden, hidden_features], dim=-1)
        return (
            self.feature_network(concatenated_hidden),
            edge_index,
            spatial_features,
            grav_fact,
        )

    def setup_neighborhood_configuration(self):
        self.current_epoch = 0
        self.use_radius = bool("r_train" in self.hparams and self.hparams["r_train"])
        self.use_knn = bool("emb_knn" in self.hparams and self.hparams["emb_knn"])
        self.use_rand_k = bool("rand_k" in self.hparams and self.hparams["rand_k"])

    @property
    def r(self):
        if isinstance(self.hparams["r_train"], list):
            if len(self.hparams["r_train"]) == 2:
                return self.hparams["r_train"][0] + (
                    (self.hparams["r_train"][1] - self.hparams["r_train"][0])
                    * self.current_epoch
                    / self.hparams["max_epochs"]
                )
            elif len(self.hparams["r_train"]) == 3:
                if self.current_epoch < self.hparams["max_epochs"] / 2:
                    return self.hparams["r_train"][0] + (
                        (self.hparams["r_train"][1] - self.hparams["r_train"][0])
                        * self.current_epoch
                        / (self.hparams["max_epochs"] / 2)
                    )
                else:
                    return self.hparams["r_train"][1] + (
                        (self.hparams["r_train"][2] - self.hparams["r_train"][1])
                        * (self.current_epoch - self.hparams["max_epochs"] / 2)
                        / (self.hparams["max_epochs"] / 2)
                    )
        elif isinstance(self.hparams["r_train"], float):
            return self.hparams["r_train"]
        else:
            return 0.3

    @property
    def knn(self):
        if not isinstance(self.hparams["emb_knn"], list):
            return self.hparams["emb_knn"]
        if len(self.hparams["emb_knn"]) == 2:
            return int(
                self.hparams["emb_knn"][0]
                + (
                    (self.hparams["emb_knn"][1] - self.hparams["emb_knn"][0])
                    * self.current_epoch
                    / self.hparams["max_epochs"]
                )
            )
        elif len(self.hparams["emb_knn"]) == 3:
            return (
                int(
                    self.hparams["emb_knn"][0]
                    + (
                        (self.hparams["emb_knn"][1] - self.hparams["emb_knn"][0])
                        * self.current_epoch
                        / (self.hparams["max_epochs"] / 2)
                    )
                )
                if self.current_epoch < self.hparams["max_epochs"] / 2
                else int(
                    self.hparams["emb_knn"][1]
                    + (
                        (self.hparams["emb_knn"][2] - self.hparams["emb_knn"][1])
                        * (self.current_epoch - self.hparams["max_epochs"] / 2)
                        / (self.hparams["max_epochs"] / 2)
                    )
                )
            )
        else:
            raise ValueError("knn must be a list of length 2 or 3")

    @property
    def rand_k(self):
        if not isinstance(self.hparams["rand_k"], list):
            return self.hparams["rand_k"]
        if len(self.hparams["rand_k"]) == 2:
            return int(
                self.hparams["rand_k"][0]
                + (
                    (self.hparams["rand_k"][1] - self.hparams["rand_k"][0])
                    * self.current_epoch
                    / self.hparams["max_epochs"]
                )
            )
        elif len(self.hparams["rand_k"]) == 3:
            return (
                int(
                    self.hparams["rand_k"][0]
                    + (
                        (self.hparams["rand_k"][1] - self.hparams["rand_k"][0])
                        * self.current_epoch
                        / (self.hparams["max_epochs"] / 2)
                    )
                )
                if self.current_epoch < self.hparams["max_epochs"] / 2
                else int(
                    self.hparams["rand_k"][1]
                    + (
                        (self.hparams["rand_k"][2] - self.hparams["rand_k"][1])
                        * (self.current_epoch - self.hparams["max_epochs"] / 2)
                        / (self.hparams["max_epochs"] / 2)
                    )
                )
            )
        else:
            raise ValueError("rand_k must be a list of length 2 or 3")

    @property
    def grav_weight(self):
        if (
            isinstance(self.hparams["grav_weight"], list)
            and len(self.hparams["grav_weight"]) == 2
        ):
            return (
                self.hparams["grav_weight"][0]
                + (self.hparams["grav_weight"][1] - self.hparams["grav_weight"][0])
                * self.current_epoch
                / self.hparams["max_epochs"]
            )
        elif isinstance(self.hparams["grav_weight"], float):
            return self.hparams["grav_weight"]
        else:
            raise ValueError("grav_weight must be a list of length 2 or a float")
