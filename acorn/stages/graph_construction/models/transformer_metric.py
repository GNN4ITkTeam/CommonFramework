import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .utils import make_mlp

# 3rd party imports
from .metric_learning import MetricLearning, GraphDataset


class TransformerMetricLearning(MetricLearning):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        in_channels = len(hparams["node_features"])

        self.network = (
            None  # Remove the default metric learning network to save some space
        )

        self.encoder_network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"],
            hidden_activation=hparams["activation"],
            output_activation=hparams["activation"],
            layer_norm=True,
        )

        decoder_concat_multiple = (
            hparams["steps"] + 1 if hparams["concat_output"] else 1
        )
        self.decoder_network = make_mlp(
            hparams["emb_hidden"] * decoder_concat_multiple,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.setup_transformer_layers(hparams)

        self.dataset_class = GraphDataset
        self.use_pyg = True
        self.save_hyperparameters(hparams)

    def forward(self, x):
        # Encode the input
        x = self.encoder_network(x)
        all_x = [x]

        for i in range(self.hparams["steps"]):
            # Apply the attention layer
            if self.hparams["builtin_transformer"]:
                x = checkpoint(self.transformer_layers[i], x)
            elif self.hparams["QKV_reversed"]:
                x = checkpoint(self.attention_layer, x, i)
            else:
                x = checkpoint(self.attention_layer_2, x, i)

            all_x.append(x)

        # Concatenate and decode the embeddings
        if self.hparams["concat_output"] and self.hparams["steps"] > 0:
            x = torch.cat(all_x, dim=1)
        x = self.decoder_network(x)
        return F.normalize(x)

    def attention_layer(self, x, step_idx):
        # Create query and key embeddings
        q = self.q_layers[step_idx](x)
        k = self.k_layers[step_idx](x)
        v = self.v_layers[step_idx](x)

        # Normalize the embeddings
        q = F.normalize(q)
        k = F.normalize(k)
        v = F.normalize(v)

        # Compute the attention
        dot_product = torch.matmul(k.transpose(0, 1), v)

        # Update the embeddings
        x = x + self.update_networks[step_idx](torch.matmul(q, dot_product))
        return x

    def attention_layer_2(self, x, step_idx):
        # Create query and key embeddings
        q = self.q_layers[step_idx](x)
        k = self.k_layers[step_idx](x)
        v = self.v_layers[step_idx](x)

        # Compute the attention
        dot_product = torch.matmul(q, k.transpose(0, 1)) / (
            self.hparams["emb_hidden"] ** 0.5
        )
        dot_product = F.softmax(dot_product, dim=1)

        # Update the embeddings
        x = x + self.update_networks[step_idx](torch.matmul(dot_product, v))
        return x

    def setup_transformer_layers(self, hparams):
        if not hparams["builtin_transformer"]:
            self.update_networks = torch.nn.ModuleList(
                [
                    make_mlp(
                        hparams["emb_hidden"],
                        [hparams["emb_hidden"]] * hparams["nb_layer"],
                        hidden_activation=hparams["activation"],
                        output_activation=hparams["activation"],
                        layer_norm=True,
                    )
                    for _ in range(hparams["steps"])
                ]
            )
            self.q_layers, self.k_layers, self.v_layers = (
                [
                    torch.nn.ModuleList(
                        [
                            make_mlp(
                                hparams["emb_hidden"],
                                [hparams["emb_hidden"]] * hparams["nb_layer"],
                                hidden_activation=hparams["activation"],
                                output_activation=hparams["activation"],
                                layer_norm=True,
                            )
                            for _ in range(hparams["steps"])
                        ]
                    )
                    for _ in range(3)
                ]
                if hparams["nonlinear_QKV"]
                else [
                    torch.nn.ModuleList(
                        [
                            torch.nn.Linear(
                                hparams["emb_hidden"], hparams["emb_hidden"]
                            )
                            for _ in range(hparams["steps"])
                        ]
                    )
                    for _ in range(3)
                ]
            )
        else:
            self.transformer_layers = torch.nn.ModuleList(
                [
                    torch.nn.TransformerEncoderLayer(
                        d_model=hparams["emb_hidden"],
                        nhead=hparams["nb_heads"],
                        dim_feedforward=hparams["emb_hidden"],
                        dropout=0,
                        activation=hparams["activation"].lower(),
                        batch_first=True,
                    )
                    for _ in range(hparams["steps"])
                ]
            )


class MultiheadedTransformerMetricLearning(MetricLearning):
    """
    The same functionality as the above TransformerMetricLearning class, but with a multihead hyperparameter
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.num_heads = hparams["num_heads"]

    def attention_layer(self, x, step_idx):
        # Create query and key embeddings
        q = self.q_layers[step_idx](x)
        k = self.k_layers[step_idx](x)
        v = self.v_layers[step_idx](x)

        # Normalize the embeddings
        q = F.normalize(q)
        k = F.normalize(k)
        v = F.normalize(v)

        # Compute the attention
        dot_product = torch.matmul(k.transpose(1, 2), v)

        # Update the embeddings
        x = x + self.update_networks[step_idx](torch.matmul(q, dot_product))
        return x
