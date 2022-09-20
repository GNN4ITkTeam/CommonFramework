# 3rd party imports
from ..graph_construction_stage import GraphConstructionStage
import torch.nn.functional as F

# Local imports
from ..utils import make_mlp


class LayerlessEmbedding(GraphConstructionStage):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        # Construct the MLP architecture
        if "ci" in hparams["regime"]:
            in_channels = hparams["spatial_channels"] + hparams["cell_channels"]
        else:
            in_channels = hparams["spatial_channels"]

        self.network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):

        x_out = self.network(x)

        if "norm" in self.hparams["regime"]:
            return F.normalize(x_out)
        else:
            return x_out
