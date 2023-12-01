from torch import nn
import importlib


class GCNEncoder(nn.Module):
    def __init__(self, gnn_config) -> None:
        super().__init__()
        self.gnn_config = gnn_config
        self.layers = nn.ModuleList()

        for conf in gnn_config:
            # gnn_config should be a list of dict like
            # {module_name: torch_geometric.nn, class_name: SAGEConv, init_kwargs: {in_channels: 37, out_channels: 256, }, inputs: [x, adj_t]},
            # import parent module
            module = importlib.import_module(conf["module_name"])
            # get class and initialize programmatically
            layer = getattr(module, conf["class_name"])(**conf["init_kwargs"])
            self.layers.append(layer)

    def forward(self, x, adj_t):
        for layer, conf in zip(self.layers, self.gnn_config):
            # gather input elements from conf['inputs']
            inputs = []
            for key in conf["inputs"]:
                if key == "x":
                    inputs.append(x)
                if key == "adj_t":
                    inputs.append(adj_t)
            x = layer(*inputs)
        return x
