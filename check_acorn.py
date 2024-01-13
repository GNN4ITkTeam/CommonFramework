#!/usr/bin/env python
"""Check if the python environment is set up correctly for the acorn project.
Requirement following key python packages:
pytorch
pytorch-lightning
pyg
frnn
cugraph
torch_scatter
"""


def check():
    # python interpreter
    import sys

    print("python interpreter: ", sys.executable)

    try:
        import torch

        print("torch: ", torch.__version__)
        print("torch cuda: ", torch.cuda.is_available())
        print("torch cuda device count: ", torch.cuda.device_count())
        print("torch cuda device name: ", torch.cuda.get_device_name())
        print("torch cuda device capability: ", torch.cuda.get_device_capability())
        print("torch distributed     :", torch.distributed.is_available())
    except ImportError:
        print("torch not found")

    try:
        import pytorch_lightning

        print("pytorch_lightning: ", pytorch_lightning.__version__)
    except ImportError:
        print("pytorch_lightning not found")

    try:
        import torch_geometric

        print("pyg: ", torch_geometric.__version__)
    except (ImportError, OSError) as error:
        print(error)
        print("pyg not found")

    try:
        import frnn  # noqa

        print("frnn found")
    except ImportError:
        print("frnn not found")

    try:
        import cugraph

        print("cugraph: ", cugraph.__version__)
    except ImportError:
        print("cugraph not found")

    try:
        import cudf

        print("cudf: ", cudf.__version__)
    except ImportError:
        print("cudf not found")

    try:
        import torch_scatter

        print("torch_scatter: ", torch_scatter.__version__)
        import torch
        from torch_scatter import scatter_max

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Test scatter_max in {device}.")
        src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]).to(device)
        index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]]).to(device)

        out, argmax = scatter_max(src, index, dim=-1)
        print("out:", out)
        print("argmax:", argmax)

    except ImportError:
        print("torch_scatter not found")


if __name__ == "__main__":
    check()
