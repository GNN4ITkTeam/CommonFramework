"""Helpers shared by the PyMMG stages."""

import torch


def configure_mmg_device(hparams):
    """Resolve the CUDA device index the PyMMG models should run on.

    `accelerator` and `devices` follow the Lightning convention, where a list
    enumerates GPU indices and an integer is a *count* of GPUs. ModuleMapGraph
    is single-GPU, so only one index, or a count of one, is accepted. Asking for
    more raises rather than quietly running on a single device, so that the
    misconfiguration is fixed in the yaml rather than discovered in the logs.

    Raises ValueError for a non-CUDA accelerator or an unusable `devices`, and
    RuntimeError when no CUDA runtime is available.
    """
    accelerator = hparams.get("accelerator", "cuda")
    if accelerator != "cuda":
        raise ValueError("ModuleMapGraph currently only supports CUDA.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA runtime not available. MMG currently requires an NVIDIA GPU with CUDA."
        )

    devices = hparams.get("devices", [0])
    if isinstance(devices, list):
        if len(devices) != 1:
            raise ValueError(
                f"ModuleMapGraph only supports a single GPU, but {devices} were "
                "requested. Set a single device index, such as [2]."
            )
        return int(devices[0])
    if isinstance(devices, int):
        if devices != 1:
            raise ValueError(
                f"ModuleMapGraph only supports a single GPU, but a count of "
                f"{devices} was requested. Use 1, or a list such as [2] to select "
                "a device index."
            )
        return 0
    raise ValueError(f"Invalid device configuration: {devices}")
