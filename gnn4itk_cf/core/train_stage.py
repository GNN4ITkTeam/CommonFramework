# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script:
1. Loads a training config
2. Checks the stage to train
3. Loads the stage module
4. Trains the stage
5. Tests the output
"""
import os
import yaml
import click

try:
    import wandb
except ImportError:
    wandb = None

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelPruning

# import torch.nn.utils.prune as prune # only needed for evil hack, see below

from .core_utils import str_to_class, get_trainer, get_stage_module
from ..utils import onnx_export


@click.command()
@click.argument("config_file")
# Add an optional click argument to specify the checkpoint to use
@click.option("--checkpoint", "-c", default=None, help="Checkpoint to use for training")
def main(config_file, checkpoint):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """
    train(config_file, checkpoint)


# Refactoring to allow for auto-resume and manual resume of training
# 1. We cannot init a model before we know if we are resuming or not
# 2. First check if the module is a lightning module


def train(config_file, checkpoint=None):
    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # allows to use wandb.ai sweep functionality
    if wandb is not None:
        wandb.init(
            project=config["project"],
            # track hyperparameters and run metadata
            config=config,
            group="DDP",  # all runs for the experiment in one group
        )
        config.update(dict(wandb.config))

    print(config)
    # load stage
    stage = config["stage"]
    model = config["model"]
    stage_module_class = str_to_class(stage, model)

    # setup stage
    os.makedirs(config["stage_dir"], exist_ok=True)

    # run training, depending on whether we are using a Lightning trainable model or not
    if issubclass(stage_module_class, LightningModule):
        lightning_train(config, stage_module_class, checkpoint=checkpoint)
    else:
        stage_module = stage_module_class(config)
        stage_module.setup(stage="fit")
        stage_module.train()


def lightning_train(config, stage_module_class, checkpoint=None):
    global stage_module, trainer, parameters_to_prune
    stage_module, config, default_root_dir = get_stage_module(
        config, stage_module_class, checkpoint_path=checkpoint
    )

    trainer = get_trainer(config, default_root_dir)

    if config["onnx_export"]:
        trainer.callbacks.append(onnx_export())

    if config["pruning_allow"]:
        # we only want to prune weights from the Linear or QuantLinear layers;
        # checking if layer has weights, and excluding LayerNorm (we may need to expand that list; added BatchNorm1d)
        parameters_to_prune = [
            (x, "weight")
            for x in stage_module.network[:]
            if (
                hasattr(x, "weight")
                and (x.__class__.__name__ != "LayerNorm")
                and (x.__class__.__name__ != "BatchNorm1d")
            )
        ]
        # print(parameters_to_prune)

        trainer.callbacks.append(
            ModelPruning(
                pruning_fn=config["pruning_fn"],
                parameters_to_prune=parameters_to_prune,
                amount=config["pruning_amount"],
                apply_pruning=apply_pruning,
                # settings below only for structured!
                pruning_dim=config["pruning_dim"],
                pruning_norm=config["pruning_norm"],
                use_global_unstructured=config["use_global_unstructured"],
                verbose=1,  # 2 for per-layer sparsity, #1 for overall sparsity
            )
        )
    # if wanted, add here with another config to the callback function
    # , auc_score()],

    trainer.fit(stage_module)


# ToDo: kinda ugly using global stage_module and global trainer I guess???
def apply_pruning(epoch):
    # evil hack, to prune after 0'th epoch --> should be fine!
    # if(epoch == 0):
    #    prune.global_unstructured(
    #        parameters_to_prune,
    #        pruning_method=prune.L1Unstructured,
    #        amount=0.9265,
    #    )
    #    stage_module.last_pruned = epoch
    #    return False

    stage_module.val_loss.append(trainer.callback_metrics["val_loss"].cpu().numpy())
    # print(max(stage_module.val_loss), min(stage_module.val_loss))
    # include feedback from validation loss here
    if (len(stage_module.val_loss) > 50) and (stage_module.last_pruned > -1):
        stage_module.val_loss.pop(0)
        if (
            max(stage_module.val_loss) - min(stage_module.val_loss)
        ) < stage_module.hparams["pruning_val_loss"]:
            stage_module.val_loss = []
            stage_module.last_pruned = epoch
            stage_module.pruned = stage_module.pruned + 1
            # ToDo: set up proper warm up again
            if stage_module.hparams["rewind_lr"]:
                stage_module.lr_schedulers().last_epoch = 0
            stage_module.log("pruned", stage_module.pruned)
            print("pruning val_loss")
            return True
    # simply reached pruning frequency
    if ((epoch - stage_module.last_pruned) % stage_module.hparams["pruning_freq"]) == 0:
        stage_module.val_loss = []
        stage_module.last_pruned = epoch
        stage_module.pruned = stage_module.pruned + 1
        # ToDo: set up proper warm up again
        if stage_module.hparams["rewind_lr"]:
            stage_module.lr_schedulers().last_epoch = 0
        stage_module.log("pruned", stage_module.pruned)
        print("pruning freq")
        return True
    else:
        return False


if __name__ == "__main__":
    main()
