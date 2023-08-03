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


from .core_utils import str_to_class, get_trainer, get_stage_module


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
    global stage_module
    stage_module, config, default_root_dir = get_stage_module(
        config, stage_module_class, checkpoint_path=checkpoint
    )

    def apply_pruning2(epoch):
        if not config["pruning_allow"]:
            return False

        stage_module.val_loss.append(
            trainer.callback_metrics["val_loss"].cpu().numpy()
        )  # could include feedback from validation or training loss here
        if (len(stage_module.val_loss) > 10) and (stage_module.last_pruned > -1):
            stage_module.val_loss.pop(0)
            if (max(stage_module.val_loss) - min(stage_module.val_loss)) < config[
                "pruning_val_loss"
            ]:
                stage_module.val_loss = []
                stage_module.last_pruned = epoch
                stage_module.pruned = stage_module.pruned + 1
                if config["rewind_lr"]:
                    stage_module.optimizers().param_groups[0]["lr"] = config["lr"]

                stage_module.log("pruned", stage_module.pruned)
                return True
        if ((epoch - stage_module.last_pruned) % config["pruning_freq"]) == 0:
            stage_module.val_loss = []
            stage_module.last_pruned = epoch
            stage_module.pruned = stage_module.pruned + 1
            if config["rewind_lr"]:
                stage_module.optimizers().param_groups[0]["lr"] = config["lr"]
                stage_module.log("pruned", stage_module.pruned)
            return True
        else:
            return False

    if config["quantized_network"]:
        parameters_to_prune = [
            (stage_module.network[1], "weight"),
            (stage_module.network[4], "weight"),
            (stage_module.network[7], "weight"),
            (stage_module.network[10], "weight"),
            (stage_module.network[13], "weight"),
        ]
    else:
        parameters_to_prune = [
            (stage_module.network[0], "weight"),
            (stage_module.network[3], "weight"),
            (stage_module.network[6], "weight"),
            (stage_module.network[9], "weight"),
            (stage_module.network[12], "weight"),
        ]

    trainer = get_trainer(config, default_root_dir)
    if config["pruning_allow"]:
        trainer.callbacks.append(
            ModelPruning(
                pruning_fn=config["pruning_fn"],
                parameters_to_prune=parameters_to_prune,
                amount=config["pruning_amount"],
                apply_pruning=apply_pruning2,
                # settings below only for structured!
                #                pruning_dim = metric_learning_configs["pruning_dim"],
                #                pruning_norm = metric_learning_configs["pruning_norm"],
                #                use_global_unstructured = metric_learning_configs["use_global_unstructured"],
                verbose=1,  # 2 for per-layer sparsity, #1 for overall sparsity
            )
        )

    trainer.fit(stage_module)


if __name__ == "__main__":
    main()
