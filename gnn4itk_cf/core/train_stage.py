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

from pytorch_lightning import LightningModule

from gnn4itk_cf.utils import str_to_class, get_trainer, get_stage_module

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

    stage_module, config, default_root_dir = get_stage_module(config, stage_module_class, checkpoint_path=checkpoint)
    trainer = get_trainer(config, default_root_dir)
    trainer.fit(stage_module)

if __name__ == "__main__":
    main()