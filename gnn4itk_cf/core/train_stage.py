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

from gnn4itk_cf.utils import str_to_class, get_default_root_dir, get_trainer

@click.command()
@click.argument("config_file")

def main(config_file):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """
    train(config_file)

def train(config_file):
    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)
    # load stage
    stage = config["stage"]
    model = config["model"]
    stage_module = str_to_class(stage, model)(config)

    # setup stage
    stage_module.setup(stage="fit")
    os.makedirs(config["stage_dir"], exist_ok=True)

    # run training, depending on whether we are using a Lightning trainable model or not
    if isinstance(stage_module, LightningModule):
        lightning_train(config, stage_module)
    else:
        stage_module.train()

def lightning_train(config, stage_module):

    default_root_dir = get_default_root_dir()
    trainer = get_trainer(config, default_root_dir)
    trainer.fit(stage_module)

if __name__ == "__main__":
    main()