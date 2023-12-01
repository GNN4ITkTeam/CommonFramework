import click
from . import train_stage, infer_stage, eval_stage


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_file")
@click.option("--checkpoint", "-c", default=None, help="Checkpoint to use for training")
@click.option("--sweep", "-s", default=None, help="Sweep parameter for training")
@click.option(
    "--checkpoint_resume_dir",
    "-r",
    default=None,
    help="Directory to resume checkpoint from",
)
def train(config_file, checkpoint, sweep, checkpoint_resume_dir):
    train_stage.train(config_file, checkpoint, sweep, checkpoint_resume_dir)


@cli.command()
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for inference"
)
def infer(config_file, verbose, checkpoint):
    infer_stage.infer(config_file, verbose, checkpoint)


@cli.command()
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for evaluation"
)
@click.option("--dataset", "-d", default="valset", help="Dataset to use for evaluation")
def eval(config_file, verbose, checkpoint, dataset):
    eval_stage.evaluate(config_file, verbose, checkpoint, dataset)


if __name__ == "__main__":
    cli()
