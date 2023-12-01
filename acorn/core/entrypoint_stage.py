import click
from . import train_stage, infer_stage, eval_stage


@click.command()
@click.argument("command")
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for evaluation"
)
def main(command, config_file, verbose, checkpoint):
    if command == "train":
        train_stage.main(config_file, verbose, checkpoint)
    elif command == "infer":
        infer_stage.main(config_file, verbose, checkpoint)
    elif command == "eval":
        eval_stage.main(config_file, verbose, checkpoint)
    else:
        print(f"Unknown command: {command}")
