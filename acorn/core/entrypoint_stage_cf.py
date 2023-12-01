import click
from . import train_stage, infer_stage, eval_stage


@click.command()
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")
@click.option(
    "--checkpoint", "-c", default=None, help="Checkpoint to use for evaluation"
)
@click.pass_context
def main(ctx, config_file, verbose, checkpoint):
    # print(ctx.info_name)
    if ctx.info_name == "g4i-train":
        train_stage.main(config_file, verbose, checkpoint)
    elif ctx.info_name == "g4i-infer":
        infer_stage.main(config_file, verbose, checkpoint)
    elif ctx.info_name == "g4i-eval":
        eval_stage.main(config_file, verbose, checkpoint)
    else:
        print(f"Unknown command: {ctx.info_name}")
