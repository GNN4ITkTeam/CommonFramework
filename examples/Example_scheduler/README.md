# Learning-rate scheduler example

[scheduler_step.yaml](scheduler_step.yaml) is a scheduler configuration fragment to copy into a
stage training config. It uses optimizer-step warmup and StepLR decay; it is
not a complete input to `acorn train`.

## Configuration

`scheduler_interval` selects `epoch` (the default) or `step`. Use the same
`warmup` option in either mode: it counts epochs or optimizer updates,
respectively. Omit it or set it to `null` or `0` to disable warmup.
With gradient accumulation, a step is an optimizer update, not a training batch.

For `StepLR`, `patience` is the number of selected intervals between decays,
and `factor` multiplies the learning rate at each decay. For
`CosineAnnealingWarmRestarts`, `patience` is the initial cycle length in those
same units; subsequent cycles double in length. `ReduceLROnPlateau` supports
only `epoch` and retains its metric-based patience and epoch warmup behavior.

```yaml
scheduler: StepLR
scheduler_interval: step
warmup: 4
patience: 6
lr: 0.01
factor: 0.5
min_lr: 0.0
```

This warms up over four optimizer updates, then runs six updates at `0.01`
before the first decay to `0.005`. Remove `scheduler_interval` to use the same
durations in epochs. `max_epochs` still counts epochs; `val_check_interval`
is independent and accepts a fraction of an epoch or an integer number of
training batches (default `1.0`, once per epoch).

StepLR and cosine now run warmup before starting the decay schedule, including
in epoch mode. Previously the decay scheduler advanced during warmup, so
existing epoch configurations with warmup can produce different LR sequences.
Cosine now uses `min_lr` as its minimum, defaulting to `1e-8` when omitted.
The earlier MR-only `warmup_steps` option is replaced by `warmup`.

## Test locally

From the repository root, run the CPU smoke test in an environment with the
project and pytest installed:

```bash
uv run python -m pytest tests/utils/test_trainer_schedule.py -k scheduler_yaml_example -q
```

The test loads this YAML and uses synthetic data, with no dataset or GPU
required. It checks step mode with gradient accumulation and default epoch
mode.
