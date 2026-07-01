# ColliderML Example — Data Reading

This example covers the data reading stage, which converts raw ColliderML parquet files into PyTorch Geometric `.pyg` event files consumed by downstream pipeline stages.

## Data

Download the ColliderML dataset (Hugging Face layout) and set `tracker_hits_dir` and `particles_dir` in `data_reader.yaml` to point at the respective subdirectories. The expected directory layout is described at the top of that file.

## Usage

```bash
acorn infer data_reader.yaml
```

Processed events are written to `stage_dir` as defined in the config, split into `trainset/`, `valset/`, and `testset/` subdirectories. A `dataset_split.csv` summary of the train/val/test assignment is saved alongside them.
