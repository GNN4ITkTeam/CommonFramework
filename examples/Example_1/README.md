# Example 1

## Quickstart

Assuming the CommonFramework repo requirements have been installed, the only other requirements are the data files and model files. An example dataset can be downloaded with (this is around 2.3Gb, so ensure you download to a location with sufficient space):
```bash
wget https://cernbox.cern.ch/index.php/s/CTBnzv4DlntgHJ7/download -O athena_100_events.zip
unzip athena_100_events.zip
```
and enter the password provided by the GNN4ITk team. The model files can be downloaded with
```bash
wget https://cernbox.cern.ch/index.php/s/Y72LOTgxUbP9mio/download -O Example_1.zip
unzip Example_1.zip
```
and enter the password provided by the GNN4ITk team. The location of these files should be specified in the `yaml` config files below, before running each step.

The following commands will run the Example 1 pipeline.

**1.** First, we build our input data from the raw Athena events:
```bash
g4i-infer data_reader.yaml
```

**2.** Then, we build graphs using the python implementation of the Module Map:
```bash
g4i-infer module_map_infer.yaml
```

**2a** (Optional) We can examine the performance of the Module Map, by printing some efficiency plots:
```bash
g4i-eval module_map_eval.yaml
```

**3.** Then, we train the GNN (here we will train on a toy version of the data that only includes pT>1GeV particles - this is configured with the `hard_cuts` option in the `gnn_train.yaml` file):
```bash
g4i-train gnn_train.yaml
```

**4.** Once the GNN is trained (should take around half an hour), we apply the GNN in inference to produce a dataset of scored graphs:
```bash
g4i-infer gnn_infer.yaml
```

**5.** Finally, we produce track candidates from the scored graphs:
```bash
g4i-infer track_cand.yaml
```

**6.** And plot the performance
```bash
g4i-eval track_cand.yaml
```