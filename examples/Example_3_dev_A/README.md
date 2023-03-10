# Example 3

## Setup

Assuming the CommonFramework repo requirements have been installed, the only other requirements are the data files. 

<!-- TODO: Add TrackML data instructions -->

The location of this data, as well as all parameters controlling the GNN4ITk reconstruction chain, is specified in `yaml` config files. The data directory currently has a placeholder MY_DATA_DIR. Replace this with the actual data directory with
```bash
sed -i "s/MY_DATA_DIR/$data_dir/g" *.yaml
```

## Running the Example

The following commands will run the Example 1 pipeline. In general, they follow the pattern
```
train --> infer --> eval
``` 
where `train` is used to train a model, `infer` is used to apply the model to data, and `eval` is used to evaluate the performance of the model. If a model has already been trained (in the case of the Module Map in Example 1), we do not need to train it, only provide the model to the `infer` step.

**1.** First, we build our input data from the raw Athena events:
```bash
g4i-infer data_reader.yaml
```

**2.** We start the graph construction by training the Metric Learning stage:
```bash
g4i-train metric_learning_train.yaml
``` 

**3.** Then, we build graphs using the Metric Learning in inference:
```bash
g4i-infer metric_learning_infer.yaml
```

**3a.** (Optional) We can examine the performance of the Metric Learning, by printing some efficiency plots:
```bash
g4i-eval metric_learning_eval.yaml
```

**4.** Then, we train the GNN (here we will train on a toy version of the data that only includes pT>1GeV particles - this is configured with the `hard_cuts` option in the `gnn_train.yaml` file):
```bash
g4i-train gnn_train.yaml
```

**5.** Once the GNN is trained (should take around half an hour), we apply the GNN in inference to produce a dataset of scored graphs:
```bash
g4i-infer gnn_infer.yaml
```

**6.** Finally, we produce track candidates from the scored graphs:
```bash
g4i-infer track_building_infer.yaml
```

**7.** And plot the performance
```bash
g4i-eval track_building_eval.yaml
```
