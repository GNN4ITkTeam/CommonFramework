# Example 1

## Setup

Assuming the CommonFramework repo requirements have been installed, the only other requirements are the data files and model files. An example dataset can be downloaded to a data directory `MY_DATA_DIR`. This is around 2.3Gb, so ensure you download to a location with sufficient space. Define this dir with
```bash
data_dir=MY_DATA_DIR
```
then download the data with
```bash
mkdir $data_dir/Example_1
wget https://cernbox.cern.ch/remote.php/dav/public-files/AREZqMSHGrWMIjc/athena_100_events.zip -O $data_dir/Example_1/athena_100_events.zip
unzip $data_dir/Example_1/athena_100_events.zip -d $data_dir/Example_1
```
and enter the password provided by the GNN4ITk team. The model files can be downloaded with
```bash
wget https://cernbox.cern.ch/remote.php/dav/public-files/uUCHDnGUiHdhsyl/Example_1.zip -O $data_dir/Example_1/Example_1.zip
unzip $data_dir/Example_1/Example_1.zip -d $data_dir/Example_1
```
and enter the password provided by the GNN4ITk team. The location of this data, as well as all parameters controlling the GNN4ITk reconstruction chain, is specified in `yaml` config files. The data directory currently has a placeholder MY_DATA_DIR. Replace this with the actual data directory with
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

**2.** Then, we build graphs using the python implementation of the Module Map:
```bash
g4i-infer module_map_infer.yaml
```

**2a.** (Optional) We can examine the performance of the Module Map, by printing some efficiency plots:
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
g4i-infer track_building_infer.yaml
```

**6.** And plot the performance
```bash
g4i-eval track_building_eval.yaml
```

## Understanding the Example
