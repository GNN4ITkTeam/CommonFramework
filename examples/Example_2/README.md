# Example 2

## Setup

Assuming the CommonFramework repo requirements have been installed, the only other requirements are the data files. An example dataset can be downloaded to a data directory `MY_DATA_DIR`. This is around 2.3Gb, so ensure you download to a location with sufficient space. Define this dir and your CERN username with
```bash
data_dir=[insert your data directory here]
my_username=[insert your CERN username here]
```
Assuming you have EOS ATLAS group access, download the data (not necessary if you have already downloaded the data for Example 1):
```bash
mkdir $data_dir/Example_2
scp $my_username@lxplus.cern.ch:/eos/user/d/dmurnane/GNN4ITk/FrameworkExamples/Example_1/Example_1_Data.zip $data_dir/Example_2/
unzip $data_dir/Example_2/Example_1_Data.zip -d $data_dir/Example_2
```
Some necessary files used to read the Athena events can be downloaded with
```bash
scp $my_username@lxplus.cern.ch:/eos/user/d/dmurnane/GNN4ITk/FrameworkExamples/Example_1/Example_1_Artifacts.zip $data_dir/Example_2/
unzip $data_dir/Example_1/Example_1_Artifacts.zip -d $data_dir/Example_2
```

The location of this data, as well as all parameters controlling the GNN4ITk reconstruction chain, is specified in `yaml` config files. The data directory currently has a placeholder MY_DATA_DIR. Replace this with the actual data directory with
```bash
sed -i "s|MY_DATA_DIR|$data_dir|g" *.yaml
```

## Running the Example

The following commands will run the Example 1 pipeline. In general, they follow the pattern
```
train --> infer --> eval
``` 
where `train` is used to train a model, `infer` is used to apply the model to data, and `eval` is used to evaluate the performance of the model. If a model has already been trained (in the case of the Module Map in Example 1), we do not need to train it, only provide the model to the `infer` step.

**1.** First, we build our input data from the raw Athena events:
```bash
acorn infer data_reader.yaml
```

**2.** We start the graph construction by training the Metric Learning stage:
```bash
acorn train metric_learning_train.yaml
``` 

**3.** Then, we build graphs using the Metric Learning in inference:
```bash
acorn infer metric_learning_infer.yaml
```

**3a.** (Optional) We can examine the performance of the Metric Learning, by printing some efficiency plots:
```bash
acorn eval metric_learning_eval.yaml
```

**4.** Then, we train the GNN (here we will train on a toy version of the data that only includes pT>1GeV particles - this is configured with the `hard_cuts` option in the `gnn_train.yaml` file):
```bash
acorn train gnn_train.yaml
```

**5.** Once the GNN is trained (should take around half an hour), we apply the GNN in inference to produce a dataset of scored graphs:
```bash
acorn infer gnn_infer.yaml
```

**5a.** (Optional) We can examine the performance of the GNN, by printing some efficiency plots:
```bash
acorn infer gnn_infer.yaml
```

**6.** Finally, we produce track candidates from the scored graphs:
```bash
acorn infer track_building_infer.yaml
```

**7.** And plot the performance
```bash
acorn eval track_building_eval.yaml
```
