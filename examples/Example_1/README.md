# Example 1

## Setup

Assuming the CommonFramework repo requirements have been installed, the only other requirements are the data files and model files. An example dataset can be downloaded to a data directory `MY_DATA_DIR`. This is around 2.3Gb, so ensure you download to a location with sufficient space. Define this dir with
```bash
data_dir=MY_DATA_DIR
```
then download the data with (assuming you have EOS ATLAS group access)
```bash
mkdir $data_dir/Example_1
scp MY_USERNAME@lxplus.cern.ch:/eos/user/d/dmurnane/GNN4ITk/FrameworkExamples/Example_1/Example_1_Data.zip $data_dir/Example_1/
unzip $data_dir/Example_1/Example_1_Data.zip -d $data_dir/Example_1
```
The model files can be downloaded with
```bash
scp MY_USERNAME@lxplus.cern.ch:/eos/user/d/dmurnane/GNN4ITk/FrameworkExamples/Example_1/Example_1_Artifacts.zip $data_dir/Example_1/
unzip $data_dir/Example_1/Example_1_Artifacts.zip -d $data_dir/Example_1
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

**2.** Then, we build graphs using the python implementation of the Module Map:
```bash
acorn infer module_map_infer.yaml
```

**2a.** (Optional) We can examine the performance of the Module Map, by printing some efficiency plots:
```bash
acorn eval module_map_eval.yaml
```

**3.** Then, we train the GNN (here we will train on a toy version of the data that only includes pT>1GeV particles - this is configured with the `hard_cuts` option in the `gnn_train.yaml` file):
```bash
acorn train gnn_train.yaml
```

**4.** Once the GNN is trained (should take around half an hour), we apply the GNN in inference to produce a dataset of scored graphs:
```bash
acorn infer gnn_infer.yaml
```

**4a.** (Optional) Evaluate the GNN performance:
```bash
acorn eval gnn_eval.yaml
```

**5.** Finally, we produce track candidates from the scored graphs:
```bash
acorn infer track_building_infer.yaml
```

**6.** And plot the performance
```bash
acorn eval track_building_eval.yaml
```

## Understanding the Example

This example pipeline was first proposed in *C. Biscarat, S. Caillou, C. Rougier, J. Stark & J. Zahreddine* [arxiv:2103.00916](https://arxiv.org/abs/2103.00916). It uses a data-driven "module map" to assign the possibility of triplets of track hits moving through each module to every other module in the ITk detector. This is used for graph construction, the output of which is subsequently used to train an Interaction Network edge classifier. Once edges are classified, we place a score cut (between 0 and 1, which should be manually tuned to get the best performance) in the track building stage, and label all components that remain connected after the score cut with a unique track label. 