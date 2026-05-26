# Example - pruning pipeline

## Setup

For the setup you can follow the steps of any of the other examples, selecting the dataset of your choice.
This example focuses only on the GNN, therefore, follow the steps for graph construction and producing track candidates from respective example.


## Running the Example

The pruning pipeline consists of four stages, with one additional, performing the model compression
```
train and prune --> compress --> infer --> eval
``` 
The first stage focuses on applying pruning and retraining. It is recommended to use already pretrained checkpoint.
```bash
acorn train -c [PATH_TO_CHECKPOINT] gnn_train.yaml
``` 

The training is followed by a model compression stage that physically removes pruned parameters from the model. As an argument pass the path to chosen pruned checkpoint from the previous stage. The script will additionally save the new sizes of the hidden layers within the model, that will be used for the future model construction.
```bash
python scripts/compress_model.py -c [PATH_TO_PRUNED_CHECKPOINT] 
``` 

Standard infer and eval stage support usage of model with irregular dimensions


## Pruning configuration
The implemented pruning pipeline allows application of different pruning configurations

*Importance score*
Importance score is a metric used to assess the importance of parameters (structures) of the model.
- weight magnitude pruning `pruning_score="wmp"` uses the absolute value of the parameter to assess its importance
- gradient magnitude pruning `pruning_score="gmp"` uses product of weight and gradient multiplication to assess its importance, considering the change of the weight value in training


*Importance score aggregation*
For structured pruning, the importance scores for each of the parameters have to be aggregated into a structure. This work supports L1 or L2 norms, available by setting `pruning_aggregation="2"`


*Pruning criteria*
- TopK `pruning_criterion="topk"` removes k parameters with the lowest importance score. For this structured pruning approach k refers to number or fraction of output neurons
- Threshold `pruning_criterion="threshold"` removes parameters (structures) for which the importance score is lower than given threshold


*Pruning frequency*
- one-shot `pruning_frequency="one-shot"` will prune the model only one time. Afterwards, the model will be retrained for N epochs specified by `retraining_epochs="N"`. To disable retraining set `retraining_epochs="0"`
- iterative `pruning_frequency="iterative"` will prune the model every N epochs, specified by `retraining_epochs="N"`. The retraining allows to recover from accuracy loss after removing the parameters. This way, the performance of the model with a given size can be fairly assessed. The iterative approach allows for search for the most optimal model size, balancing the model's accuracy and computational cost.


*Pruning amount*
Defines the amount of parameters (or structures) pruned in each iteration, for example `pruning_amount=0.2`. Can be a function name, if the amount is defined by a function.
There is a generic pruning amount function available, accessible by passing a dictionary to `pruning_amount` parameter. If the current pruning percentage is lower than the pruning percentage theshold, the given pruning amount will be chosen.
For example, for dictionary `{50: 16, 100: 8}` if current pruning percentage is 25%, 16 structures will be pruned, if it's 75% - 8 structures.


The pruning can be interrupted and restarted - the state is saved in the checkpoint file `pruning_iteration` `current_retraining_epoch`