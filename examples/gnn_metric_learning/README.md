# Recursive GAT for Object Condensation

This is the directory that contains examples for the one-shot object condensation for tracking using recursive GAT. Before running these examples, please be sure to MODIFY ALL OUTPUT DIRECTORIES (namely `stage_dir`).

The input data are stored in `/global/cfs/cdirs/m2616/data/GNN4ITK/trackml_gnnml_feature_store`.

To train the recursive GAT, run
```
acorn train gnn_ml_train.yaml
```

To run the inference, run
```
acorn infer gnn_ml_infer.yaml -c PATH_TO_CHECKPOINT
```

To run the evaluation, run
```
acorn eval gnn_ml_eval.yaml -c PATH_TO_CHECKPOINT
```