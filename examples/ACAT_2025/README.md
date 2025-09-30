# Example yaml files for the metric learning pipeline results for ACAT 2025
The references can be found in https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PLOTS/IDTR-2025-05/ and https://indico.cern.ch/event/1488410/contributions/6561506/.

There are two different methods for graph construction. The simple metric learning and double metric learning. For simple metric learning, use `metric_learning_train.yaml`, `metric_learning_infer.yaml` and `metric_learning_eval.yaml`. For double metric learning, use `double_metric_learning_train.yaml`, `double_metric_learning_infer.yaml` and `double_metric_learning_eval.yaml`. One needs to replace `input_dir` and `stage_dir` and also with correct `data_split`.

Currently, the checkpoints of the pretrained models are available on Perlmutter: `/global/cfs/projectdirs/m3443/data/GNN4ITK/AcornModels`. Information is given in `/global/cfs/projectdirs/m3443/data/GNN4ITK/AcornModels/README.md`.