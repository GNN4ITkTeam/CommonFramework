# Connecting The Dots (CTD) 2023 reproducibility

<div align="center">

<img src="images/banniere_CTD2023_INDICO-final-try_960-400pixels.png" width="400px">

</div>


This example gives instructions to reproduce the results presented by the GNN4ITk team at Connecting The Dots 2023. 

## Simulated event sample

MC event simulation samples are used for this study, specifically pp collisions at √s = 14 TeV with a tt pair
in the final state, with 200 pp interaction pileup per bunch crossing, as expected at the LH-LHC, and a full
simulation of the ATLAS detector based on Geant4. Updated ITk layout version 23-00-03 was used in the detector simulation.

The simulated data dumped in ROOT format from RDO simulated samples can be download from this container on the grid: 

```bash
user.jstark:GNN4Itk_v2__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3770_s3773_r14431
```

Before running the example please replace `MY_DATA_DIR` in all the *.yaml config files or set a environment variable accordignly. 

In the following replace `MY_ACORN_DIR` by you local `ACORN` installation directory or set a environment variable accordignly.

## Data reading stage

### Inference

Inference will read the dumped data and will: 

- samplethe events in thetrain, val and test set accordingly with the sampling in the `train_set_ttbar_uncorr.txt`, `test_set_ttbar_uncorr.txt`, and `val_set_ttbar_uncorr.txt` files.
- represent the event as a pyg Data object
- store requested hits features and particles (tracks) features
- preprocess requested features
- build track edges
- data from the dumped root data and build track edges. Results are saved as pyg files.

Run the command:

```bash
g4i-infer MY_ACORN_DIR/examples/CTD_2023/data_reader.yaml
```

The result will be pyg files containing the events represented as pyg Data object:
- a trainset of 9.8K events
- a valset of 1K events
- a testset of 1K events

---
**NOTE**

This stage can take a lot of time and ressources. If you want you can directly get the results of the inference from the three container on the grid:

```bash
user.avallier:ATLAS-P2-ITK-23-00-03_Rel.21.9_feature_store_ttbar_uncorr_v1_trainset
user.avallier:ATLAS-P2-ITK-23-00-03_Rel.21.9_feature_store_ttbar_uncorr_v1_valset
user.avallier:ATLAS-P2-ITK-23-00-03_Rel.21.9_feature_store_ttbar_uncorr_v1_testset
```

Please copy them in:
```bash
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/feature_store_ttbar_uncorr/
``````
before running the next stages.

---

## Graph construction stage

### Inference

Inference will: 
- construct the graph (i.e. the `edge_index` of the graph) connecting nodes (representing hits/Space Points) based on a ITK-23-00 version of the Module Map
- create a `truth_map` between `track edges` and `edge_index`.
- create the target edge label `y` (boolean: `y==False` for fake edges and `y==True` for true edges)

The Module Map can be found here : 

```bash
/eos/atlas/atlascerngroupdisk/perf-idtracking/GNN4ITk/module_maps/MM_23/MMtriplet_1GeV_3hits_noE__merged__sorted.txt
```

Please copy the Module Map in :

```bash 
MY_ACORN_DIR/model_store/
```

Then run the command:

```bash
g4i-infer MY_ACORN_DIR/examples/CTD_2023/module_map_infer.yaml
```

The result is the train, val and test sets of pyg files updated by the `edge_index`, the `truth_map` and the edges target `y`

### Evaluation

Run the command:

```bash
g4i-eval MY_ACORN_DIR/examples/CTD_2023/module_map_eval.yaml
```

The result will be the graph construction efficiency plots vs eta and pT:

```bash
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/module_map_ttbar_uncorr/edgewise_efficiency_eta.png
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/module_map_ttbar_uncorr/edgewise_efficiency_pt.png
```


## Edge Classifier stage

### Training

The training has to be launch on a GPU interfaced with CUDA and with at least 80GB of memory.  

The `InteractionGNN2` is trained using the train dataset computed on the previous stage. 
Details on the model can be found in [REF] and the code here : 

```bash
MY_ACORN_DIR/gnn4itk_cf/stages/edge_classifier/models/interaction_gnn.py
```

Model and training parameters can be found in the config file here:

```bash
MY_ACORN_DIR/examples/CTD_2023/gnn_train.yaml
```
 
Run the command:

```bash
g4i-train MY_ACORN_DIR/examples/CTD_2023/gnn_train.yaml
```

Typically the training will last at least ~100 epochs to start to plateau significantly.

The hyperparameters of the model and training can be found here:
```bash
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn_ttbar_uncorr/lightning_logs/version_XXX/hparams.yaml
```
(XXX = run version number)

The log of the training can be found here:
```bash
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn_ttbar_uncorr/lightning_logs/version_XXX/metrics.csv
```
(XXX = run version number)

The last and the 'best' checkpoints of the model are stored here:

```bash
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn_ttbar_uncorr/artifacts
```

A already trained Edge Classifier GNN model checkpoint can be pull from the ACORN git repo:

```bash
cd MY_ACORN_DIR/model_store
git lfs pull --include "IN2_ep92_eff99.2_pur92.8.ckpt"
```

### Inference

Inference of the GNN model will:
- predict scores for the edges of each event in testset stored in `event.scores` 
- create a tensor of boolean `event.pred` (`True` if the edge passes the cut `False` if not)
- update the `event.truth_map` to take into account true edges which are not passing the cut

Run the command:

```bash
g4i-infer MY_ACORN_DIR/examples/CTD_2023/gnn_infer.yaml
```

The result is the scored testset events.

### Evaluation

Run the command:

```bash
g4i-infer MY_ACORN_DIR/examples/CTD_2023/gnn_eval.yaml
```
The results will be the plots of cumulative efficiency vs (r,z) (graph construction efficiency * gnn signal efficiency), efficiency vs (r,z) (gnn signal efficiency), target purity, purity and total purity vs (r, z):

```bash
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn_ttbar_uncorr/cumulative_edgewise_efficiency_rz.png
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn_ttbar_uncorr/edgewise_efficiency_rz.png
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn_ttbar_uncorr/edgewise_target_purity_rz.png
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn_ttbar_uncorr/edgewise_masked_purity_rz.png
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn_ttbar_uncorr/edgewise_total_purity_rz.png
```

## Track building stage

### Inference

The inference will:
- Apply a cut on edge scores
- Compute Connected Components and identified simple path as track candidates
- Compute a Walk Through algorithm on the remaining part of the graph (not simple path i.e. branching) to desambiguize road and create new track candidates
- Create a `event.label` for each node (hit/Space Point) which is the id of the track candidates the node (hit/Space Point) belongs. 

Run the command:

```bash
g4i-infer MY_ACORN_DIR/examples/CTD_2023/track_building_infer.yaml
```

The result is the labelled events of testset.

### Evaluation

```bash
g4i-eval MY_ACORN_DIR/examples/CTD_2023/track_building_eval.yaml
```
The result is the plots of track reconstruction efficiency
vs eta and of track reconstruction efficiency
vs pT.

```bash
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/cc_and_walk_ttbar_uncorr/track_reconstruction_eff_vs_eta.png
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/cc_and_walk_ttbar_uncorr/track_reconstruction_eff_vs_pt.png
```

### Track generation

```bash
python dump_reco_trks_to_ascii.py -i MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/cc_and_walk_ttbar_uncorr/testset -o MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/cc_and_walk_ttbar_uncorr/tracks/
```