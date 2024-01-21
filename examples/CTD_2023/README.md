# Connecting The Dots (CTD) 2023 reproducibility

![CTD2023](images/banniere_CTD2023_INDICO-final-try_960-400pixels.png "CTD2023")


This example gives instructions to reproduce the results presented by the GNN4ITk team at Connecting The Dots 2023. 

## Simulated event sample

MC event simulation samples are used for this study, specifically pp collisions at √s = 14 TeV with a tt pair
in the final state, with 200 pp interaction pileup per bunch crossing, as expected at the LH-LHC, and a full
simulation of the ATLAS detector based on Geant4. Updated ITk layout version 23-00-03 was used in the detector simulation.

The simulated data dumped in ROOT format from RDO simulated samples can be download from this container on the grid: 

```bash
user.jstark:GNN4Itk_v2__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3770_s3773_r14431
```

## Setup

Assuming the ACORN requirements have been installed, the only other requirements are the data files and model files. 
The data can be download to a data directory `MY_DATA_DIR``. This is around 2.3Gb, so ensure you download to a location with sufficient space. Define this dir with:

```bash
data_dir=MY_DATA_DIR
```
Create the following sub directories:

```bash
mkdir $data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9
mkdir $data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar
```
then download the data with the following commands (assuming you have configure your ATLAS environnement on the grid):

```bash
cd $data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar
lsetup rucio
voms-proxy-init -voms atlas
rucio download user.jstark:GNN4Itk_v2__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3770_s3773_r14431
```

The location of this data, as well as all parameters controlling the GNN4ITk reconstruction chain, is specified in yaml config files. The data directory currently has a placeholder MY_DATA_DIR. Replace this with the actual data directory with:

```bash
sed -i "s|MY_DATA_DIR|$data_dir|g" *.yaml
```

Assuming ACORN directory currently has a placeholder MY_ACORN_DIR, define it by:

```bash
acorn_dir=MY_ACORN_DIR
```

## Data reading stage

### Sampling the events

The events sampling use for CTD 2023 can be found on `eos` here:

```bash
/eos/user/s/scaillou/CTD_2023/sampling/train_set_ttbar_uncorr.txt
/eos/user/s/scaillou/CTD_2023/sampling/valid_set_ttbar_uncorr.txt
/eos/user/s/scaillou/CTD_2023/sampling/test_set_ttbar_uncorr.txt
```

please copy this files in:

```bash
$data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/sampling/
```

### Inference

Inference will read the dumped data and will: 

- Sample the events in train, val and test sets accordingly with the sampling files
- Represent the event as a pyg Data object
- Store requested hits features and particles (tracks) features
- Preprocess requested features
- Build track edges

Run the command:

```bash
acorn infer $acorn_dir/examples/CTD_2023/data_reader.yaml
```

The result will be pyg files containing the events represented as pyg Data object:
- A trainset of 9.8K events
- A valset of 1K events
- A testset of 1K events

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
$data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/feature_store/
``````
before running the next stages.

---

## Graph construction stage

### Inference

Inference will:
- Construct the graph (i.e. the `event.edge_index`) connecting nodes (representing hits/Space Points) based on a ITK-23-00 version of the Module Map
- Create `event.truth_map` which is the truth map between `event.track edges` and `event.edge_index`.
- Create the target `event.y` (`event.y==False` for fake edges and `event.y==True` for true edges)

The CTD_2023 Module Map can be found here : 

```bash
/eos/user/s/scaillou/CTD_2023/model_store/module_map/Mtriplet_1GeV_3hits_noE__merged__sorted.txt
```

Please copy the Module Map in :

```bash 
MY_ACORN_DIR/model_store/module_map/
```

Then run the command:

```bash
acorn infer $acorn_dir/examples/CTD_2023/module_map_infer.yaml
```

The result is the train, val and test sets of pyg files updated by the `event.edge_index`, the `event.truth_map` and the edges target `event.y`.

### Evaluation

Run the command:

```bash
acorn eval $acorn_dir/examples/CTD_2023/module_map_eval.yaml
```

The result will be the graph construction efficiency plots vs eta and pT:

```bash
$data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/module_map/edgewise_efficiency_eta.png
$data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/module_map/edgewise_efficiency_pt.png
```

## Edge Classifier stage

### Training

The training has to be launch on a GPU interfaced with CUDA and with at least 80GB of memory.  

The `InteractionGNN2` edge classifier GNN model is trained using the train dataset computed on the previous stage. 
Details on the model can be found in [REF] and the code is here : 

```bash
$acorn_dir/gnn4itk_cf/stages/edge_classifier/models/interaction_gnn.py
```

GNN Model and training parameters can be found in the config file here:

```bash
$acorn_dir/examples/CTD_2023/gnn_train.yaml
```
 
Run the command to launch the training:

```bash
acorn train $acorn_dir/examples/CTD_2023/gnn_train.yaml
```

Typically the training will last between ~100 to ~200 epochs to start to plateau significantly.

The hyperparameters of the model and training can be found here:
```bash
$data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn/lightning_logs/version_XXX/hparams.yaml
```
(XXX = run version number)

The log of the training can be found here:
```bash
$data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn/lightning_logs/version_XXX/metrics.csv
```
(XXX = run version number)

The last and the 'best' checkpoints of the model are stored here:

```bash
$data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn/artifacts
```

### Inference

Inference of the GNN model will:
- Predict scores for the edges of each event in testset stored in `event.scores` 
- Create a tensor of boolean `event.pred` (`True` if the edge passes the cut `False` if not)
- Update the `event.truth_map` to take into account true edges which are not passing the cut

#### Inference from the local training best checkpoint

Run the command:

```bash
acorn infer $acorn_dir/examples/CTD_2023/gnn_infer.yaml
```
#### Inference from pre-trained GNN checkpoint

A pre-trained `InteractionGNN2` edge classifier GNN model checkpoint can be found here:

```bash
/eos/user/s/scaillou/CTD_2023/model_store/gnn/GNN_IN2_epochs169.ckpt
```

copy it here:

```bash
$acorn_dir/model_store/gnn/
```

It gives ~99.2% of efficency and ~92.9% of masked purity (see [REF] for masked purity definition) on the test set.

Run the command:

```bash
acorn infer $acorn_dir/examples/CTD_2023/gnn_infer.yaml -c MY_ACORN_DIR/model_store/gnn/GNN_IN2_epochs169.ckpt
```

The result is the scored testset events.

### Evaluation

Run the command:

```bash
acorn eval $acorn_dir/examples/CTD_2023/gnn_eval.yaml
```
The results will be the plots of cumulative efficiency vs (r,z) (graph construction efficiency * gnn signal efficiency), efficiency vs (r,z) (gnn signal efficiency), target purity, purity and total purity vs (r, z):

```bash
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn/cumulative_edgewise_efficiency_rz.png
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn/edgewise_efficiency_rz.png
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn/edgewise_target_purity_rz.png
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn/edgewise_masked_purity_rz.png
MY_DATA_DIR/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/gnn/edgewise_total_purity_rz.png
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
acorn infer $acorn_dir/examples/CTD_2023/track_building_infer.yaml
```

The result is
- the events of testset in pyg format with nodes labelled by the track candidates id
- the track candidates in ASCII files format, which can be found here:
  
```bash
$data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/track/testset/tracks/
```

Each lines of the event files contains a sequence of hits which are the track candidates, for instance:

```bash
head $data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/track/testset_reco/event005000901_reco_trks.txt
11 68 133 185 15921 15995 17197 17261 17324 48877 101589
18 78 866 929 17096 17160 17226 18636 18700 18760 52253 80084 92282 101587 240153 243270 246988
50 112 166 221 900 17238 17298 17359 17424 17490 60676 60730 60780 79951 80005 80139 92235 92345
94 152 208 262 335 1001 17284 17345 17415 17478 60719 60771 60823 61249 79992 80171 92229 92338 92368
97 154 207 260 16038 16115 16184 16247 16321 17341 17411 49068 49147 49227 49343 60129 60194 60259 79784 92134
101 159 214 264 334 945 17342 17412 17472 17570 60763 60814 61293 80166 80211 92355
420 1197 1269 1347 1425 19091 19195 19311 19420 19523 61440 61529 61612
528 598 680 755 1384 1465 1535 1600 17907 18039 18134 19168 19273 19381 19484 19606 19691 19779
535 1322 1394 1470 1540 1611 19171 19275 19385 19488 19610 19696 19783
580 662 734 830 16579 16653 16742 16829 18101 18212 49859 49957 50068 51933 51985 52032
```

### Evaluation

```bash
acorn eval $acorn_dir/examples/CTD_2023/track_building_eval.yaml
```
The result is the plots of track reconstruction efficiency
vs eta and of track reconstruction efficiency
vs pT.

```bash
$data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/track/track_reconstruction_eff_vs_eta.png
$data_dir/ATLAS-P2-ITK-23-00-03_Rel.21.9/ttbar/track/track_reconstruction_eff_vs_pt.png
```