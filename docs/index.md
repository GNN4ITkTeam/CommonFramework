# ATLAS GNN Tracking

Welcome to the documentation for the ATLAS GNN tracking pipeline.

???+ warning "Work in progress!"

    Note: This site is under construction and may contain incomplete/incorrect information

This repository contains the framework used for developing, testing and presenting the GNN-based ITk track reconstruction project GNN4ITk.

Related work can be found here:
1. https://arxiv.org/abs/2103.06995
2. https://indico.cern.ch/event/948465/contributions/4323753/https://doi.org/10.1051/epjconf/202125103047
3. https://cds.cern.ch/record/2815578?ln=en.

**This repository is still under development and may be subject to breaking changes.**

## Get Started

To get started, run the setup commands (Install instructions section below), then take a look at the examples in the `examples` directory. Instructions and further details about the framework are available under the subdirectory of interest - `examples`, `gnn4itk_cf/stages` or `gnn4itk_cf/core`.
 
## Install

**IMPORTANT! Please use the `dev` branch to run all Examples: it is the latest version and is fully supported!**

To install the GNN4ITK common framework, assuming GPU capability, run

```
git checkout dev
conda env create -f gpu_environment.yml
conda activate gnn4itk
pip install -e .
```

otherwise use the `cpu_environment.yml` file. Note however that CPU performance of the networks in this framework is not guaranteed or optimized.

## Framework Structure

The framework is structured as follows:

- `examples` - contains examples of how to use the framework. Each example is a self-contained directory that contains a set of configuration files. The examples are described in more detail below.
- `gnn4itk_cf/core` - contains the core functionality of the framework, including the `train`, `infer`, and `eval` scripts.
- `gnn4itk_cf/stages` - contains the stages that can be used in the framework. Stages are the building blocks of the framework, and can be combined to create pipelines. Stages are implemented as python classes, and are located in the `gnn4itk_cf/stages` directory. Stages are registered in the `gnn4itk_cf/stages/__init__.py` file, and can be used in the framework by specifying their name in the configuration file used.

## Flow of the Framework

<div align="center">
<figure>
  <a href="https://ibb.co/b76jbjn"><img src="https://i.ibb.co/pn16h6p/stage-simplified.jpg" alt="stage-simplified" border="0" width=500></a><br /><a target='_blank' href='https://imgbb.com/'>Diagram of a generic stage, with the three commands used to run the stage</a><br />
</figure>
</div>

## Examples

### First Example: ITk Pipeline with Module Map, GNN and Connected Components

This example is available in the `examples/Example_1` directory. The example is a simple pipeline that takes raw Athena events as input, and produces a set of track candidates as output. The pipeline consists of three steps:

1. **Module Map** - a python implementation of the Module Map algorithm, which produces a set of graphs from the raw events
2. **GNN** - a graph neural network that scores the edges of the graphs produced by the Module Map
3. **Connected Components** - a simple algorithm that applies a threshold to the scores of the edges, and produces a set of track candidates from the resulting graph

### Second Example: ITk Pipeline with Metric Learning, GNN and Connected Components

This example is available in the `examples/Example_2` directory. The example is a simple pipeline that takes raw Athena events as input, and produces a set of track candidates as output. The pipeline consists of three steps:

1. **Metric Learning** - a python implementation of the metric learning algorithm, which produces a set of graphs from the raw events
2. **GNN** - a graph neural network that scores the edges of the graphs produced by the Module Map
3. **Connected Components** - a simple algorithm that applies a threshold to the scores of the edges, and produces a set of track candidates from the resulting graph

### Third Example: TrackML Pipeline with Metric Learning, Filter, GNN and Connected Components

This example is available in the `examples/Example_3` directory. The example is a simple pipeline that takes raw TrackML events as input, and produces a set of track candidates as output. The pipeline consists of three steps:

1. **Metric Learning** - a python implementation of the metric learning algorithm, which produces a set of graphs from the raw events
2. (Optional). **Filter** - a simple MLP edge classifier to prune down graphs that are too large for GNN training
3. **GNN** - a graph neural network that scores the edges of the graphs produced by the Module Map
4. **Connected Components** - a simple algorithm that applies a threshold to the scores of the edges, and produces a set of track candidates from the resulting graph


### Fourth Example: Reproducing the Results of the GNN4ITk V1 CTD Proceedings (ATL-ITK-PROC-2022-006)

Work in progress!

## CITATION

If this work is useful for your research, please cite our vCHEP2021 and CTD2022 proceedings:

```
@ARTICLE{YourReferenceHere,
       author = {{Ju}, Xiangyang and {Murnane}, Daniel and {Calafiura}, Paolo and {Choma}, Nicholas and {Conlon}, Sean and {Farrell}, Steve and {Xu}, Yaoyuan and {Spiropulu}, Maria and {Vlimant}, Jean-Roch and {Aurisano}, Adam and {Hewes}, Jeremy and {Cerati}, Giuseppe and {Gray}, Lindsey and {Klijnsma}, Thomas and {Kowalkowski}, Jim and {Atkinson}, Markus and {Neubauer}, Mark and {DeZoort}, Gage and {Thais}, Savannah and {Chauhan}, Aditi and {Schuy}, Alex and {Hsu}, Shih-Chieh and {Ballow}, Alex},
        title = "{Performance of a Geometric Deep Learning Pipeline for HL-LHC Particle Tracking}",
      journal = {arXiv e-prints},
     keywords = {Physics - Data Analysis, Statistics and Probability, Computer Science - Machine Learning, High Energy Physics - Experiment},
         year = 2021,
        month = mar,
          eid = {arXiv:2103.06995},
        pages = {arXiv:2103.06995},
          doi = {10.48550/arXiv.2103.06995},
archivePrefix = {arXiv},
       eprint = {2103.06995},
 primaryClass = {physics.data-an},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210306995J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

```
@article{YourReferenceHere,
	author = {{Biscarat, Catherine} and {Caillou, Sylvain} and {Rougier, Charline} and {Stark, Jan} and {Zahreddine, Jad}},
	title = {Towards a realistic track reconstruction algorithm based on graph neural networks for the HL-LHC},
	DOI= "10.1051/epjconf/202125103047",
	url= "https://doi.org/10.1051/epjconf/202125103047",
	journal = {EPJ Web Conf.},
	year = 2021,
	volume = 251,
	pages = "03047",
}
```
```
@techreport{YourReferenceHere,
      author        = "Caillou, Sylvain and Calafiura, Paolo and Farrell, Steven
                       Andrew and Ju, Xiangyang and Murnane, Daniel Thomas and
                       Rougier, Charline and Stark, Jan and Vallier, Alexis",
      collaboration = "ATLAS",
      title         = "{ATLAS ITk Track Reconstruction with a GNN-based
                       pipeline}",
      institution   = "CERN",
      reportNumber  = "ATL-ITK-PROC-2022-006",
      address       = "Geneva",
      year          = "2022",
      url           = "https://cds.cern.ch/record/2815578",
}
```

If you use this code in your work, please cite the gnn4itk framework:

```
@misc{YourReferenceHere,
author = {Atkinson, Markus Julian and Caillou, Sylvain and Clafiura, Paolo and Collard, Christophe and Farrell, Steven Andrew and Huth, Benjamin and Ju, Xiangyang and Liu, Ryan and Minh Pham, Tuan and Murnane, Daniel (corresponding author) and Neubauer, Mark and Rougier, Charline and Stark, Jan and Torres, Heberth and Vallier, Alexis},
title = {gnn4itk},
url = {https://github.com/GNN4ITkTeam/CommonFramework}
}
```

## Developing

(Optional)

Pre-commit hooks are available for running linting and code formatting. To set them up, run
```
pre-commit install
pre-commit run
``
