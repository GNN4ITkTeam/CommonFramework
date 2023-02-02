# Common Framework for GNN4ITk Project

This repository contains the framework used for developing, testing and presenting the GNN-based ITk track reconstruction project GNN4ITk. 

Related work can be found here:

1. https://indico.cern.ch/event/948465/contributions/4323753/https://doi.org/10.1051/epjconf/202125103047
2. https://cds.cern.ch/record/2815578?ln=en.

**This repository is still under development and may be subject to breaking changes.**

## Get Started

To get started, run the setup commands (Install instructions section below), then take a look at the examples in the `examples` directory. Instructions and further details about the framework are available under the subdirectory of interest - `examples`, `gnn4itk_cf/stages` or `gnn4itk_cf/core`.

## Install

To install the GNN4ITK common framework, assuming GPU capability, run

```
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

If you use this code in your work, please cite the gnn4itk framework:

@misc{YourReferenceHere,
author = {Murnane, Daniel and Caillou, Sylvain and Clafiura, Paolo and Stark, Jan and Vallier, Alexis and Rougier, Charline and Torres, Heberth and Collard, Christophe and Farrell, Steven Andrew and Ju, Xiangyang and Liu, Ryan and Minh Pham, Tuan and Neubauer, Mark and Atkinson, Markus Julian and Huth, Benjamin},
title = {gnn4itk},
url = {https://github.com/GNN4ITkTeam/CommonFramework}
}




