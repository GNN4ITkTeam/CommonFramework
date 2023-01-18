# Common Framework for GNN4ITk Project

This repository contains the framework used for developing, testing and presenting the GNN-based ITk track reconstruction project GNN4ITk. To get started, run the setup commands, then take a look at the examples in the `examples` directory. Instructions and further details about the framework are available under the subdirectory of interest - `examples`, `gnn4itk_cf/stages` or `gnn4itk_cf/core`.

**This repository is still under development and may be subject to breaking changes.**

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
