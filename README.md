# Common Framework for GNN4ITk Project

**This repository is still under development and may be subject to breaking changes.**

## Install

To install the GNN4ITK common framework, assuming GPU capability, run

```
conda env create -f gpu_environment.yml
conda activate gnn4itk
pip install -e .
```

otherwise use the `cpu_environment.yml` file.

## First Example: Inference of Pipeline with Module Map, GNN and Connected Components

TODO

## Second Example: Inference of Pipeline with Metric Learning, GNN and Walkthrough

TODO

## Third Example: Training a Custom Pipeline

TODO

## Planning Doc

### Intro
This document is intended to guide the planning, collaboration and implementation of the common codebase of the GNN4ITk project - the effort to replace traditional tracking algorithms with a graph-based pipeline for ATLAS Inner Tracker offline track reconstruction.

Given a set of wishes for how to work together, what do we need to produce to get this done? It may be a single repository, it may be several; it may be a style of working; it may require some pieces of code to radically change to fit the style/format/structure of the common location.

### Goals / Wishlist
- [ ] A reference location for members of the GNN4ITk group to keep updated with each other’s progress
- [ ] It should be easy to add to, easy to navigate, easy to use, easy to understand the logic
- [ ] Any member of the group should be able to easily run the pipeline training with various configurations (module map / metric learning, homo/heterogeneous GNN, iterative / recurrent)
- [ ] Any member should be able to run the pipeline in inference with various configurations - meaning there need to be pretrained models for graph construction, and models for graph edge classification (these will then need to depend on the graph construction, i.e. can choose module_map model for graph construction, then choose  module_map:heteroGNN GNN model, which has been trained on module map graphs)
- [ ] Focus on python implementation for research & development
- [ ] Try to avoid use of C++ directly - if certain code is impossible/impractical to convert to python, then wrap/bind it and call it from python
- [ ] Have consistent interface - a graph can be constructed with multiple techniques, but each use the same class, the same inputs, and the same output format
- [ ] Be well-documented
- [ ] Private for the use of the GNN4ITk group but can freeze certain examples and configurations for public release (using a technique like this one: a public release mirror that can be cited)
- [ ] Be clean - keep messy development and random ideas out of the repository. Can include several different models and configurations, but these should have been tested already elsewhere

### Roadmap for Documentation & Tutorials
- [X] Make password-protected example data available for testing and examples: available at 
https://portal.nersc.gov/cfs/m3443/dtmurnane/GNN4ITk/ExampleData/athena_100_events.zip and https://cernbox.cern.ch/index.php/s/CTBnzv4DlntgHJ7
- [ ] Make a tutorial for how to run Example 1: An inference of a pipeline with module map, GNN and connected components
- [ ] Make a tutorial for how to run Example 2: An inference of a pipeline with metric learning, GNN and walkthrough
- [ ] Make a tutorial for how to run Example 3: Training a custom pipeline

### Roadmap for Codebase
- [ ] Training script written and tested
- [ ] Inference script written and tested
- [ ] Evaluation script written and tested