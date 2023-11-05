# GNN4ITk Common Framework Examples

This directory contains examples of how to use the GNN4ITk common framework. Since the framework is targeted to being flexible, these examples are suggested to be used as templates for your particular use case. They are also used to freeze explicit configurations for public release.

## Example List

In general, most of these examples place `hard_cuts` on the input data, by removing spacepoints from particles with pT < 1 GeV. This is obviously "cheating", and thus the examples here are toys to quickly train, see some performance plots, and get a feel for the framework. To start a realistic analysis, copy the `Example` directory you're interested in, and remove any `hard_cuts` from stage configurations. Be aware that these examples can run in around an hour with a GPU (CPU training is possible but not well-supported and *very* slow), but that running on full events may take days to train.

The examples are listed below:

1. [ITk Pipeline with Module Map, GNN and Connected Components](./Example_1)
2. [ITk Pipeline with Metric Learning, GNN and Walkthrough](./Example_2)
3. [TrackML Pipeline with Metric Learning, (Optional Filter), GNN and Connected Components)](./Example_3)
4. ~~[Training a Custom Pipeline](./Example_4)~~
5. ~~[Approved ITk V1 Public Release](./Public_Release_V1)~~