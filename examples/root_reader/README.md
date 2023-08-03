# Example of how to run the files dumped from Athena

Usage
```
g4-infer data_reader_root.yaml
```

In the `.yaml` config file, you need to set:
*  `input_dir` : parameter corresponding to the folder where the root files are stored
* `input_sets` for `train`, `valid` and `test` : `txt` files listing the `run_number`, `event_number`, TTree entry, root file name for each of the three samples (made with the package [samples_managing](https://gitlab.cern.ch/gnn4itkteam/samples_managing)

If you want to use only one file per sample edit `data_reader_root.yaml`, to use the `one_*_set_ttbat_uncorr.txt` files, or for a reduced number of files (80/10/10 files) use the `red_*_set_ttbar_uncorr.txt` instead.



