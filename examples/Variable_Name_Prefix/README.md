# New Variable Naming Scheme (adding prefix to variable name)

This example gives instructions to adapt to the new variable naming scheme in the pyg objects. The new naming scheme adds a prefix of either `hit_`, `edge_` or `track_` to each variable name in order to accurately reflect the variable type. The example configuration yaml files are a copy of the CTD2023 example, but with the changes that are made to adapt to the new variable naming scheme.

In order to adapt to the new variable naming scheme, users need to change the configuration yaml files. These include:

1. Add `hit_` to all node-like variables; add `edge_` to all edge-like variables; add `track_` to all track-like variables. Note that for track-like variables that correspond to the particle truths, they should be added with `track_particle_` instead of `track_` (e.g. `pt` -> `track_particle_pt`).

2. Set the flag `variable_with_prefix` to `true`. If `variable_with_prefix` is set to `false` (current default), the code will execute with backward compatibility, and automatically convert all variable names in the input pyg objects, and in the config yaml files to new naming scheme. It will also convert them back to the old naming scheme in the output pyg format for backward compatibility. If `variable_with_prefix` is set to `true`, no conversion will be made. In this case, users need to make sure both the configuration yaml files and the input pyg objects are already with the new naming scheme.

Some additional features are also added to make it easier for users to transition from old naming scheme to the new scheme:

1. The flag `add_variable_name_prefix` can be set to `true` along with `variable_with_prefix` set to `true`. In this case, the code will convert the variable names in the input pyg objects. This is useful when a new configuration yaml file (with new naming scheme) is prepared, but the input pyg files are produced with the old naming scheme. Note that with this setting, the output pyg objects will be with the new naming scheme (variable names won't be converted back).

2. If users need to rerun the data reading stage in order to produce the input pyg objects with the new naming scheme, the csv conversion step doesn't need to be rerun, and only the csv to pyg step needs to be rerun. In this case, users can set the flag `skip_csv_conversion` to `true` in the data reader yaml and rerun the data reading stage (need to first remove the existing pyg files).

