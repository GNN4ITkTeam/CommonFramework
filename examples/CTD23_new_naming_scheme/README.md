# New Variable Naming Scheme (adding prefix to variable name)

This example gives instructions to adapt to the new variable naming scheme in the pyg objects. The new naming scheme adds a prefix of either `hit_`, `edge_` or `track_` to each variable name in order to accurately reflect the variable type. The example configuration yaml files are a copy of the CTD2023 example, but with the changes that are made to adapt to the new variable naming scheme.

In order to adapt to the new variable naming scheme, users need to change the configuration yaml files. These include:

1. Add `hit_` to all node-like variables; add `edge_` to all edge-like variables; add `track_` to all track-like variables. Note that for track-like variables that correspond to the particle truths, they should be added with `track_particle_` instead of `track_` (e.g. `pt` -> `track_particle_pt`).

2. Set the flag `variable_with_prefix` to `true`. If `variable_with_prefix` is set to `false` (current default), the code will execute with backward compatibility, and automatically convert all variable names in the input pyg files, in the config yaml files and in the model checkpoints (if used) to the new naming scheme. It will also convert them back to the old naming scheme in the output pyg files for backward compatibility. If `variable_with_prefix` is set to `true`, no conversion will be made. In this case, users need to make sure all the configuration yaml files, the input pyg objects, and the model checkpoints are already with the new naming scheme.

Some additional features are also added to make it easier for users to transition from old naming scheme to the new scheme:

1. The flag `add_variable_name_prefix_in_pyg` can be set to `true` along with `variable_with_prefix` set to `true`. In this case, the code will convert the variable names in the input pyg objects. This is useful when a new configuration yaml file (with new naming scheme) is prepared, but the input pyg files are produced with the old naming scheme. Note that with this setting, the output pyg objects will be with the new naming scheme (variable names won't be converted back).

2. The flag `add_variable_name_prefix_in_ckpt` can be set to `true` along with `variable_with_prefix` set to `true`. In this case, the code will convert the variable names in the model checkpoints. This is useful when a new configuration yaml file (with new naming scheme) is prepared, but the model checkpoints are produced with the old naming scheme. Note that with this setting, the output pyg objects will be with the new naming scheme (variable names won't be converted back).

3. If users need to rerun the data reading stage in order to produce the input pyg objects with the new naming scheme, the csv conversion step doesn't need to be rerun, and only the csv to pyg step needs to be rerun. In this case, users can set the flag `skip_csv_conversion` to `true` in the data reader yaml and rerun the data reading stage (need to first remove the existing pyg files).

In summray:

- If you want everything to stay like the old days (old naming scheme in both inputs and outputs), set `variable_with_prefix: false` (other flags are irrelevant in this case)

- If all your inputs (pyg + ckpt + yaml) are already with the new naming scheme and you want your outputs to be with the new naming scheme as well, set `variable_with_prefix: true` (other flgas can be set to `false`)

- If all but pyg are with the new naming scheme, set `variable_with_prefix: true` + `add_variable_name_prefix_in_pyg: true`

- If all but ckpt are with the new naming scheme, set `variable_with_prefix: true` + `add_variable_name_prefix_in_ckpt: true`

- If both pyg and ckpt are with the old naming scheme, but yaml is with the new naming scheme, set `variable_with_prefix: true` + `add_variable_name_prefix_in_pyg: true` + `add_variable_name_prefix_in_pyg: true`
