# Hardcode the following datatypes for the athena data
SPACEPOINTS_DATATYPES = {
    'cluster_index_1': "int64",
    'cluster_index_2': "int64",
    'hardware': "str",
    'cluster_x_1': "float32",
    'cluster_y_1': "float32",
    'cluster_z_1': "float32",
    'cluster_x_2': "float32",
    'cluster_y_2': "float32",
    'cluster_z_2': "float32",
    'barrel_endcap': "int32",
    'layer_disk': "int32",
    'eta_module': "int32",
    'phi_module': "int32",
    'eta_angle_1': "float32",
    'phi_angle_1': "float32",
    'eta_angle_2': "float32",
    'phi_angle_2': "float32",
    'norm_x_1': "float32",
    'norm_y_1': "float32",
    'norm_z_1': "float32",
    'norm_x_2': "float32",
    'norm_y_2': "float32",
    'norm_z_2': "float32",
    'count_1': "int32",
    'count_2': "int32",
    'charge_count_1': "int32",
    'charge_count_2': "int32",
    'loc_eta_1': "float32",
    'loc_eta_2': "float32",
    'loc_phi_1': "float32",
    'loc_phi_2': "float32",
    'localDir0_1': "float32",
    'localDir0_2': "float32",
    'localDir1_1': "float32",
    'localDir1_2': "float32",
    'localDir2_1': "float32",
    'localDir2_2': "float32",
    'glob_eta_1': "float32",
    'glob_eta_2': "float32",
    'glob_phi_1': "float32",
    'glob_phi_2': "float32",
}

PARTICLES_DATATYPES = {
    'particle_id': "int64",
    'subevent': "int64",
    'barcode': "int64",
    'px': "float32",
    'py': "float32",
    'pz': "float32",
    'pt': "float32",
    'eta': "float32",
    'vx': "float32",
    'vy': "float32",
    'vz': "float32",
    'radius': "float32",
    'status': "int32",
    'charge': "float32",
    'pdgId': "int32",
    'pass': "str",
    'vProdNIn': "int32",
    'vProdNOut': "int32",
    'vProdStatus': "int32",
    'vProdBarcode': "int64",
}