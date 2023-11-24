submit_scr=/global/cfs/cdirs/m3443/usr/pmtuan/commonframework/batch_script/pm_submit_batch_1gpu.sh

submit_command="sbatch $submit_scr"

$submit_command configs/weight1_small_graphs_S_batchnorm.yaml --checkpoint_resume_dir 14936798/
$submit_command configs/weight1_small_graphs_S.yaml --checkpoint_resume_dir 14692980/
$submit_command configs/interaction_gnn2_with_pyg_gcn.yaml --checkpoint_resume_dir 15026721/
$submit_command configs/weight1_small_graphs_withGCN_batchnorm.yaml --checkpoint_resume_dir 14999421/
$submit_command configs/sylvains_config_pyg_prop_loss.yaml --checkpoint_resume_dir 14998153/
$submit_command configs/sylvains_config_pyg.yaml --checkpoint_resume_dir 14659987/
$submit_command configs/sylvains_config.yaml --checkpoint_resume_dir 14649209/



