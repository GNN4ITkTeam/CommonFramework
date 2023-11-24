
infer_template=/global/cfs/cdirs/m3443/usr/pmtuan/staged_ctf/itk/examples/ttbar/track_building/track_building_infer_template
# strict_eval_template=/global/cfs/cdirs/m3443/usr/pmtuan/commonframework/examples/uncorr_2023/track_building/igcn/strict_track_building_eval
eval_template=/global/cfs/cdirs/m3443/usr/pmtuan/staged_ctf/itk/examples/ttbar/track_building/track_building_eval_template
 
config_dir=/global/cfs/cdirs/m3443/usr/pmtuan/staged_ctf/itk/examples/ttbar/track_building/scan_cccut
mkdir -vp $config_dir

for ccut in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo $ccut
    infer_config=${config_dir}/infer_$ccut.yaml
    eval_config=${config_dir}/eval_$ccut.yaml
    # strict_eval_config=${strict_eval_template}_$ccut.yaml

    cp ${infer_template}.yaml $infer_config
    cp ${eval_template}.yaml $eval_config
    # cp ${strict_eval_template}_template.yaml $strict_eval_config

    sed -i "s/CCCUT/$ccut/g" $infer_config
    sed -i "s/CCCUT/$ccut/g" $eval_config
    # sed -i "s/CCCUT/$ccut/g" $strict_eval_config
    sbatch /global/cfs/cdirs/m3443/usr/pmtuan/staged_ctf/itk/batch_script/submit_infer_eval_track_building.sh $infer_config $eval_config 
    # sbatch /global/cfs/cdirs/m3443/usr/pmtuan/commonframework/batch_script/submit_eval_track_building.sh $eval_config $strict_eval_config
done