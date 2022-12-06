"""Module for processing Athena-generated data.

This module takes Athena-generated events and processes it into a more standard form, including merging the various pieces of spacepoint and cluster data. This version handles the case of events scattered throughout multiple directories.

Example:
    Run `python process_events.py process_config.yaml` to process the data specified in the config file. 
    
Attributes:
    - num_workers (int): How many processes to run on
    - overwrite (bool): Whether to overwrite existing data that shares the file name
    
Todo:
    * Merge the multi-directory module with the single-directory module, and have the directory structure be a choice
"""

#!/usr/bin/env python
import glob
import os
import sys
import re
import yaml
import csv
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

sys.path.append("../")
from utils.processing_utils import (read_particles, convert_barcodes, 
                                    read_spacepoints, read_clusters, get_reconstructable_particles, truth_match_clusters,
                                    merge_spacepoints_clusters)

def get_file_names(file, inputdir):   
    
    evtid = re.findall("[0-9]+", file)[-1]
    subdir = os.path.dirname(file)
        
    cluster_fname = os.path.join(inputdir, subdir, "clusters_evt{}.txt".format(evtid))
    particle_fname = os.path.join(inputdir, subdir, "particles_evt{}.txt".format(evtid))
    hit_fname = os.path.join(inputdir, subdir, "spacepoints_evt{}.txt".format(evtid))

    return cluster_fname, particle_fname, hit_fname


def get_new_evtid(file, inputdir, evtId_offset=0,num_events=None):

    """
    Here we handle whether the file is from a subdirectory or not, and thus whether to include the 
    subdirectory ID in the event ID
    """

    subdir_depth = len(os.path.relpath(file, inputdir).split("/")) 
    
    m = re.findall("[0-9]+", file)
    id_length = len(str(num_events)) if num_events else len(m[-1])
    
    if subdir_depth > 1:
        new_evtid = m[-1] + "{:0{id_length}}".format(int(m[-1]), id_length=id_length)
    else:
        new_evtid = m[-1]
    
    return int(new_evtid)+evtId_offset

def process_event(file, inputdir, outputdir, particles_datatypes, spacepoints_datatypes, debug,evtId_offset, overwrite=False, num_events=None, **kwargs):

    cluster_fname, particle_fname, hit_fname = get_file_names(file, inputdir)
    evtid = get_new_evtid(file, inputdir,evtId_offset, num_events)

    # Check if all files are there
    if (
        not os.path.exists(cluster_fname)
        or not os.path.exists(particle_fname)
        or not os.path.exists(hit_fname)
    ):
        print(evtid, "does not exist")
        return

    # Read in particles
    particles = read_particles(particle_fname)

    
    # Build unique barcodes and fix particle datatypes
    particles = convert_barcodes(particles)
    particles = particles.astype(particles_datatypes)
    #print(particles)
    tot_particles = particles.shape[0]

    
    # Read space point measurements
    pixel_hits, strip_hits = read_spacepoints(hit_fname)

    # Get hit statistics
    num_hits_pixel = pixel_hits.shape[0]
    num_hits_sct = strip_hits.shape[0]
    tot_hits = num_hits_pixel + num_hits_sct
    # Read cluster info
    clusters = read_clusters(cluster_fname, particles)


    nClusters =  clusters.groupby("particle_id")["cluster_id"].count()
    nClusters= nClusters.reset_index()
    nClusters = nClusters.rename(columns={"cluster_id": "nClusters"})

    particles = pd.merge(particles, nClusters, on='particle_id').fillna(method='ffill')
    reco_particles = get_reconstructable_particles(particles)
    # Build truth list of spacepoints by handling matching clusters
    truth_spacepoints = truth_match_clusters(pixel_hits, strip_hits, clusters)
    #print(truth_spacepoints)
    # Tidy up the truth dataframe and add in all cluster information
    truth_spacepoints["cluster_index_2"] = truth_spacepoints["cluster_index_2"].fillna(-1)
    
    truth_spacepoints = merge_spacepoints_clusters(truth_spacepoints, clusters)
    
    # Fix spacepoint datatypes
    truth_spacepoints = truth_spacepoints.astype(spacepoints_datatypes)
    
    # Get reconstruction statistics
    num_reco_particles = reco_particles.shape[0]
    num_true_hits = truth_spacepoints.particle_id.isin(reco_particles.particle_id).sum()

    if debug:
        print("---Event {:,}----".format(evtid))
        print("Total {:,} hits".format(tot_hits))
        print(
            "Total {:,} particles, out of which {} are of interest: {:.4f}%.".format(
                tot_particles,
                num_reco_particles,
                num_reco_particles * 100 / tot_particles,
            )
        )
        print(
            "{} hits in PIXEL: {:.1f}% ".format(
                num_hits_pixel, 100 * num_hits_pixel / tot_hits
            )
        )
        print(
            "{} hits in SCT:   {:.1f}% ".format(
                num_hits_sct, 100 * num_hits_sct / tot_hits
            )
        )
        print(
            "Expecting total {:,} cluster info".format(
                num_hits_pixel + num_hits_sct * 2
            )
        )
        print(
            "Total {:,} hits, out of which {:,} are true hits".format(
                hits.shape[0], num_true_hits
            )
        )
        
    os.makedirs(outputdir, exist_ok=True)
    if (
        not os.path.exists(os.path.join(outputdir, "event{:09}-truth.csv").format(evtid))
        or overwrite
    ):
        opt = {"index": False}
        # Save truth data
        truth_spacepoints.to_csv(
            os.path.join(outputdir, "event{:09}-truth.csv".format(evtid)), **opt
        )

        # Save particle data
        reco_particles.to_csv(
            os.path.join(outputdir, "event{:09}-particles.csv".format(evtid)), **opt
        )

    return (
        tot_hits,
        tot_particles,
        num_hits_pixel,
        num_hits_sct,
        num_true_hits,
        num_reco_particles,
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="read atlas dataset")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="process_config.yaml")
    add_arg("--debug", action="store_true", help="print out debug info")
    args = parser.parse_args()

    with open(args.config) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    inputdir = configs["inputdir"]
    outputdir = configs["outputdir"]
    particles_datatypes = configs["particles_datatypes"]
    spacepoints_datatypes = configs["spacepoints_datatypes"]

    num_workers = configs["num_workers"]
    max_evts = configs["max_evts"]
    evtId_offset = configs["evtId_offset"]
    overwrite = configs["overwrite"]

    file_pattern = os.path.join(inputdir, "**/clusters*.txt")
    files = glob.glob(file_pattern, recursive=True)
                
    if max_evts is not None:
        files = files[:max_evts]
    print("Input directory:", inputdir)
    print("Output directory:", outputdir)
    print(
        "Total {} events to be processed with {} workers.".format(
            len(files), num_workers
        )
    )
    if len(files) > 1 and num_workers > 1:
        with Pool(num_workers) as p:
            process_fnc = partial(
                process_event, inputdir=inputdir, outputdir=outputdir, 
                particles_datatypes=particles_datatypes, spacepoints_datatypes=spacepoints_datatypes, 
                debug=args.debug,evtId_offset=evtId_offset, overwrite=overwrite, num_events=len(files)
            )
            res = p.map(process_fnc, files)
    else:
        res = [
            process_event(
                file, inputdir=inputdir, outputdir=outputdir, 
                particles_datatypes=particles_datatypes, spacepoints_datatypes=spacepoints_datatypes, 
                debug=args.debug,evtId_offset=evtId_offset, overwrite=overwrite, num_events=len(files)
            )
            for file in files
        ]
        
    res = np.array(res).T
    (
        tot_hits,
        tot_particles,
        num_hits_pixel,
        num_hits_sct,
        num_true_hits,
        num_reco_particles,
    ) = [np.mean(res[idx]) for idx in range(6)]

    print("---{:,} events----".format(len(files)))
    print("Follow numbers are averaged.")
    print(
        "Total {:,.0f} hits, out of which {:,} are true hits: {:.4f}%".format(
            tot_hits, num_true_hits, 100 * num_true_hits / tot_hits
        )
    )
    print(
        "Total {:,.0f} particles, out of which {:,.0f} are of interest: {:.4f}%.".format(
            tot_particles, num_reco_particles, num_reco_particles * 100 / tot_particles
        )
    )
    print(
        "{:,.0f} hits in PIXEL: {:.1f}% ".format(
            num_hits_pixel, 100 * num_hits_pixel / tot_hits
        )
    )
    print(
        "{:,.0f} hits in SCT:   {:.1f}% ".format(
            num_hits_sct, 100 * num_hits_sct / tot_hits
        )
    )
    print(
        "Expecting total {:,.0f} cluster info".format(num_hits_pixel + num_hits_sct * 2)
    )
