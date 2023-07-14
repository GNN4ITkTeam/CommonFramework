
from atlasify import atlasify
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
import torch
from tqdm import tqdm

from gnn4itk_cf.stages.track_building.utils import rearrange_by_distance
from gnn4itk_cf.utils.plotting_utils import get_ratio, plot_eff_pur_region, plot_efficiency_rz

def graph_scoring_efficiency(lightning_module, plot_config, config):
    """
    Plot the graph construction efficiency vs. pT of the edge.
    """
    print("Plotting efficiency against pT")
    true_positive, target_pt  = [], []
    pred = []
    graph_truth = []

    for event in tqdm(lightning_module.testset):
        event = event.to(lightning_module.device)
        
        # Need to apply score cut and remap the truth_map 
        if "score_cut" in config:
            lightning_module.apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.target_mask = torch.ones(event.truth_map.shape[0], dtype = torch.bool)
        
        # get all target true positives
        true_positive.append((event.truth_map[event.target_mask] > -1).cpu())
        # get all target pt. Length = number of target true
        target_pt.append(event.pt[event.target_mask].cpu())
        # get all edges passing edge cut
        pred.append((event.scores >= config["score_cut"]).cpu())
        # get all target edges in input graphs
        graph_truth.append((event.graph_truth_map[event.target_mask] > -1))
        
    target_pt = torch.cat(target_pt).cpu().numpy()
    true_positive = torch.cat(true_positive).cpu().numpy()
    graph_truth = torch.cat(graph_truth).cpu().numpy()
    n_graphs = len(pred)
    pred = torch.cat(pred).cpu().numpy()
    mean_graph_size = pred.sum() / n_graphs
    target_efficiency = true_positive.sum() / len(target_pt)
    target_purity = true_positive.sum() / pred.sum()
    graph_construction_efficiency = graph_truth.mean()

    # Get the edgewise efficiency
    # Build a histogram of true pTs, and a histogram of true-positive pTs
    pt_min, pt_max = 1, 50
    if "pt_units" in plot_config and plot_config["pt_units"] == "MeV":
        pt_min, pt_max = pt_min * 1000, pt_max * 1000
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)

    true_pt_hist, true_bins = np.histogram(target_pt, bins = pt_bins)
    true_pos_pt_hist, _ = np.histogram(target_pt[true_positive], bins = pt_bins)

    # Divide the two histograms to get the edgewise efficiency
    eff, err = get_ratio(true_pos_pt_hist,  true_pt_hist)
    xvals = (true_bins[1:] + true_bins[:-1]) / 2
    xerrs = (true_bins[1:] - true_bins[:-1]) / 2

    # Plot the edgewise efficiency
    pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(xvals, eff, xerr=xerrs, yerr=err, fmt='o', color='black', label='Efficiency')
    ax.set_xlabel(f'$p_T [{pt_units}]$', ha='right', x=0.95, fontsize=14)
    ax.set_ylabel(plot_config["title"], ha='right', y=0.95, fontsize=14)
    ax.set_xscale('log')
    ax.set_ylim(plot_config.get('ylim', [0.9, 1.04]))

    # Save the plot
    atlasify("Internal", 
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        r"Edge score cut: " + str(config["score_cut"]) + "\n"
        f"Mean graph size: {mean_graph_size:.0f}, Graph Construction Efficiency: {graph_construction_efficiency:.3f}" + "\n"
        f"Signal Efficiency: {target_efficiency:.3f}" + "\n"
    #  f"Signal Purity: {target_purity:.4f}" + "\n"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(config["stage_dir"], "edgewise_efficiency.png"))
    print(f'Finish plotting. Find the plot at {os.path.join(config["stage_dir"], "edgewise_efficiency.png")}')

def graph_roc_curve(lightning_module, plot_config, config):
    """
    Plot the ROC curve for the graph construction efficiency.
    """
    print("Plotting the ROC curve")
    all_y_truth, all_scores  = [], []

    for event in tqdm(lightning_module.testset):
        event = event.to(lightning_module.device)
            
        # Need to apply score cut and remap the truth_map 
        if "weights" in event.keys:
            target_y = event.weights.bool() & event.y.bool()
        else:
            target_y = event.y.bool()

        all_y_truth.append(target_y)
        all_scores.append(event.scores)

    all_scores = torch.cat(all_scores).cpu().numpy()
    all_y_truth = torch.cat(all_y_truth).cpu().numpy()

    # Get the ROC curve
    fpr, tpr, _ = roc_curve(all_y_truth, all_scores)
    auc_score = auc(fpr, tpr)

    # Plot the ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='black', label='ROC curve')
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random classifier')
    ax.set_xlabel('False Positive Rate', ha='right', x=0.95, fontsize=14)
    ax.set_ylabel('True Positive Rate', ha='right', y=0.95, fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', fontsize=14)
    ax.text(0.95, 0.20, f"AUC: {auc_score:.3f}", ha='right', va='bottom', transform=ax.transAxes, fontsize=14)

    # Save the plot
    atlasify("Internal", 
        f"{plot_config['title']} \n"
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$GeV, $|\eta| < 4$")
    fig.savefig(os.path.join(config["stage_dir"], "roc_curve.png"))
    print(f'Finish plotting. Find the ROC curve at {os.path.join(config["stage_dir"], "roc_curve.png")}')

def graph_region_efficiency_purity(lightning_module, plot_config, config):
    print("Plotting efficiency and purity by region")
    edge_truth, edge_regions, edge_positive  = [], [], []
    node_r, node_z, node_regions = [], [], []

    for event in tqdm(lightning_module.testset):
        with torch.no_grad():
            eval_dict = lightning_module.shared_evaluation(event.to(lightning_module.device), 0)
        event = eval_dict['batch']
        event.scores = torch.sigmoid(eval_dict['output'])

        edge_truth.append(event.y)
        edge_regions.append(event.x_region[event.edge_index[0]]) # Assign region depending on first node in edge
        edge_positive.append(event.scores > config["edge_cut"])

        node_r.append(event.x_r)
        node_z.append(event.x_z)
        node_regions.append(event.x_region)

    edge_truth = torch.cat(edge_truth).cpu().numpy()
    edge_regions = torch.cat(edge_regions).cpu().numpy()
    edge_positive = torch.cat(edge_positive).cpu().numpy()

    node_r = torch.cat(node_r).cpu().numpy()
    node_z = torch.cat(node_z).cpu().numpy()
    node_regions = torch.cat(node_regions).cpu().numpy()

    fig, ax = plot_eff_pur_region(edge_truth, edge_positive, edge_regions, node_r, node_z, node_regions, plot_config)
    fig.savefig(os.path.join(config["stage_dir"], "region_eff_pur.png"))
    print(f'Finish plotting. Find the plot at {os.path.join(config["stage_dir"], "region_eff_pur.png")}')

def gnn_efficiency_rz(lightning_module, plot_config: dict, config: dict):
    """_summary_

    Args:
        plot_config (dict): any plotting config
        config (dict): config
    
    Plot GNN edgewise efficiency against rz
    """

    print("Plotting GNN edgewise efficiency as a function of rz")
    target = {'z': torch.empty(0), 'r': torch.empty(0)}
    all_target = target.copy()
    true_positive = target.copy()
    for key in ['z_tight', 'r_tight', 'z_loose', 'r_loose']:
        true_positive[key] = torch.empty(0)

    for event in tqdm(lightning_module.testset):
        event = event.to(lightning_module.device)

        # Need to apply score cut and remap the truth_map 
        if "score_cut" in config:
            lightning_module.apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.target_mask = torch.ones(event.truth_map.shape[0], dtype = torch.bool)
        
        # scale r and z
        event.r /= 1000
        event.z /= 1000
        
        # flip edges that point inward if not undirected, since if undirected is True, lightning_module.apply_score_cut takes care of this
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)
        event=event.cpu()

        # indices of all target edges present in the input graph
        target_edges = event.track_edges[:, event.target_mask & (event.graph_truth_map > -1)]

        # indices of all target edges (may or may not be present in the input graph)
        all_target_edges = event.track_edges[:, event.target_mask]

        # get target z r
        for key, item in target.items():
            target[key] = torch.cat([item, event[key][target_edges[0]]], dim=0)
        for key, item in all_target.items():
            all_target[key] = torch.cat([item, event[key][all_target_edges[0]]], dim=0)
        if lightning_module.hparams.get('undirected'):
            for threshold in ['loose', 'tight']:
                target_true_positive_edges = event.track_edges[:, event.target_mask & (event[f'truth_map_{threshold}'] > -1)]
                for key in ['r', 'z']:
                    true_positive[f'{key}_{threshold}'] = torch.cat([true_positive[f'{key}_{threshold}'], event[key][target_true_positive_edges[0]]], dim=0)

        # indices of all true positive target edges
        target_true_positive_edges = event.track_edges[:, event.target_mask & (event.truth_map > -1)]
        for key in ['r', 'z']:
            true_positive[key] = torch.cat([true_positive[key], event[key][target_true_positive_edges[0]]], dim=0)
    
    fig, ax = plot_efficiency_rz(target['z'], target['r'], true_positive['z'], true_positive['r'], plot_config)
    # Save the plot
    atlasify("Internal", 
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        r"Edge score cut: " + str(config["score_cut"]) + "\n"
        # f"Mean graph size: {mean_graph_size:.0f}, 
        f"Graph Construction Efficiency: {(target['z'].shape[0] / all_target['z'].shape[0]):.3f}" + "\n"
        f"Signal Efficiency: {true_positive['z'].shape[0] / target['z'].shape[0] :.3f}" + "\n"
        # f"Signal Purity: {target_purity:.4f}" + "\n"
    )
    plt.tight_layout()
    save_dir = os.path.join(config["stage_dir"], "edgewise_efficiency_rz.png")
    fig.savefig(save_dir)
    print(f'Finish plotting. Find the plot at {save_dir}')
    plt.close()

    fig, ax = plot_efficiency_rz(all_target['z'], all_target['r'], true_positive['z'], true_positive['r'], plot_config)
    # Save the plot
    atlasify("Internal", 
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        r"Edge score cut: " + str(config["score_cut"]) + "\n"
    )
    plt.tight_layout()
    save_dir = os.path.join(config["stage_dir"], "cumulative_edgewise_efficiency_rz.png")
    fig.savefig(save_dir)
    print(f'Finish plotting. Find the plot at {save_dir}')
    plt.close()
    if lightning_module.hparams.get('undirected'):
        for threshold in ['loose', 'tight']:
            fig, ax = plot_efficiency_rz(target['z'], target['r'], true_positive[f'z_{threshold}'], true_positive[f'r_{threshold}'], plot_config)
            atlasify("Internal", 
                r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
                r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
                r"Edge score cut: " + str(config["score_cut"]) + f", {threshold} prediction matching" + "\n"
                f"Graph Construction Efficiency: {(target['z'].shape[0] / all_target['z'].shape[0]):.3f}" + "\n"
                f"Signal Efficiency: {true_positive[f'z_{threshold}'].shape[0] / target['z'].shape[0] :.3f}" + "\n"
            )
            plt.tight_layout()
            save_dir = os.path.join(config["stage_dir"], f"edgewise_efficiency_rz_{threshold}.png")
            fig.savefig(save_dir)
            print(f'Finish plotting. Find the plot at {save_dir}')
            plt.close()

            fig, ax = plot_efficiency_rz(all_target['z'], all_target['r'], true_positive[f'z_{threshold}'], true_positive[f'r_{threshold}'], plot_config)
            # Save the plot
            atlasify("Internal", 
                r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
                r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
                r"Edge score cut: " + str(config["score_cut"]) + f", {threshold} prediction matching" + "\n"
                f"Graph Construction Efficiency: {(target['z'].shape[0] / all_target['z'].shape[0]):.3f}" + "\n"
                f"Signal Efficiency: {true_positive[f'z_{threshold}'].shape[0] / all_target['z'].shape[0] :.3f}" + "\n"
            )
            plt.tight_layout()
            save_dir = os.path.join(config["stage_dir"], f"cumulative_edgewise_efficiency_rz_{threshold}.png")
            fig.savefig(save_dir)
            print(f'Finish plotting. Find the plot at {save_dir}')
            plt.close()

    pass

def gnn_purity_rz(lightning_module, plot_config: dict, config: dict):
    """_summary_

    Args:
        plot_config (dict): any plotting config
        config (dict): config
    
    Plot GNN edgewise efficiency against rz
    """

    print("Plotting GNN edgewise efficiency as a function of rz")
    
    true_positive = {}
    for key in ['z_tight', 'r_tight', 'z_loose', 'r_loose', 'z', 'r']:
        true_positive[key] = torch.empty(0)
    pred = true_positive.copy()

    for event in tqdm(lightning_module.testset):
        event = event.to(lightning_module.device)
        if config.get('reprocess_classifier'):
            with torch.no_grad():
                eval_dict = lightning_module.shared_evaluation(event, 0)
            event = eval_dict['batch']
            event.scores = torch.sigmoid(eval_dict['output'])
        # Need to apply score cut and remap the truth_map 
        if "score_cut" in config:
            lightning_module.apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.target_mask = torch.ones(event.truth_map.shape[0], dtype = torch.bool)
        
        # scale r and z
        event.r /= 1000
        event.z /= 1000
        
        # flip edges that point inward if not undirected, since if undirected is True, lightning_module.apply_score_cut takes care of this
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)
        event=event.cpu()

        # indices of all positive edges
        positive_edges = event.edge_index[:, event.pred]

        # indices of all target edges included in the graph
        # is_present_and_target = event.target_mask & (event.graph_truth_map > -1)
        all_included_target_edges = event.track_edges[:, event.target_mask & (event.graph_truth_map > -1)]
        all_included_target_hits = all_included_target_edges.view(-1)
        target_hit_edge_mask = torch.isin(event.edge_index, all_included_target_hits).any(dim=0)
        # print(target_hit_edge_mask)

        # get target z r
        # for key, item in pred.items():
        #     pred[key] = torch.cat([item, event[key][positive_edges[0]]], dim=0)
        if lightning_module.hparams.get('undirected'):
            for threshold in ['loose', 'tight']:
                target_true_positive_edges = event.track_edges[:, event.target_mask & (event[f'truth_map_{threshold}'] > -1)]
                positive_edges = event.edge_index[:, event[f"passing_edge_mask_{threshold}"] & target_hit_edge_mask]
                for key in ['r', 'z']:
                    true_positive[f'{key}_{threshold}'] = torch.cat([true_positive[f'{key}_{threshold}'], event[key][target_true_positive_edges[0]]], dim=0)
                    pred[f'{key}_{threshold}'] = torch.cat([pred[f'{key}_{threshold}'], event[key][positive_edges[0]]], dim=0)

        # indices of all true positive target edges
        target_true_positive_edges = event.track_edges[:, event.target_mask & (event.truth_map > -1)]
        positive_edges = event.edge_index[:, event.pred]
        for key in ['r', 'z']:
            true_positive[key] = torch.cat([true_positive[key], event[key][target_true_positive_edges[0]]], dim=0)
            pred[key] = torch.cat([pred[key], event[key][positive_edges[0]]], dim=0)
    
    fig, ax = plot_efficiency_rz(pred['z'], pred['r'], true_positive['z'], true_positive['r'], plot_config)
    # Save the plot
    atlasify("Internal", 
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        r"Edge score cut: " + str(config["score_cut"]) + "\n"
    )
    plt.tight_layout()
    save_dir = os.path.join(config["stage_dir"], "edgewise_purity_rz.png")
    fig.savefig(save_dir)
    print(f'Finish plotting. Find the plot at {save_dir}')
    plt.close()

    # fig, ax = plot_efficiency_rz(all_target['z'], all_target['r'], true_positive['z'], true_positive['r'], plot_config)
    # # Save the plot
    # atlasify("Internal", 
    #     r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
    #     r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
    #     r"Edge score cut: " + str(config["score_cut"]) + "\n"
    # )
    # plt.tight_layout()
    # save_dir = os.path.join(config["stage_dir"], "cumulative_edgewise_efficiency_rz.png")
    # fig.savefig(save_dir)
    # print(f'Finish plotting. Find the plot at {save_dir}')
    # plt.close()
    if lightning_module.hparams.get('undirected'):
        for threshold in ['loose', 'tight']:
            fig, ax = plot_efficiency_rz(pred[f'z_{threshold}'], pred[f'r_{threshold}'], true_positive[f'z_{threshold}'], true_positive[f'r_{threshold}'], plot_config)
            atlasify("Internal", 
                r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
                r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
                r"Edge score cut: " + str(config["score_cut"]) + f", {threshold} prediction matching" + "\n"
                r"Global average purity: " + f"{len(true_positive[f'z_{threshold}']) / len(pred[f'z_{threshold}']) : .3f}"
            )
            plt.tight_layout()
            save_dir = os.path.join(config["stage_dir"], f"edgewise_purity_rz_{threshold}.png")
            fig.savefig(save_dir)
            print(f'Finish plotting. Find the plot at {save_dir}')
            plt.close()

            # fig, ax = plot_efficiency_rz(all_target['z'], all_target['r'], true_positive[f'z_{threshold}'], true_positive[f'r_{threshold}'], plot_config)
            # # Save the plot
            # atlasify("Internal", 
            #     r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t \bar{t}$ and soft interactions) " + "\n"
            #     r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
            #     r"Edge score cut: " + str(config["score_cut"]) + f", {threshold} prediction matching" + "\n"
            # )
            # plt.tight_layout()
            # save_dir = os.path.join(config["stage_dir"], f"cumulative_edgewise_efficiency_rz_{threshold}.png")
            # fig.savefig(save_dir)
            # print(f'Finish plotting. Find the plot at {save_dir}')
            # plt.close()

    pass