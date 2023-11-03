from atlasify import atlasify
import os
from matplotlib import pyplot as plt
import numpy as np
from pytorch_lightning import LightningModule
from sklearn.metrics import auc, roc_curve
import torch
from tqdm import tqdm

from gnn4itk_cf.stages.track_building.utils import rearrange_by_distance
from gnn4itk_cf.utils.plotting_utils import (
    get_ratio,
    plot_1d_histogram,
    plot_eff_pur_region,
    plot_efficiency_rz,
)


def graph_construction_efficiency(lightning_module, plot_config, config):
    """
    Plot the graph construction efficiency vs. pT of the edge.
    """
    all_y_truth, all_pt = [], []
    all_eta = []
    graph_size = []

    for event in tqdm(lightning_module.testset):
        if isinstance(lightning_module, LightningModule):
            event = event.to(lightning_module.device)
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.target_mask = torch.ones(event.truth_map.shape[0], dtype=torch.bool)

        event = event.cpu()
        all_y_truth.append((event.truth_map[event.target_mask] >= 0).cpu())
        all_pt.append((event.pt[event.target_mask].cpu()))

        all_eta.append((event.eta[event.track_edges[:, event.target_mask][0]].cpu()))
        graph_size.append(event.edge_index.size(1))

    #  TODO: Handle different pT units!
    all_pt = torch.cat(all_pt).cpu().numpy()
    all_eta = torch.cat(all_eta).cpu().numpy()
    all_y_truth = torch.cat(all_y_truth).cpu().numpy()

    # Get the edgewise efficiency
    # Build a histogram of true pTs, and a histogram of true-positive pTs
    pt_min, pt_max = 1, 50
    if "pt_units" in plot_config and plot_config["pt_units"] == "MeV":
        pt_min, pt_max = pt_min * 1000, pt_max * 1000
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)
    eta_bins = np.linspace(-4, 4)

    true_pt_hist, _ = np.histogram(all_pt, bins=pt_bins)
    true_pos_pt_hist, _ = np.histogram(all_pt[all_y_truth], bins=pt_bins)

    true_eta_hist, true_eta_bins = np.histogram(all_eta, bins=eta_bins)
    true_pos_eta_hist, _ = np.histogram(all_eta[all_y_truth], bins=eta_bins)

    pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]

    for true_pos_hist, true_hist, bins, xlabel, logx, filename in zip(
        [true_pos_pt_hist, true_pos_eta_hist],
        [true_pt_hist, true_eta_hist],
        [pt_bins, eta_bins],
        [f"$p_T [{pt_units}]$", r"$\eta$"],
        [True, False],
        ["edgewise_efficiency_pt.png", "edgewise_efficiency_eta.png"],
    ):
        hist, err = get_ratio(true_pos_hist, true_hist)
        if plot_config.get("filename_template") is not None:
            filename = config["filename_template"] + "_" + filename

        fig, ax = plot_1d_histogram(
            hist,
            bins,
            err,
            xlabel,
            plot_config["title"],
            plot_config.get("ylim", [0.9, 1.04]),
            "Efficiency",
            logx=logx,
        )

        # Save the plot
        atlasify(
            atlas="Internal",
            subtext=(
                r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
                r" $t \bar{t}$ and soft interactions) "
            )
            + "\n"
            r"$p_T > 1$GeV, $|\eta| < 4$" + "\n"
            r"Mean graph size: "
            + f"{np.mean(graph_size):.2e}"
            + r"$\pm$"
            + f"{np.std(graph_size):.2e}"
            + "\n"
            + f"Global efficiency: {all_y_truth.sum() / all_pt.shape[0] :.4f}",
        )
        fig.savefig(os.path.join(config["stage_dir"], filename))

        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(config["stage_dir"], filename)}'
        )


def graph_scoring_efficiency(lightning_module, plot_config, config):
    """
    Plot the graph construction efficiency vs. pT of the edge.
    """
    print("Plotting efficiency against pT and eta")
    true_positive, target_pt, target_eta = [], [], []
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
            event.target_mask = torch.ones(event.truth_map.shape[0], dtype=torch.bool)

        # get all target true positives
        true_positive.append((event.truth_map[event.target_mask] > -1).cpu())
        # get all target pt. Length = number of target true
        target_pt.append(event.pt[event.target_mask].cpu())
        # target_eta.append(event.eta[event.target_mask])
        target_eta.append(event.eta[event.track_edges[:, event.target_mask][0]])
        # get all edges passing edge cut
        if "scores" in event.keys:
            pred.append((event.scores >= config["score_cut"]).cpu())
        else:
            pred.append(event.y.cpu())
        # get all target edges in input graphs
        graph_truth.append((event.graph_truth_map[event.target_mask] > -1))

    # concat all target pt and eta
    target_pt = torch.cat(target_pt).cpu().numpy()
    target_eta = torch.cat(target_eta).cpu().numpy()

    # get all true positive
    true_positive = torch.cat(true_positive).cpu().numpy()

    # get all positive
    graph_truth = torch.cat(graph_truth).cpu().numpy()

    # count number of graphs to calculate mean efficiency
    n_graphs = len(pred)

    # get all predictions
    pred = torch.cat(pred).cpu().numpy()

    # get mean graph size
    mean_graph_size = pred.sum() / n_graphs

    # get mean target efficiency
    target_efficiency = true_positive.sum() / len(target_pt)
    target_purity = true_positive.sum() / pred.sum()

    # get graph construction efficiency
    graph_construction_efficiency = graph_truth.mean()

    # Get the edgewise efficiency
    # Build a histogram of true pTs, and a histogram of true-positive pTs
    pt_min, pt_max = 1, 50
    if "pt_units" in plot_config and plot_config["pt_units"] == "MeV":
        pt_min, pt_max = pt_min * 1000, pt_max * 1000
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)

    eta_bins = np.linspace(-4, 4)

    true_pt_hist, true_pt_bins = np.histogram(target_pt, bins=pt_bins)
    true_pos_pt_hist, _ = np.histogram(target_pt[true_positive], bins=pt_bins)

    true_eta_hist, true_eta_bins = np.histogram(target_eta, bins=eta_bins)
    true_pos_eta_hist, _ = np.histogram(target_eta[true_positive], bins=eta_bins)

    pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]

    filename = plot_config.get("filename", "edgewise_efficiency")

    for true_pos_hist, true_hist, bins, xlabel, logx, filename in zip(
        [true_pos_pt_hist, true_pos_eta_hist],
        [true_pt_hist, true_eta_hist],
        [true_pt_bins, true_eta_bins],
        [f"$p_T [{pt_units}]$", r"$\eta$"],
        [True, False],
        [f"{filename}_pt.png", f"{filename}_eta.png"],
    ):
        # Divide the two histograms to get the edgewise efficiency
        hist, err = get_ratio(true_pos_hist, true_hist)

        fig, ax = plot_1d_histogram(
            hist,
            bins,
            err,
            xlabel,
            plot_config["title"],
            plot_config.get("ylim", [0.9, 1.04]),
            "Efficiency",
            logx=logx,
        )

        # Save the plot
        atlasify(
            "Internal",
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
            r" \bar{t}$ and soft interactions) " + "\n"
            r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
            r"Edge score cut: " + str(config["score_cut"]) + "\n"
            f"Input graph size: {pred.shape[0]/n_graphs:.2e}, Graph Construction"
            f" Efficiency: {graph_construction_efficiency:.3f}" + "\n"
            f"Mean graph size: {mean_graph_size:.2e}, Signal Efficiency:"
            f" {target_efficiency:.3f}",
        )

        fig.savefig(os.path.join(config["stage_dir"], filename))
        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(config["stage_dir"], filename)}'
        )


def multi_edgecut_graph_scoring_efficiency(lightning_module, plot_config, config):
    """Plot graph scoring efficiency across multiple score cuts

    Args:
        lightning_module (_type_): lightning module from which to draw evaluation
        plot_config (_type_): Plot config, must contain
            'score_cuts: LIST OF CUTS
            'filename_template': A TEMPLATE FOR FILENAME
        config (_type_): Usual config from lightning module and evaluation config
    """

    filenames = [
        f"{plot_config['template_filename']}_{cut*100:.0f}"
        for cut in plot_config["score_cuts"]
    ]
    for score_cut, filename in zip(plot_config["score_cuts"], filenames):
        config["score_cut"] = score_cut
        plot_config["filename"] = filename
        graph_scoring_efficiency(lightning_module, plot_config, config)


def graph_roc_curve(lightning_module, plot_config, config):
    """
    Plot the ROC curve for the graph construction efficiency.
    """
    print("Plotting the ROC curve")
    all_y_truth, all_scores = [], []

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
    ax.plot(fpr, tpr, color="black", label="ROC curve")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Random classifier")
    ax.set_xlabel("False Positive Rate", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("True Positive Rate", ha="right", y=0.95, fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="lower right", fontsize=14)
    ax.text(
        0.95,
        0.20,
        f"AUC: {auc_score:.3f}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=14,
    )

    # Save the plot
    atlasify(
        "Internal",
        f"{plot_config['title']} \n"
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
        r" \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$GeV, $|\eta| < 4$",
    )
    fig.savefig(os.path.join(config["stage_dir"], "roc_curve.png"))
    print(
        "Finish plotting. Find the ROC curve at"
        f' {os.path.join(config["stage_dir"], "roc_curve.png")}'
    )


def graph_region_efficiency_purity(lightning_module, plot_config, config):
    print("Plotting efficiency and purity by region")
    edge_truth, edge_regions, edge_positive = [], [], []
    node_r, node_z, node_regions = [], [], []

    for event in tqdm(lightning_module.testset):
        with torch.no_grad():
            eval_dict = lightning_module.shared_evaluation(
                event.to(lightning_module.device), 0
            )
        event = eval_dict["batch"]
        event.scores = torch.sigmoid(eval_dict["output"])

        edge_truth.append(event.y)
        edge_regions.append(
            event.x_region[event.edge_index[0]]
        )  # Assign region depending on first node in edge
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

    fig, ax = plot_eff_pur_region(
        edge_truth,
        edge_positive,
        edge_regions,
        node_r,
        node_z,
        node_regions,
        plot_config,
    )
    fig.savefig(os.path.join(config["stage_dir"], "region_eff_pur.png"))
    print(
        "Finish plotting. Find the plot at"
        f' {os.path.join(config["stage_dir"], "region_eff_pur.png")}'
    )


def gnn_efficiency_rz(lightning_module, plot_config: dict, config: dict):
    """_summary_

    Args:
        plot_config (dict): any plotting config
        config (dict): config

    Plot GNN edgewise efficiency against rz
    """

    print("Plotting GNN edgewise efficiency as a function of rz")
    target = {"z": torch.empty(0), "r": torch.empty(0)}
    all_target = target.copy()
    true_positive = target.copy()
    for key in ["z_tight", "r_tight", "z_loose", "r_loose"]:
        true_positive[key] = torch.empty(0)

    for event in tqdm(lightning_module.testset):
        event = event.to(lightning_module.device)

        # Need to apply score cut and remap the truth_map
        if "score_cut" in config:
            lightning_module.apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.target_mask = torch.ones(event.truth_map.shape[0], dtype=torch.bool)

        # scale r and z
        event.r /= 1000
        event.z /= 1000

        # flip edges that point inward if not undirected, since if undirected is True, lightning_module.apply_score_cut takes care of this
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)
        event = event.cpu()

        # indices of all target edges present in the input graph
        target_edges = event.track_edges[
            :, event.target_mask & (event.graph_truth_map > -1)
        ]

        # indices of all target edges (may or may not be present in the input graph)
        all_target_edges = event.track_edges[:, event.target_mask]

        # get target z r
        for key, item in target.items():
            target[key] = torch.cat([item, event[key][target_edges[0]]], dim=0)
        for key, item in all_target.items():
            all_target[key] = torch.cat([item, event[key][all_target_edges[0]]], dim=0)

        # indices of all true positive target edges
        target_true_positive_edges = event.track_edges[
            :, event.target_mask & (event.truth_map > -1)
        ]
        for key in ["r", "z"]:
            true_positive[key] = torch.cat(
                [true_positive[key], event[key][target_true_positive_edges[0]]], dim=0
            )

    fig, ax = plot_efficiency_rz(
        target["z"], target["r"], true_positive["z"], true_positive["r"], plot_config
    )
    # Save the plot
    atlasify(
        "Internal",
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
        r" \bar{t}$ and soft interactions) \n"
        r"$p_T > 1$ GeV, $ | \eta | < 4$ \n"
        r"Edge score cut: " + str(config["score_cut"]) + "\n"
        "Graph Construction Efficiency:"
        f" {(target['z'].shape[0] / all_target['z'].shape[0]):.3f} \n"
        "Signal Efficiency:"
        f" {true_positive['z'].shape[0] / target['z'].shape[0] :.3f} \n",
    )
    plt.tight_layout()
    save_dir = os.path.join(config["stage_dir"], "edgewise_efficiency_rz.png")
    fig.savefig(save_dir)
    print(f"Finish plotting. Find the plot at {save_dir}")
    plt.close()

    fig, ax = plot_efficiency_rz(
        all_target["z"],
        all_target["r"],
        true_positive["z"],
        true_positive["r"],
        plot_config,
    )
    # Save the plot
    atlasify(
        "Internal",
        r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
        r" \bar{t}$ and soft interactions) " + "\n"
        r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        r"Edge score cut: " + str(config["score_cut"]) + "\n",
    )
    plt.tight_layout()
    save_dir = os.path.join(
        config["stage_dir"], "cumulative_edgewise_efficiency_rz.png"
    )
    fig.savefig(save_dir)
    print(f"Finish plotting. Find the plot at {save_dir}")
    plt.close()


def gnn_purity_rz(lightning_module, plot_config: dict, config: dict):
    """_summary_

    Args:
        plot_config (dict): any plotting config
        config (dict): config

    Plot GNN edgewise efficiency against rz
    """

    print("Plotting GNN edgewise efficiency as a function of rz")

    true_positive = {
        key: torch.empty(0).to(lightning_module.device) for key in ["z", "r"]
    }
    target_true_positive = true_positive.copy()

    pred = true_positive.copy()
    masked_pred = true_positive.copy()

    for event in tqdm(lightning_module.testset):
        event = event.to(lightning_module.device)
        # Need to apply score cut and remap the truth_map
        if "score_cut" in config:
            lightning_module.apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.target_mask = torch.ones(event.truth_map.shape[0], dtype=torch.bool)

        # scale r and z
        event.r /= 1000
        event.z /= 1000

        # flip edges that point inward if not undirected, since if undirected is True, lightning_module.apply_score_cut takes care of this
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)
        # event = event.cpu()

        # target true positive edge indices, used as numerator of target purity and purity
        target_true_positive_edges = event.track_edges[
            :, event.target_mask & (event.truth_map > -1)
        ]

        # true positive edge indices, used as numerator of total purity
        true_positive_edges = event.track_edges[:, (event.truth_map > -1)]

        # all positive edges, used as denominator of total and target purity
        positive_edges = event.edge_index[:, event.pred]

        # masked positive edge indices, including true positive target edges and all false positive edges
        fake_positive_edges = event.edge_index[:, event.pred & (event.y == 0)]
        masked_positive_edges = torch.cat(
            [target_true_positive_edges, fake_positive_edges], dim=1
        )

        for key in ["r", "z"]:
            target_true_positive[key] = torch.cat(
                [
                    target_true_positive[key].float(),
                    event[key][target_true_positive_edges[0]].float(),
                ],
                dim=0,
            )
            true_positive[key] = torch.cat(
                [
                    true_positive[key].float(),
                    event[key][true_positive_edges[0]].float(),
                ],
                dim=0,
            )
            pred[key] = torch.cat(
                [pred[key].float(), event[key][positive_edges[0]].float()], dim=0
            )
            masked_pred[key] = torch.cat(
                [
                    masked_pred[key].float(),
                    event[key][masked_positive_edges[0]].float(),
                ],
                dim=0,
            )

    for numerator, denominator, suffix in zip(
        [true_positive, target_true_positive, target_true_positive],
        [pred, pred, masked_pred],
        ["total_purity", "target_purity", "masked_purity"],
    ):
        fig, ax = plot_efficiency_rz(
            denominator["z"].cpu(),
            denominator["r"].cpu(),
            numerator["z"].cpu(),
            numerator["r"].cpu(),
            plot_config,
        )
        # Save the plot
        atlasify(
            "Internal",
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries $t"
            r" \bar{t}$ and soft interactions) " + "\n"
            r"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
            r"Edge score cut: " + str(config["score_cut"]) + "\n"
            r"Global purity: "
            + f"{numerator['z'].size(0) / denominator['z'].size(0) : .5f}",
        )
        plt.tight_layout()
        save_dir = os.path.join(
            config["stage_dir"],
            f"{plot_config.get('filename', 'edgewise')}_{suffix}_rz.png",
        )
        fig.savefig(save_dir)
        print(f"Finish plotting. Find the plot at {save_dir}")
        plt.close()
