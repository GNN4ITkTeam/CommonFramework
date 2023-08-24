# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List
import numpy as np
import matplotlib.pyplot as plt

import torch
import scipy


def clopper_pearson(passed: float, total: float, level: float = 0.68):
    """
    Estimate the confidence interval for a sampled binomial random variable with Clopper-Pearson.
    `passed` = number of successes; `total` = number trials; `level` = the confidence level.
    The function returns a `(low, high)` pair of numbers indicating the lower and upper error bars.
    """
    alpha = (1 - level) / 2
    lo = scipy.stats.beta.ppf(alpha, passed, total - passed + 1) if passed > 0 else 0.0
    hi = (
        scipy.stats.beta.ppf(1 - alpha, passed + 1, total - passed)
        if passed < total
        else 1.0
    )
    average = passed / total
    return (average - lo, hi - average)


def get_ratio(passed: List[int], total: List[int]):
    if len(passed) != len(total):
        raise ValueError(
            "Length of passed and total must be the same"
            f"({len(passed)} != {len(total)})"
        )

    res = np.array([x / y if y != 0 else 0.0 for x, y in zip(passed, total)])
    error = np.array([clopper_pearson(x, y) for x, y in zip(passed, total)]).T
    return res, error


def plot_eff_pur_region(
    edge_truth, edge_positive, edge_regions, node_r, node_z, node_regions, plot_config
):
    # Draw a few nodes to get a feeling for the geometry
    fig, ax = plt.subplots()

    draw_idxs = np.arange(len(node_z))
    np.random.shuffle(draw_idxs)
    draw_idxs = draw_idxs[:1000]
    ax.scatter(node_z[draw_idxs], node_r[draw_idxs], s=1, color="lightgrey")

    colors = ["r", "g", "b", "y", "c", "m"]
    for region, color in zip(range(1, 7), colors):
        edge_mask = edge_regions == region
        node_mask = node_regions == region
        true = edge_truth[edge_mask]
        positive = edge_positive[edge_mask]
        true_positive = np.logical_and(true, positive)

        if sum(true) == 0 or sum(positive) == 0:
            continue

        eff = sum(true_positive) / sum(true)
        pur = sum(true_positive) / sum(positive)

        r_range = node_r[node_mask].min(), node_r[node_mask].max()
        z_range = node_z[node_mask].min(), node_z[node_mask].max()

        fmt_str = "eff: {:.3f} \npur: {:.3f}"
        ax.text(
            np.mean(z_range) * 1.1 - 0.5, np.mean(r_range), fmt_str.format(eff, pur)
        )
        rectangle_args = {
            "xy": (z_range[0], r_range[0]),
            "width": (z_range[1] - z_range[0]),
            "height": (r_range[1] - r_range[0]),
        }
        ax.add_patch(
            plt.Rectangle(**rectangle_args, alpha=0.1, color=color, linewidth=0)
        )

    ax.set_xlabel("z")
    ax.set_ylabel("r")

    return fig, ax


def plot_efficiency_rz(
    target_z: torch.Tensor,
    target_r: torch.Tensor,
    true_positive_z: torch.Tensor,
    true_positive_r: torch.Tensor,
    plot_config: dict,
):
    z_range, r_range = plot_config.get("z_range", [-3, 3]), plot_config.get(
        "r_range", [0, 1.0]
    )
    z_bins, r_bins = plot_config.get("z_bins", 6 * 64), plot_config.get("r_bins", 64)
    z_bins = np.linspace(z_range[0], z_range[1], z_bins, endpoint=True)
    r_bins = np.linspace(r_range[0], r_range[1], r_bins, endpoint=True)

    fig, ax = plt.subplots(1, 1, figsize=plot_config.get("fig_size", (12, 6)))
    true_hist, _, _ = np.histogram2d(
        target_z.numpy(),
        target_r.numpy(),
        bins=[z_bins, r_bins],
    )
    true_positive_hist, z_edges, r_edges = np.histogram2d(
        true_positive_z.numpy(), true_positive_r.numpy(), bins=[z_bins, r_bins]
    )

    eff = true_positive_hist / true_hist

    c = ax.pcolormesh(
        z_bins,
        r_bins,
        eff.T,
        cmap="jet_r",
        vmin=plot_config.get("vmin", 0.9),
        vmax=plot_config.get("vmax", 1),
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("r [m]")

    return fig, ax


def plot_1d_histogram(
    hist, bins, err, xlabel, ylabel, ylim, label, canvas=None, logx=False
):
    """Plot 1D histogram from direct output of np.histogram

    Args:
        hist (_type_): _description_
        bins (_type_): _description_
        err (_type_): _description_
        xlabel (_type_): _description_
        ylabel (_type_): _description_
        ylim (_type_): _description_
        canvas (_type_, optional): tuple of (fig, ax). Defaults to None. If not provided, create fig, ax
        logx (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    xvals = (bins[1:] + bins[:-1]) / 2
    xerrs = (bins[1:] - bins[:-1]) / 2

    fig, ax = plt.subplots(figsize=(8, 6)) if canvas is None else canvas
    ax.errorbar(xvals, hist, xerr=xerrs, yerr=err, fmt="o", color="black", label=label)
    ax.set_xlabel(xlabel, ha="right", x=0.95, fontsize=14)
    ax.set_ylabel(ylabel, ha="right", y=0.95, fontsize=14)
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(ylim)
    plt.tight_layout()

    return fig, ax
