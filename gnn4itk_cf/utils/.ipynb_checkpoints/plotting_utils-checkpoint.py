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


import math
import numpy as np
import matplotlib.pyplot as plt


# fontsize=16
# minor_size=14
# pt_min, pt_max = 1, 20
# default_pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)
# default_pt_configs = {
#     'bins': default_pt_bins,
#     'histtype': 'step',
#     'lw': 2,
#     'log': False
# }

# default_eta_bins = np.arange(-4., 4.4, step=0.4)
# default_eta_configs = {
#     'bins': default_eta_bins,
#     'histtype': 'step',
#     'lw': 2,
#     'log': False
# }


# def get_plot(nrows=1, ncols=1, figsize=6, nominor=False):

#     fig, axs = plt.subplots(nrows, ncols,
#         figsize=(figsize*ncols, figsize*nrows),
#         constrained_layout=True)

#     def format(ax):
#         ax.xaxis.set_minor_locator(AutoMinorLocator())
#         ax.yaxis.set_minor_locator(AutoMinorLocator())
#         return ax

#     if nrows * ncols == 1:
#         ax = axs
#         if not nominor: format(ax)
#     else:
#         ax = [x if nominor else format(x) for x in axs.flatten()]

#     return fig, ax

# def add_up_xaxis(ax):
#     ax2 = ax.twiny()
#     ax2.set_xticks(ax.get_xticks())
#     ax2.set_xbound(ax.get_xbound())
#     ax2.set_xticklabels(["" for _ in ax.get_xticks()])
#     ax2.xaxis.set_minor_locator(AutoMinorLocator())

def get_ratio(x_vals, y_vals):
    res = [x / y if y != 0 else 0.0 for x, y in zip(x_vals, y_vals)]
    err = [x / y * math.sqrt((x + y) / (x * y)) if y != 0 and x != 0 else 0.0 for x, y in zip(x_vals, y_vals)]
    return res, err


# def pairwise(iterable):
#   """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
#   a, b = itertools.tee(iterable)
#   next(b, None)
#   return zip(a, b)

# def add_mean_std(array, x, y, ax, color='k', dy=0.3, digits=2, fontsize=12, with_std=True):
#     this_mean, this_std =import itertools np.mean(array), np.std(array)
#     ax.text(x, y, "Mean: {0:.{1}f}".format(this_mean, digits), color=color, fontsize=12)
#     if with_std:
#         ax.text(x, y-dy, "Standard Deviation: {0:.{1}f}".format(this_std, digits), color=color, fontsize=12)

# def make_cmp_plot(
#     arrays, legends, configs,
#     xlabel, ylabel, ratio_label,
#     ratio_legends, ymin=0):

#     _, ax = get_plot()
#     vals_list = []
#     for array,legend in zip(arrays, legends):
#         vals, bins, _ = ax.hist(array, **configs, label=legend)
#         vals_list.append(vals)

#     ax.set_xlabel(xlabel, fontsize=fontsize)
#     ax.set_ylabel(ylabel, fontsize=fontsize)
#     add_up_xaxis(ax)
#     ax.legend()
#     ax.grid(True)
#     plt.show()

#     # make a ratio plot
#     _, ax = get_plot()
#     xvals = [0.5*(x[1]+x[0]) for x in pairwise(bins)]
#     xerrs = [0.5*(x[1]-x[0]) for x in pairwise(bins)]

#     for idx in range(1, len(arrays)):
#         ratio, ratio_err = get_ratio(vals_list[-1], vals_list[idx-1])
#         label = None if ratio_legends is None else ratio_legends[idx-1]
#         ax.errorbar(
#             xvals, ratio, yerr=ratio_err, fmt='o',
#             xerr=xerrs, lw=2, label=label)


#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ratio_label)
#     add_up_xaxis(ax)

#     if ratio_legends is not None:
#         ax.legend()
#     ax.grid(True)
#     plt.show()


# def plot_observable_performance(particles: pd.DataFrame):

#     pt = particles.pt.values
#     eta = particles.eta.values

#     fiducial = (particles.status == 1) & (particles.barcode < 200000) & (particles.eta.abs() < 4) & (particles.radius < 260) & (particles.charge.abs() > 0)
#     trackable = particles.is_trackable
#     matched = particles.is_double_matched


#     # plot the performance `metric` as a function of `observable`
#     make_cmp_plot_fn = partial(make_cmp_plot,
#         legends=["Generated", "Reconstructable", "Matched"],
#         ylabel="Num. particles", ratio_label='Track efficiency',
#         ratio_legends=["Physics Eff", "Technical Eff"])

#     all_cuts = [(1000, 4)]
#     for (cut_pt, cut_eta) in all_cuts:
#         cuts = (pt > cut_pt) & (np.abs(eta) < cut_eta)
#         gen_pt = pt[cuts & fiducial]
#         true_pt = pt[cuts & fiducial & trackable]
#         reco_pt = pt[cuts & fiducial & trackable & matched]
#         make_cmp_plot_fn([gen_pt, true_pt, reco_pt],
#             configs=default_pt_configs, xlabel="pT [MeV]", ymin=0.6)

#         gen_eta = eta[cuts & fiducial]
#         true_eta = eta[cuts & fiducial & trackable]
#         reco_eta = eta[cuts & fiducial & trackable & matched]
#         make_cmp_plot_fn([gen_eta, true_eta, reco_eta], configs=default_eta_configs, xlabel=r"$\eta$", ymin=0.6)

# def plot_pt_eff(particles):

#     pt = particles.pt.values

#     true_pt = pt[particles["is_reconstructable"]]
#     reco_pt = pt[particles["is_reconstructable"] & particles["is_reconstructed"]]

#     # Get histogram values of true_pt and reco_pt
#     true_vals, true_bins = np.histogram(true_pt, bins=default_pt_bins)
#     reco_vals, _ = np.histogram(reco_pt, bins=default_pt_bins)

#     # Plot the ratio of the histograms as an efficiency
#     eff, err = get_ratio(reco_vals, true_vals)

#     xvals = (true_bins[1:] + true_bins[:-1]) / 2
#     xerrs = (true_bins[1:] - true_bins[:-1]) / 2

#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.errorbar(xvals, eff, xerr=xerrs, yerr=err, fmt='o', color='black', label='Efficiency')
#     # Add x and y labels
#     ax.set_xlabel('$p_T [GeV]$', fontsize=16)
#     ax.set_ylabel('Efficiency', fontsize=16)

#     # Save the plot
#     fig.savefig('pt_efficiency.png')
def plot_eff_pur_region(edge_truth, edge_positive, edge_regions, node_r, node_z, node_regions, plot_config):
    # Draw a few nodes to get a feeling for the geometry
    fig, ax = plt.subplots()

    draw_idxs = np.arange(len(node_z))
    np.random.shuffle(draw_idxs)
    draw_idxs = draw_idxs[:1000]
    ax.scatter(node_z[draw_idxs], node_r[draw_idxs], s=1, color='lightgrey')

    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for region, color in zip(range(1, 7), colors):
        edge_mask = (edge_regions == region)
        node_mask = (node_regions == region)
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
            np.mean(z_range) * 1.1 - 0.5,
            np.mean(r_range),
            fmt_str.format(eff, pur))
        rectangle_args = {
            "xy": (z_range[0], r_range[0]),
            "width": (z_range[1] - z_range[0]),
            "height": (r_range[1] - r_range[0]),
        }
        ax.add_patch(plt.Rectangle(**rectangle_args, alpha=0.1, color=color, linewidth=0))

    ax.set_xlabel("z")
    ax.set_ylabel("r")

    return fig, ax
