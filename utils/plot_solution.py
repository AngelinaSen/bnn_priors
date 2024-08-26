import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

import scipy.stats as sps

from deconvolution_1d.deconv_bnn import ObservationData, SignalType

import matplotlib
import arviz as az

from utils.quality_metrics import compute_iact, compute_rel_error

# for LaTeX style plots
matplotlib.rcParams.update({"font.size": 22})
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
matplotlib.rcParams["text.usetex"] = True

# plot constants
X_MIN = -1
X_MAX = 1

Y_MIN = -2
Y_MAX = 2

DPI = 300  # resolution
LINE_WIDTH_SOL = 2
LINE_WIDTH = 3


def plot_tracking_points(
    obs_data: ObservationData,
    points: List,
    x_lims: List[float],
    res_fig_path: str,
) -> None:
    """
    Function to make a plot showing the domain points at which we track the signal values
    :param obs_data: class containing data
    :param points: list of tracking points
    :param x_lims: plot limits
    :param res_fig_path: path to save the resulting figure
    :return:
    """
    n_discr_colormap = 20
    custom_cmap = plt.cm.get_cmap("coolwarm", n_discr_colormap)
    colors = custom_cmap(range(n_discr_colormap))
    ind = [1, 3, 15, 17, 19]
    colors = colors[ind]

    font_size_ticks = 32
    font_size = 36

    if obs_data.signal_type == SignalType.smooth:
        y_lims = [-1.2, 8]
    elif obs_data.signal_type == SignalType.datasky:
        # y_lims = [-0.2, 1.7]
        y_lims = [-0.2, 1.7]
    else:
        raise ValueError("Wrong signal type")

    plt.figure(figsize=(8, 5))
    plt.plot(obs_data.domain_discr, obs_data.signal_true, "k-", linewidth=2.5, label="True")

    j = 0
    for i in points:

        plt.vlines(obs_data.domain_discr[i], y_lims[0], y_lims[1], colors=colors[j], linestyles="--", linewidth=2.5)
        print(f"Ind = {i}, x_i = {obs_data.domain_discr[i]}")
        j += 1

    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
    plt.yticks(fontsize=font_size_ticks)
    plt.xticks(fontsize=font_size_ticks)
    plt.xlabel(r"$t$", fontsize=font_size)
    plt.ylabel(r"$x$", fontsize=font_size)
    # plt.legend(ncol=2, loc="upper center", fontsize=25)
    plt.tight_layout(pad=0.3)
    plt.savefig(res_fig_path + "tracked_points_u.pdf", dpi=DPI)
    # plt.show()

    return


def plot_solution(
    u_chain: np.ndarray,
    out_path: str,
    observations: ObservationData,
    x_lims: List[float],
    u_map: Optional[np.ndarray] = None,
) -> None:
    """
    Plot solution as mean of MCMC chain together with true solution of the problem
    :param u_chain: chain of vectors u values (corresponds to solution - signal approximation)
    :param out_path: path to the directory containing the MCMC results
    :param observations: observation data containing true solution
    :param x_lims: limits for x-axis
    :param y_lims: limits for y-axis
    :param u_map: MAP estimate (optional)
    :return:

    """

    if observations.signal_type == SignalType.smooth:
        y_lims = [-1.2, 8]

    elif observations.signal_type == SignalType.datasky:
        y_lims = [-0.2, 1.7]
        # y_lims = [-1.5, 3.0]

    else:
        raise ValueError("Wrong signal type")

    font_size_ticks = 32
    font_size = 36

    # find mean and 98% HDI (the highest density interval)
    u_mean = np.mean(u_chain, axis=0)  # for pCN results axis = 0

    hdi_prob = 0.98
    chain_hdi_95 = az.hdi(u_chain, hdi_prob=hdi_prob)

    # compute the relative error for the mean solution
    rel_error = compute_rel_error(observations.signal_true, u_mean)

    # mean with 95% confidence interval
    plt.figure(figsize=(8, 5))
    # true signal
    plt.plot(observations.domain_discr, observations.signal_true, "k-", linewidth=LINE_WIDTH_SOL, label="True")
    if u_map is not None:
        # MAP estimate
        plt.plot(observations.domain_discr, u_map, color="red", linestyle="dashdot", linewidth=LINE_WIDTH, label="MAP")

    plt.plot(observations.domain_discr, u_mean, "b--", linewidth=LINE_WIDTH, label="Mean")

    plt.fill_between(
        observations.domain_discr,
        chain_hdi_95[:, 0],
        chain_hdi_95[:, 1],
        color="blue",
        alpha=0.20,
        label=rf"${int(hdi_prob*100)}\%$ HDI",
    )
    # plt.legend()
    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
    plt.yticks(fontsize=font_size_ticks)
    plt.xticks(fontsize=font_size_ticks)
    plt.xlabel(r"$t$", fontsize=font_size)
    # create a secondary x-xis to add info on th relative reconstruction error
    plt.title(r"$\varepsilon_{\mathrm{rel}} = $" + rf" ${rel_error:.3f}$", fontsize=font_size)
    plt.ylabel(r"$x$", fontsize=font_size)
    plt.tight_layout()
    plt.savefig(out_path + f"mean_solution.pdf", dpi=DPI)

    return


def plot_chain_statistics_together(
    chain_u: np.ndarray,
    chain_theta: np.ndarray,
    chain_ksi: np.ndarray,
    res_fig_path: str,
) -> None:
    """
    Plot chain statistics (trace plot, cumulative mean and autocorrelation) for parameters x, theta and ksi together
    :param chain_u: corresponds to signal
    :param chain_theta: transformed (scaled) BNN parameters
    :param chain_ksi: BNN parameters before scaling
    :param res_fig_path: path to save the resulting figure
    :return:
    """
    # define all the parameter chains to plot and their labels
    all_param_chains = [chain_u, chain_theta, chain_ksi]
    param_labels = [r"$x$", r"$\theta$", r"$\xi$"]

    # how many chains?
    n_chains = chain_u.shape[1]

    # how many samples are in each chain?
    n_samples = chain_u.shape[0]

    # plot parameters
    font_size = 38
    font_size_ticks = 35

    # make plot
    n_row = 3
    n_col = 3
    fig, axs = plt.subplots(n_row, n_col, figsize=(20, 10))
    fig.tight_layout(pad=3.8, w_pad=1.7, h_pad=2)
    fig.add_gridspec(n_row, n_col)

    # set color scheme for the plot
    n_discr_colormap = 20
    custom_cmap = plt.cm.get_cmap("coolwarm", n_discr_colormap)
    colors = custom_cmap(range(n_discr_colormap))
    ind = [1, 3, 15, 17, 19]
    colors = colors[ind]

    for j in range(n_col):

        # choose parameter and its chains:
        # j = 0: x;   j = 1: theta;   j = 2: ksi

        chain_s = all_param_chains[j]

        for i in range(n_chains):
            print("=========================================================")
            print(f"Chain {i+1}: ")

            # take the current chain
            chain = chain_s[:, i]

            # compute  its mean and standard deviation
            chain_mean = np.mean(chain)
            chain_std = np.std(chain)
            print(f"Chain mean = {chain_mean:.7f}; chain std = {chain_std:.7f}")

            # plot statistics from arviz
            data_t = az.convert_to_inference_data(chain.reshape(1, n_samples))
            stats_t = az.summary(data_t)
            # print("Statistics for parameter " + par_label.value + " :")
            print(stats_t)

            # Compute integrated autocorrelation time
            iact_value, m = compute_iact(chain)
            print(f"Integrated autocorrelation time = {iact_value}, counter = {m}")

            # compute ergodic mean and autocorrelation function
            erg_mean = np.array([np.mean(chain[: i + 1]) for i in range(n_samples)])
            autocorr_func = az.autocorr(chain)

            # histogram
            low_lim = chain.min()
            up_lim = chain.max()
            n_bins = int(np.ceil(np.sqrt(n_samples)))
            binning = np.linspace(low_lim, up_lim, num=n_bins)
            hist, bins = np.histogram(chain, bins=binning, density=True)

            # The 1st subplot: trace plot
            axs[j, 0].plot(chain, linewidth=1, color=colors[i])
            axs[j, 0].set_xlim([0, n_samples])
            axs[j, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axs[j, 0].yaxis.get_offset_text().set_fontsize(font_size_ticks)
            if j == 0:
                axs[j, 0].set_title("Chain", fontsize=font_size)
            if j == 2:
                axs[j, 0].set_xlabel("Sample index", fontsize=font_size)
            axs[j, 0].set_ylabel(param_labels[j], fontsize=font_size)
            axs[j, 0].tick_params(axis="both", which="major", labelsize=font_size_ticks)
            axs[j, 0].tick_params(axis="both", which="minor", labelsize=font_size_ticks)
            axs[j, 1].set_xticks(np.arange(0, n_samples + 0.01, n_samples / 2))

            # The 2d subplot: Ergodic mean
            axs[j, 1].plot(erg_mean, linewidth=1.5, color=colors[i])
            axs[j, 1].set_xlim([0, n_samples])
            # ax_2.set_ylim([1.0, 1.2])  # these limits are for datasky signal (both NUTS and GIBBS, parameter nu)
            # ax_2.set_ylim([0, 40])  # these limits are for smooth signal (both NUTS and GIBBS, parameter nu)
            axs[j, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axs[j, 1].set_xticks(np.arange(0, n_samples + 0.01, n_samples / 2))
            axs[j, 1].yaxis.get_offset_text().set_fontsize(font_size_ticks)
            if j == 0:
                axs[j, 1].set_title("Cumulative mean", fontsize=font_size)
            if j == 2:
                axs[j, 1].set_xlabel("Sample index", fontsize=font_size)
            axs[j, 1].tick_params(axis="both", which="major", labelsize=font_size_ticks)
            axs[j, 1].tick_params(axis="both", which="minor", labelsize=font_size_ticks)

            # The 3d subplot: Autocorrelation function
            acf_lim = 50
            axs[j, 2].plot(autocorr_func, linewidth=1.5, color=colors[i])
            axs[j, 2].set_xlim(0, acf_lim)
            axs[j, 2].set_xticks(np.arange(0, acf_lim + 0.01, acf_lim / 2))
            axs[j, 2].set_ylim(0, 1)
            if j == 0:
                axs[j, 2].set_title("Autocorrelation", fontsize=font_size)
            if j == 2:
                axs[j, 2].set_xlabel("Lag", fontsize=font_size)
            axs[j, 2].tick_params(axis="both", which="major", labelsize=font_size_ticks)
            axs[j, 2].tick_params(axis="both", which="minor", labelsize=font_size_ticks)

    fig.align_labels()

    plt.savefig(res_fig_path + "all_param_chains.pdf", dpi=DPI)

    return


def plot_map_s(obs_data: ObservationData, u_map_s: np.ndarray, x_lims: List[float], res_fig_path: str) -> None:

    """
    Plot MAP estimates together with the true signal
    :param obs_data: data
    :param u_map_s: MAP estimates for the signal
    :param x_lims: plot limits
    :param res_fig_path: path to save the resulting figure
    :return:
    """

    # set the color map
    n_discr_colormap = 20
    custom_cmap = plt.cm.get_cmap("coolwarm", n_discr_colormap)
    colors = custom_cmap(range(n_discr_colormap))
    ind = [1, 3, 15, 17, 19]
    colors = colors[ind]

    font_size_ticks = 32
    font_size = 36

    plt.figure(figsize=(8, 6))
    plt.plot(obs_data.domain_discr, obs_data.signal_true, "k-", linewidth=LINE_WIDTH_SOL, label="True")

    # how many MAPs we have?
    num_maps = u_map_s.shape[0]

    for i in range(num_maps):
        rel_error = compute_rel_error(obs_data.signal_true, u_map_s[i].T)
        plt.plot(
            obs_data.domain_discr,
            u_map_s[i].T,
            label=r"$\varepsilon_{\mathrm{rel}} = $" + rf" ${rel_error:.3f}$",
            color=colors[i],
            linewidth=1.5,
        )
    if obs_data.signal_type == SignalType.smooth:
        y_lims = [-1.2, 11.5]
    elif obs_data.signal_type == SignalType.datasky:
        y_lims = [-0.7, 3.0]
    else:
        raise ValueError("Wrong signal type")
    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
    plt.yticks(fontsize=font_size_ticks)
    plt.xticks(fontsize=font_size_ticks)
    plt.xlabel(r"$t$", fontsize=font_size)
    plt.ylabel(r"$x$", fontsize=font_size)
    plt.legend(ncol=2, loc="upper center", fontsize=25)
    plt.tight_layout(pad=0.3)
    plt.savefig(res_fig_path + "map_estimates.pdf", dpi=DPI)
    plt.show()

    return
