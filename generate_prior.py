import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import torch
import os

from bnn.bnn_utils import DistributionType

from bnn.bnn_pytorch import ActivationFunctions
from utils.prior_plots import (
    plot_different_priors,
    make_bnn_convergence_plot,
)

import matplotlib

matplotlib.rcParams.update({"font.size": 26})
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
matplotlib.rcParams["text.usetex"] = True

# define constants
N_DISCR = 130
LOWER_BOUND = 0
UPPER_BOUND = 1

N_REALISATIONS = 5

# resolution
DPI = 300

PLOT_PATH = "figures/priors/"

layer_vector = [2, 40, 80, 1]


def main():

    # np.random.seed(7)

    parser = argparse.ArgumentParser(description="Process command line " "arguments")

    parser.add_argument(
        "--distr",
        "-d",
        dest="distr_type",
        choices=[
            DistributionType.GAUSSIAN.value,
            DistributionType.CAUCHY.value,
        ],
        default=DistributionType.GAUSSIAN,
        type=DistributionType,
        help="Distribution for the network weights and biases",
    )

    parser.add_argument(
        "--dim",
        "-di",
        dest="input_dim",
        choices=[1, 2],
        default=1,
        type=int,
        help="Input dimension: 1 or 2",
    )

    args = parser.parse_args()

    # create directory to save figures (if it does not exist)
    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH)

    # The convergence test for the shallow Gaussian neural network
    # -> with large number of hidden units output should converge to Gaussian process
    # make_bnn_convergence_plot(PLOT_PATH)

    # plot different priors
    x_input = torch.tensor(np.linspace(LOWER_BOUND, UPPER_BOUND, N_DISCR), dtype=torch.double).view(1, -1)

    # check if we plot one-dimensional or two-dimensional prior realisations
    if args.input_dim == 1:
        bnn_input = x_input
    elif args.input_dim == 2:
        # generate a grid of points in 2D
        y_input = torch.tensor(np.linspace(LOWER_BOUND, UPPER_BOUND, N_DISCR), dtype=torch.double).view(1, -1)
        x_coord, y_coord = np.meshgrid(x_input, y_input)
        # create input for the BNN
        bnn_input = torch.tensor(
            np.concatenate((x_coord.reshape(1, -1), y_coord.reshape(1, -1)), axis=0), dtype=torch.double
        )
    else:
        logging.info("Please use either 1 or 2 as a dimension of a problem, higher dimensions are not supported")
        return

    torch.manual_seed(1996)

    # study different network architectures (how they affect the prior realizations)
    plot_different_priors(
        input_dim=args.input_dim,
        x_input=bnn_input,
        distr_type=args.distr_type,
        act_func=ActivationFunctions.TANH,
        n_units=[50, 100, 500],
        n_layers=[1, 2, 3],
        data_path=PLOT_PATH,
        n_realisations=9,
        x_lims=[LOWER_BOUND, UPPER_BOUND],
        mu_weights=0,
        mu_biases=0,
        std_weights=10.0,
        std_biases=5.0,
    )

    plt.show()
    return


if __name__ == "__main__":
    main()
