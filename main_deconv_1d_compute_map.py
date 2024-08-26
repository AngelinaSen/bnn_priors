import numpy as np
import argparse
import torch
import sys
import enum
import time
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as sps
import pickle
import os

from bnn.bnn_pytorch import BayesianNeuralNet, ActivationFunctions
from bnn.bnn_utils import Layers, BNNParameterScale, ParamStdDistr
from deconvolution_1d.deconv_bnn import ObservationData, SignalType, NoiseLevel

from utils.optimisation_tools import Optimization

from utils.plot_solution import plot_map_s
from student_bnn.prob_for_nuts import NUTSForBNN
from utils.quality_metrics import compute_rel_error

from matplotlib import rc


N_DATA = 128
N_DISCR = 130

# data paths
# PATH_CONST = "deconvolution_1d/results/"  # create a new path for saving the results, for example "results... "

RES_PATH_CONST = "results/"  # create a new path for saving the results, for example "results... "
DATA_PATH_CONST = "deconvolution_1d/data/"

# plot parameters:
LOWER_BOUND = 0
UPPER_BOUND = 1

# specify the noise level
NOISE = NoiseLevel.low

# activation function for the hidden layers in BNN
act_func = ActivationFunctions.TANH

# list defining the size of input as a first entity and  number of nodes in the hidden layers
layer_vector = [1, 40, 80, 1]

# How many MAPs we want to compute?
num_maps = 5


def main():
    # specify arguments
    parser = argparse.ArgumentParser(description="Process command line " "arguments")

    parser.add_argument(
        "--problem",
        "-p",
        dest="signal_type",
        choices=[
            SignalType.smooth.value,
            SignalType.datasky.value,
        ],
        default=SignalType.smooth,
        type=SignalType,
        help="Type of the problem/signal: smooth or sharp",
    )

    parser.add_argument(
        "--std",
        "-d",
        dest="std_distr",
        choices=[
            ParamStdDistr.exp.value,
            ParamStdDistr.const.value,
        ],
        default=ParamStdDistr.exp,
        type=ParamStdDistr,
        help="Does std for the BNN parameters follow some distribution (e.g. exponential) or is it constant?",
    )

    args = parser.parse_args()

    print(f"Signal type: {args.signal_type.value}")
    print(f"BNN activation function: {act_func}")
    print(f"BNN architecture: {layer_vector}")
    print(f"Initial std of the BNN parameters: {args.std_distr.value}")

    # handle the info on the dimensionality of a problem
    try:
        layers = Layers(values=layer_vector)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    # set the parameter of the alpha-stable distribution
    signal_type = args.signal_type
    if signal_type == SignalType.smooth:
        alpha = 2
    elif signal_type == SignalType.datasky:
        alpha = 1
    else:
        raise ValueError("Wrong value of alpha (parameter of alpha-stable distribution, should be 1 or 2)")

    # set the data path common for all the samplers
    res_path = (
        RES_PATH_CONST
        + f"{signal_type.value}_signal_{args.std_distr.value}_std/"
        + f'{act_func}_{"_".join(str(x) for x in layers.hidden_layer_dimensions)}_nodes/'
    )

    fig_path = res_path + "figures/"
    # create the path if it does not exist
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # MAP estimate file name
    map_file_name = fig_path + "map_estimates.pkl"

    # load observations from a  file
    data_file_name = DATA_PATH_CONST + f"{signal_type.value}_m_{N_DATA}_n_{N_DISCR}_{NOISE.value}.pkl"
    obs_data = ObservationData(data_file_name, signal_type)

    # dimensionality of the parameter vector
    param_dim = layers.param_vector_dimension

    # input vector (discretization of the  interval [LOWER_BOUND, UPPER_BOUND])
    x_input = torch.tensor(np.linspace(LOWER_BOUND, UPPER_BOUND, N_DISCR), dtype=torch.double).view(1, -1)

    # set the initial standard deviation for the BNN parameters
    if args.std_distr == ParamStdDistr.exp:
        # use random stds for weights and biases in all layers
        std_weights_vec = sps.expon.rvs(loc=0, scale=5, size=layers.num_transforms)
        std_biases_vec = sps.expon.rvs(loc=0, scale=4, size=layers.num_transforms)
    elif args.std_distr == ParamStdDistr.const:
        # use constant value 1 for all the layers
        std_weights_vec = 1
        std_biases_vec = 1
    else:
        raise ValueError("Wrong value of the distribution for the BNN parameter std")

    # create an instance of the class managing the BNN parameter scales
    param_scale = BNNParameterScale(
        layers,
        alpha,
        mu_weights=0,
        mu_biases=0,
        std_weights=std_weights_vec,
        std_biases=std_biases_vec,
    )

    # create an instance of the class containing likelihood, posterior and potential for our problem
    nuts_class = NUTSForBNN(
        x_input,
        obs_data,
        act_func,
        layers,
        param_scale,
    )

    np.random.seed(1)  # remove randomness if needed

    # call the potential
    potential_func = nuts_class.potential

    # initializer the optimizer
    optimizer = Optimization(param_dim, potential=potential_func)

    # Initial guess for the target parameter ksi
    ksi_0 = np.random.randn(num_maps, param_dim)

    # initialize vectors to store MAPs of ksi and potential evaluations
    ksi_map_s = np.empty((num_maps, param_dim))
    potential_bfgs = np.empty(num_maps)
    potential_adam = np.empty(num_maps)

    tic = time.time()
    for i in range(num_maps):
        print(f"Computing MAP: {i + 1}/{num_maps} " + "\n")
        ksi_map_s[i, :], potential_bfgs[i], ksi_adam, potential_adam[i] = optimizer.adam_bfgs(ksi_0[i, :], 5000)
    toc = time.time() - tic
    print(f"Elapsed time optimization: {toc}")

    # find MAPs for theta
    theta_map_s = np.array([param_scale.transform_to_theta(ksi_map_s[i, :])[0] for i in range(num_maps)])

    # find MAPs for u
    u_map_s = np.empty((num_maps, N_DISCR))
    for i in range(num_maps):
        theta_map_torch = torch.tensor(theta_map_s[i, :], dtype=torch.double)
        bnn_torch = BayesianNeuralNet(act_func, layers, theta_map_torch)
        u_map = bnn_torch(x_input).view(-1, 1)
        u_map_s[i, :] = u_map.detach().numpy().squeeze()

    # save MAP to the file
    with open(map_file_name, "wb") as f:
        pickle.dump([ksi_map_s, theta_map_s, u_map_s], f)

    print(f"MAP estimate is saved to: {map_file_name}")

    # plot true solution with MAP estimate
    plot_map_s(obs_data, u_map_s, [LOWER_BOUND, UPPER_BOUND], res_path)

    plt.show()

    return


if __name__ == "__main__":
    main()
