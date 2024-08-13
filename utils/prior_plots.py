import torch
import matplotlib.pyplot as plt
import numpy as np
import math

from typing import List

from bnn.bnn_utils import DistributionType, Layers, BNNParameterScale, ALPHA
from bnn.bnn_pytorch import BayesianNeuralNet, ActivationFunctions

# from student_bnn.prob_for_nuts import NUTSForBNN

DPI = 300


def plot_different_priors(
    input_dim: int,
    x_input: torch.Tensor,
    distr_type: DistributionType,
    act_func: ActivationFunctions,
    n_units: List,
    n_layers: List,
    data_path: str,
    n_realisations: int,
    x_lims: List,
    mu_weights: float,
    mu_biases: float,
    std_weights: float,
    std_biases: float,
) -> None:
    """
    Function to create a matrix plot showing Bayesian neural network (BNN) priors
    in connection to the number of hidden layers and number of hidden units
    :param input_dim: BNN input dimension: 1 (for a signal) or 2 (for an image)
    :param x_input: discretization of the domain
    :param distr_type: prior distribution for the weights and biases (either Cauchy or Gaussian)
    :param act_func: NN activation function type
    :param n_units: number of units in each of the hidden layers, e.g., [25, 50, 100]
    :param n_layers: list containing numbers of hidden layers, e.g., [1, ..., 5]
    :param data_path: where to save the result plots
    :param n_realisations: number of BNN prior realizations in each subplot
    :param x_lims: limits for the x-axes in the plot
    :param mu_weights: location parameter for the weights, can be a constant or a vector
    :param mu_biases: location parameter for the biases
    :param std_weights: standard deviation of the weights
    :param std_biases: standard deviation of the biases

    :return:
    """

    layers_size = len(n_layers)  # number of rows in the matrix plot
    units_size = len(n_units)  # number of columns in the matrix plot

    # set parameter used for scaling weights: alpha = 1 for Cauchy distribution, alpha = 2 for Gaussian distribution
    alpha = ALPHA[distr_type]

    # plot specifications
    fig = plt.figure(figsize=(15, 9))
    fig.subplots_adjust(wspace=0.6, hspace=0.7)

    # set the colormap for the plot
    custom_cmap = plt.cm.get_cmap("coolwarm", n_realisations)
    colors = custom_cmap(range(n_realisations))

    for i, num_l in enumerate(n_layers):  # rows
        for j, num_u in enumerate(n_units):  # columns
            # create list representing number of units num_u in each of num_l hidden layers
            hid_layers = [num_u] * num_l
            layer_units = [input_dim] + hid_layers + [1]

            print(f"Hidden layers structure: {hid_layers}")
            layers = Layers(values=layer_units)

            # create subplot
            ax = plt.subplot(layers_size, units_size, i * units_size + j + 1)

            # create an instance of the BNNParameterScale class (needed to obtain scaling on the NN weights/biases)
            param_scale = BNNParameterScale(layers, alpha, mu_weights, mu_biases, std_weights, std_biases)

            # draw n_realisations realisations from the BNN prior distribution
            for jj in range(n_realisations):

                # create random vector ksi ~ N(0,1)
                ksi = np.random.randn(layers.param_vector_dimension)

                # transform ksi to theta based on the value of alpha and initial scales/locations of the NN parameters
                theta = param_scale.transform_to_theta(ksi)[0]
                params = torch.tensor(theta, dtype=torch.double, requires_grad=True)

                # BNN prior
                cauchy_bnn = BayesianNeuralNet(act_func, layers, params)
                u_func = cauchy_bnn(x_input)

                if input_dim == 1:  # if it is a 1D case (signal)
                    ax.plot(x_input.detach().view(-1), u_func.detach().view(-1), color=colors[jj])
                    ax.set_xlim([x_lims[0], x_lims[1]])

                elif input_dim == 2:  # if it is a 2D case (an image)
                    n_discr = np.int(np.sqrt(x_input.size(dim=1)))
                    u_func = u_func.view(n_discr, n_discr).detach().numpy()

                    # make a plot - one realisation of a NN-based prior
                    ax.imshow(u_func, cmap="Greys")
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    ax.tick_params(left=False, bottom=False)

            if i == 0:
                ax.set_title(f"Units: {num_u}")

            if j == 0:
                ax.set_ylabel(f"Layers: {num_l}", fontsize=33)

    # save the figure and plot
    fig.align_labels()
    plt.tight_layout()
    plt.savefig(data_path + f"{input_dim}d_{distr_type.value}_prior_{act_func}_act_func.pdf", dpi=DPI)
    plt.show()

    return


def make_bnn_convergence_plot(data_path: str):
    """
    Function to check that the output of BNN (with Gaussian parameters) converges to Gaussian distribution as
    the number of units in the hidden layer increases.
    We use neural network with one hidden layer and only increase the number of units in the hidden layer
    :param data_path:
    :return:
    """
    # domain
    n_discr = 2
    x_input = torch.tensor([-0.2, 0.4], dtype=torch.double).view(1, -1)

    # number of neural networks to generate
    n_networks = 5000

    # options for number of units in the hidden layer
    hidden_units = [1, 3, 10]

    # to make a plot
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plt.tight_layout(w_pad=2.5, pad=1.5)

    # size of the font in the plot
    font_size = 32

    # go through different number of units in the hidden layer
    for kk in range(len(hidden_units)):
        # generate NN with given number of hidden units in the hidden layer
        layers = Layers(values=[1, hidden_units[kk], 1])

        # matrix to store output values of the NNs - n_discr x n_networks
        y_out = np.empty((n_discr, n_networks))

        param_dim = layers.param_vector_dimension

        hid_layers_dims = layers.hidden_layer_dimensions

        # get param vector partition
        param_partition = layers.get_param_vec_partition

        # generate multiple networks
        for jj in range(n_networks):
            # find dimension of the parameter vector

            params = torch.zeros((param_dim, 1), dtype=torch.double)

            # generate input-to-hidden weights (sample from Gaussian distribution)
            gaussian = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([5.0]))
            in_to_hid_weights = gaussian.sample(sample_shape=(param_partition[1], 1)).view(-1, 1)
            in_to_hid_biases = gaussian.sample(
                sample_shape=torch.Size([param_partition[2] - param_partition[1], 1])
            ).view(-1, 1)

            # generate hidden-to-output weights (sample from Gaussian distribution)
            hid_to_out_size = hid_layers_dims[-1]
            gaussian_out_w = torch.distributions.normal.Normal(
                loc=torch.tensor([0.0]), scale=torch.tensor([1.0 / math.sqrt(hid_to_out_size)])
            )
            gaussian_out_b = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.1]))
            hid_to_out_weights = gaussian_out_w.sample(sample_shape=torch.Size([hid_to_out_size, 1])).view(-1, 1)
            hid_to_out_biases = gaussian_out_b.sample(sample_shape=torch.Size([1, 1])).view(-1, 1)

            # set the parameter vector (stack weights and biases in one vector)
            params[param_partition[0] : param_partition[1], :] = in_to_hid_weights
            params[param_partition[1] : param_partition[2], :] = hid_to_out_weights
            params[param_partition[2] : param_partition[3], :] = in_to_hid_biases
            params[param_partition[3] : param_partition[4], :] = hid_to_out_biases

            params = params.type("torch.DoubleTensor")

            # create Bayesian neural network (BNN) with given activation function, layer structure and parameters
            bnn_torch = BayesianNeuralNet(ActivationFunctions.TANH, layers, params)

            # evaluate BNN for a given input x_input, that is, get a realisation from the BNN prior
            y_out[:, jj] = bnn_torch(x_input).detach().view(-1)

        # plot realisation from the prior
        ax[kk].plot(y_out[0, :], y_out[1, :], "o", color="darkblue", markersize=1.5)
        plot_lim = 3
        ax[kk].set_xlim(-plot_lim, plot_lim)
        ax[kk].set_ylim(-plot_lim, plot_lim)
        ax[kk].set_title(rf"$n_1 = {hidden_units[kk]}$", fontsize=font_size)
        ax[kk].set_xlabel(r"$x(t=-0.2)$", fontsize=font_size)
        ax[kk].set_ylabel(r"$x(t=0.4)$", fontsize=font_size)
        ax[kk].set_aspect("equal", adjustable="box")
        ax[kk].tick_params(axis="both", which="major", labelsize=font_size)
        ax[kk].tick_params(axis="both", which="minor", labelsize=font_size)

    # save the figure and display it
    plt.savefig(data_path + f"bnn_convergence.pdf", dpi=DPI)
    plt.show()

    return


