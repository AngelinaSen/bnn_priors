import numpy as np
import torch
import enum

import matplotlib
from bnn.bnn_utils import Layers


# to make LaTeX-style plots
matplotlib.rcParams.update({"font.size": 18})
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
matplotlib.rcParams["text.usetex"] = True


class ActivationFunctions(str, enum.Enum):
    TANH = "tanh"


ACTIVATION_FUNCTIONS = {ActivationFunctions.TANH: lambda x: torch.tanh(x)}


class BayesianNeuralNet(torch.nn.Module):
    def __init__(
        self,
        act_func_type: ActivationFunctions,
        layers: Layers,
        params: torch.Tensor,
    ):

        # initialise the parent class
        super().__init__()

        # parameter vector partitions (separately for weights and biases )
        weights_ind_pointers = layers.get_weights_partition
        biases_ind_pointers = layers.get_relative_biases_partition

        # number of hidden layers + 1
        self.num_transforms = layers.num_transforms

        # initialize parameters
        self.bnn_params = torch.nn.ParameterDict()

        # activation function
        self.act_func_type = act_func_type

        # number of units in hidden layers
        layers_dims = layers.values

        for jj in range(self.num_transforms):

            weights_jj = params[weights_ind_pointers[jj] : weights_ind_pointers[jj + 1]].reshape(
                layers_dims[jj + 1], layers_dims[jj]
            )

            self.bnn_params[f"weights_{jj + 1}"] = torch.nn.Parameter(weights_jj)

        for jj in range(self.num_transforms):
            biases_jj = params[biases_ind_pointers[jj] : biases_ind_pointers[jj + 1]].reshape(layers_dims[jj + 1], 1)

            self.bnn_params[f"biases_{jj + 1}"] = torch.nn.Parameter(biases_jj)

    def forward(self, x_input):

        """
        Function performing the forward pass for NN
        # Note: we do not incorporate  weight scaling to the NN formulation,
        # instead we scale weights separately in the outer function,
        # so the input weights for this function are assumed to be scaled  already
        # using the scale_limit  =  1 / (number_of_hidden_units)^{1/alpha},
        # where alpha = 1 for Cauchy and alpha = 2 for Gaussian
        :param x_input: space discretisation
        :return: output of the NN
        """

        # input layer
        f = x_input

        # set the activation function
        activation = ACTIVATION_FUNCTIONS[self.act_func_type]

        for j in range(self.num_transforms):
            # firstly, we consider hidden layers:
            # set weights/biases based on the hidden layer number

            # for Gibbs and NUTS no cauchy transform
            weights = self.bnn_params[f"weights_{j + 1}"]
            biases = self.bnn_params[f"biases_{j + 1}"]

            # create linear transform + apply activation function (based on the chosen type)
            if j == 0:  # if it is the output of the first hidden layer we do not scale it!
                h = weights.mm(f)
                f = h + biases

            else:
                act = activation(f)
                h = weights.mm(act)
                f = h + biases

        return f
