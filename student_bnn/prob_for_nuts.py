import dataclasses
import enum
import torch
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
from tqdm import tqdm
from scipy.optimize import minimize, minimize_scalar
from scipy.special import loggamma
import scipy as sp
import scipy.stats as sps

from scipy.sparse.linalg import inv

from deconvolution_1d.deconv_bnn import ObservationData
from bnn.bnn_pytorch import BayesianNeuralNet, ActivationFunctions
from bnn.bnn_utils import DistributionType, Layers, BNNParameterScale

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class NUTSForBNN:
    def __init__(
        self,
        x_input: torch.Tensor,
        obs_data: ObservationData,
        act_func: ActivationFunctions,
        layers: Layers,
        param_scale: BNNParameterScale,
    ):
        """
        Class initializer
        :param x_input: input space discretization
        :param obs_data: class containing all the info about the available data, system matrix, noise
        :param act_func: activation function used in the neural network
        :param layers: class describing the structure of the BNN, number of layers/parameters, parameter partition
        """
        self.x_input = x_input
        self.layers = layers
        self.problem_dim = obs_data.problem_dim
        self.param_dim = self.layers.param_vector_dimension
        self.signal_type = obs_data.signal_type
        self.act_func = act_func
        self.param_scale = param_scale

        self.n_data = obs_data.n_data
        self.data = obs_data.y_data_noisy
        self.domain_disc = obs_data.domain_discr
        self.signal_true = obs_data.signal_true
        self.lambda_obs = obs_data.lambda_obs
        self.sigma_obs = obs_data.sigma_obs
        self.skip = int((self.problem_dim - 3) / (self.n_data - 1))
        self.forward_matrix = obs_data.system_matrix[1 : -1 : self.skip]

    def log_likelihood(self, ksi: np.ndarray, eval_grad=False):

        """
        Function that self the log-likelihood probability for the BNN
        :param ksi:
        :param eval_grad:
        :return:
        """

        # do transform ksi to theta
        theta, grad_of_theta_transform = self.param_scale.transform_to_theta(ksi)

        # change type of the NN parameters from numpy to torch
        params = torch.tensor(theta, dtype=torch.double, requires_grad=True)

        # create a BNN with given params
        bnn_torch = BayesianNeuralNet(self.act_func, self.layers, params)

        # apply the neural network to get the u field (function)
        u_val = bnn_torch(self.x_input).view(-1, 1)

        # change type for the forward matrix from numpy to pytorch
        conv_forward_mat = torch.tensor(self.forward_matrix, dtype=torch.double, requires_grad=True)

        # apply forward operator to u field to obtain predictions
        y_pred = conv_forward_mat.mm(u_val)

        # change the type of observation data from numpy to pytorch
        y_obs = torch.tensor(self.data, dtype=torch.double).view(-1, 1)

        # compute the data misfit term
        misfit = y_pred - y_obs
        # and its transpose
        misfit_tr = misfit.T

        # change the type of lambda_obs from numpy to pytorch
        lambda_obs = torch.tensor(self.lambda_obs, dtype=torch.double)

        # compute log-likelihood
        log_like_eval = -0.5 * lambda_obs * misfit_tr.mm(misfit)

        if not eval_grad:
            log_like_eval = log_like_eval.detach().numpy()
            # TODO: initially had only log-like value to return
            u_func = u_val.view(-1).detach().numpy()
            return log_like_eval[0][0], u_func, theta  # for the gradient check need to comment out u_func, theta

        else:
            # now we need to compute the gradients
            log_like_eval.backward()

            # collect the gradients
            grad_log_like_eval = np.empty((self.param_dim, 1))
            param_part = self.layers.get_param_vec_partition
            i = 0
            for param in bnn_torch.parameters():
                # print("param")
                # first: all weights, next: all the biases
                param_grad = param.grad.view(-1, 1).detach().numpy()
                grad_log_like_eval[param_part[i] : param_part[i + 1], :] = param_grad
                i += 1

            # # cast the log-likelihood and its gradient from pytorch to numpy
            log_like_eval = log_like_eval.detach().numpy()

            # return evaluated log-likelihood, and its gradient
            grad_log_like = grad_log_like_eval.squeeze()

            # MULTIPLY BY GRADIENT OF THE THETA TRANSFORM
            grad_log_like = grad_of_theta_transform * grad_log_like

            return log_like_eval[0][0], grad_log_like

    def log_prior(self, ksi: np.ndarray, eval_grad=True):

        """
        Defines prior on the parameter ksi ~ standard Gaussian
        :param ksi:
        :param eval_grad: parameter indicating if there is a need in returning the gradients
        :return:
        """

        log_prior_eval = -0.5 * (ksi.T @ ksi)
        grad_log_prior = -ksi

        if eval_grad:
            return log_prior_eval, grad_log_prior
        else:
            return log_prior_eval

    def log_posterior(self, ksi: np.ndarray, eval_grad=True):
        """
        Logarithm of the posterior distribution = log-prior + log-likelihood
        :param ksi:
        :param eval_grad: parameter indicating if there is a need in returning the gradients
        :return:
        """
        if eval_grad:
            # log-likelihood
            log_like_eval, grad_log_like = self.log_likelihood(ksi, eval_grad)

            # log-prior
            log_prior_eval, grad_log_prior = self.log_prior(ksi, eval_grad)

            log_post_eval = log_like_eval + log_prior_eval
            grad_log_post = grad_log_like + grad_log_prior

            return log_post_eval, grad_log_post
        else:

            # log-likelihood
            log_like_eval = self.log_likelihood(ksi, eval_grad)

            # log-prior
            log_prior_eval = self.log_prior(ksi, eval_grad)

            log_post_eval = log_like_eval + log_prior_eval

            return log_post_eval

    def potential(self, ksi: np.ndarray, eval_grad=True):
        """
        Potential is negative of log-posterior
        :param ksi:
        :param eval_grad: parameter indicating if there is a need in returning the gradients
        :return:
        """
        if eval_grad:
            log_post_eval, grad_log_post = self.log_posterior(ksi, eval_grad)

            return -log_post_eval, -grad_log_post

        else:
            log_post_eval = self.log_posterior(ksi, eval_grad)

            return -log_post_eval
