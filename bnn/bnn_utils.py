import scipy.stats as sps
import numpy as np
import enum
from typing import List, Callable, Optional, Tuple

import matplotlib.pyplot as plt


class DistributionType(str, enum.Enum):
    CAUCHY = "cauchy"
    GAUSSIAN = "gaussian"
    CAUCHY_GAUSSIAN = "cauchy-gaussian"  # Cauchy hidden weights/biases & Gaussian hidden-to-output weights
    GAUSSIAN_CAUCHY = "gaussian-cauchy"  # Gaussian hidden weights/biases & Cauchy hidden-to-output weights


ALPHA = {
    DistributionType.CAUCHY: 1,
    DistributionType.GAUSSIAN: 2,
}


class Layers:
    # class Layers(pydantic.BaseModel):
    # values: pydantic.conlist(int, min_items=2) = pydantic.Field(
    #     default_factory=list,
    #     description="List defining the size of input as a first entity and number of nodes in the hidden layers",
    # )

    def __init__(self, values: list):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]

    def __len__(self):
        return len(self.values)

    @property
    def input_dimension(self) -> int:
        return self.values[0]

    @property
    def hidden_layer_dimensions(self) -> List[int]:
        return self.values[1:-1]

    @property
    def input_and_hidden_dimension(self) -> List[int]:
        return self.values[:-1]

    @property
    def num_hid_layers(self) -> int:
        return len(self.hidden_layer_dimensions)

    @property
    def num_transforms(self) -> int:
        return self.num_hid_layers + 1

    def get_weight_size(self, layer_number: int) -> int:
        return self.values[layer_number] * self.values[layer_number + 1]

    def get_bias_size(self, layer_number: int) -> int:
        return self.values[layer_number + 1]

    def get_weight_matrix_dimensions(self, w_mat_ind) -> [int, int]:

        dim_1 = self.values[w_mat_ind + 1]
        dim_2 = self.values[w_mat_ind]
        return dim_1, dim_2

    @property
    def get_weights_partition(self):
        """
        Computes the partition of the vector of weights
        that is a compilation of all weight matrices vectorized and stacked together [W^(0), W^(1), ... ]
        :return:
        """

        w_ind_pointers = [0]  # vector to store indices for weights
        current_ind_w = 0

        for i in range(self.num_transforms):
            # find dimension of the current weight matrix:
            # for W^(i), i=0, 1, ..., - in total number of transitions: dim_1 = values[i+1], dim_2 = values[i]
            dim_1, dim_2 = self.get_weight_matrix_dimensions(i)

            # calculate how many elements are in the weight matrix
            num_of_weights = dim_1 * dim_2

            # upadte the pointer for the weights
            current_ind_w += num_of_weights
            w_ind_pointers.append(current_ind_w)

        return np.array(w_ind_pointers)

    @property
    def get_total_num_weights(self):
        """Computes how many weights are in total in the neural network"""
        return self.get_weights_partition[-1]

    @property
    def get_biases_partition(self):

        # get sizes of the vectors of biases
        # they are the elements  of the "values" vector except for the first entity
        biases_sizes = self.values[1:]

        # pointers of the partition of the bias vector
        bias_pointers = [0] + biases_sizes

        return np.cumsum(bias_pointers)

    @property
    def get_total_num_biases(self):
        return self.get_biases_partition[-1]

    @property
    def get_relative_biases_partition(self):

        last_weight_ind = self.get_weights_partition[-1]

        bias_partition_independent = self.get_biases_partition

        relative_bias_pointers = last_weight_ind + bias_partition_independent

        return relative_bias_pointers

    @property
    def get_param_vec_partition(self) -> np.ndarray:

        """Gets parameter vector partition  in case when all the weights go firs and then biases,
        i.e. [w^0, w^2, ..., w^(n_trans - 1), b^0, b^2, ... b^(n_trans - 1)]
        """

        # first we recall the partition of the weights
        w_pointers = self.get_weights_partition

        # get partition indices of biases
        relative_bias_pointers = self.get_relative_biases_partition

        param_pointers = np.concatenate((w_pointers, relative_bias_pointers[1:]))

        return param_pointers

    @property
    def param_vector_dimension(self) -> int:
        return self.get_param_vec_partition[-1]


class BNNParameterScale:
    def __init__(
        self,
        layers: Layers,
        alpha: float,
        mu_weights: {float, np.ndarray},
        mu_biases: {float, np.ndarray},
        std_weights: {float, np.ndarray},
        std_biases: {float, np.ndarray},
    ):

        """

        :param layers: class describing the structure of the BNN, number of layers/parameters, parameter partition
        :param alpha: parameter of the alpha-stable distribution: alpha = 1 => Cauchy distribution,
                                                                  alpha = 2 => Gaussian distribution
        :param mu_weights: location parameter for the weights, can be a constant or a vector
        :param mu_biases: location parameter for the biases
        :param std_weights: standard deviation of the weights
        :param std_biases: standard deviation of the biases
        """
        self.layers = layers

        self.alpha = alpha

        self.mu_weights = mu_weights
        self.mu_biases = mu_biases
        self.std_weights = std_weights
        self.std_biases = std_biases
        self._prior_loc = None
        self._prior_scale = None

    # @property
    def get_prior_loc_and_scale(
        self,
        inferred_inds: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function produces four vectors:
            1) vector containing loc param for each of the weights
            <= from param specifying loc for weights in each layer/or even in whole NN altogether
            2) vector containing loc param for each of the biases
            3) vector containing scale param for each of the weights
            4) vector containing scale param for each of the biases

        :return:
        """

        # get number of transformations (weights and biases)
        n_trans = self.layers.num_transforms

        # # TODO: might not need this vectorization
        # # make a vector of locations and a vector of scales from one value
        # # E.g. mu_weights_vec = [mu_weights_0, ..., mu_weights_{n_L - 1}],
        # # where n_L is the number of weight transformations
        if not hasattr(self.mu_weights, "__len__"):
            mu_weights_vec = self.mu_weights * np.ones(n_trans)
        else:
            mu_weights_vec = self.mu_weights

        if not hasattr(self.mu_biases, "__len__"):
            mu_biases_vec = self.mu_biases * np.ones(n_trans)
        else:
            mu_biases_vec = self.mu_biases

        if not hasattr(self.std_weights, "__len__"):
            std_weights_vec = self.std_weights * np.ones(n_trans)
        else:
            std_weights_vec = self.std_weights

        if not hasattr(self.std_biases, "__len__"):
            std_biases_vec = self.std_biases * np.ones(n_trans)
        else:
            std_biases_vec = self.std_biases

        # compute scale factors for weights (proportional to te number of hidden units)
        inp_hid_units = np.array(self.layers.input_and_hidden_dimension)
        scale_factor = 1 / (inp_hid_units ** (1 / self.alpha))

        # get weights and biases vector partitions
        w_ind = self.layers.get_weights_partition
        b_ind = self.layers.get_biases_partition

        # get total number of all weights and all biases
        w_dim = self.layers.get_total_num_weights
        b_dim = self.layers.get_total_num_biases

        # create empty vectors containing scales and locations for all the weights and biases elements
        w_locations = np.empty(w_dim)
        b_locations = np.empty(b_dim)
        w_scales = np.empty(w_dim)
        b_scales = np.empty(b_dim)

        for i in range(n_trans):
            # assign the location parameter to each component of the weights and biases vectors
            # do not need to do any additional scaling
            w_locations[w_ind[i] : w_ind[i + 1]] = mu_weights_vec[i]
            b_locations[b_ind[i] : b_ind[i + 1]] = mu_biases_vec[i]

            # assign the scale parameter to each component of the weights vector
            # need to apply ADDITIONAL SCALING
            w_scales[w_ind[i] : w_ind[i + 1]] = std_weights_vec[i] * scale_factor[i]

            # # assign the scale parameter to each component of the biases vector
            # do not need to do any additional scaling
            b_scales[b_ind[i] : b_ind[i + 1]] = std_biases_vec[i]

        prior_loc = np.hstack([w_locations, b_locations])
        prior_scale = np.hstack([w_scales, b_scales])

        # plt.figure()
        # plt.plot(b_scales)
        # plt.show()

        if inferred_inds is not None:
            prior_loc = prior_loc[inferred_inds]
            prior_scale = prior_scale[inferred_inds]

        return prior_loc, prior_scale

    def scale_gaussian(self, ksi: np.ndarray):
        """
        Transformation for the case alpha = 2 (Gaussian distribution),
        just scaling according to the distribution mean and standard deviation
        :param ksi:
        :return: theta = transformation(ksi), gradient of the transformation w.r.t. ksi
        """

        # plt.figure()
        # plt.plot(self._prior_scale)
        # plt.show()

        assert (
            self._prior_scale is not None
            and len(self._prior_scale) > 0
            and self._prior_loc is not None
            and len(self._prior_loc) > 0
        ), "_prior_scale and _prior_scale parameters are not initialized"

        theta = self._prior_loc + self._prior_scale * ksi

        grad = self._prior_scale
        return theta, grad

    def transform_gaussian_to_cauchy(self, ksi: np.ndarray):
        """
        Transformation for the case alpha = 1 (Cauchy distribution),
        ksi ~ Gaussian ---> theta ~ Cauchy
        :param ksi:
        :return: theta = transformation(ksi), gradient of the transformation w.r.t. ksi
        """

        # plt.figure()
        # plt.plot(self._prior_scale)
        # plt.show()

        assert (
            self._prior_scale is not None
            and len(self._prior_scale) > 0
            and self._prior_loc is not None
            and len(self._prior_loc) > 0
        ), "_prior_scale and _prior_scale parameters are not initialized"

        term = np.pi * (sps.norm.cdf(ksi) - 0.5)
        theta = self._prior_loc + self._prior_scale * np.tan(term)

        grad = self._prior_scale * (1 / (np.cos(term) ** 2)) * np.pi * sps.norm.pdf(ksi)

        return theta, grad

    def transform_to_theta(self, ksi: np.ndarray, inferred_inds=None):
        """
        Performs transformation ksi ---> theta based on the values of alpha
        If alpha == 1: transform_to_theta is transform_gaussian_to_cauchy,
        Elseif alpha == 2: transform_to_theta is scale_gaussian
        :param ksi:
        :param inferred_inds: if this parameter is not None - the BNN is partially stochastic
        :return:
        """
        self._prior_loc, self._prior_scale = self.get_prior_loc_and_scale(inferred_inds)
        if self.alpha == 2:  # if Gaussian case
            theta, grad = self.scale_gaussian(ksi)
        elif self.alpha == 1:  # if Cauchy case
            theta, grad = self.transform_gaussian_to_cauchy(ksi)
        else:
            raise ValueError("Wrong value of alpha (parameter of alpha-stable distribution, should be 1 or 2)")

        return theta, grad


def get_limit_scaling(layers: Layers, alpha: float) -> np.ndarray:
    """
    Computes the so-called limit scaling that depends on the number of hidden nits in the NN
    :param layers: defines the NN structure
    :param alpha: parameter of alpha stable, if alpha = 1 ==> Cauchy distribution
    :return:
    """
    # compute scale factors for weights (proportional to te number of hidden units)
    inp_hid_units = np.array(layers.input_and_hidden_dimension)
    scale_factor = 1 / (inp_hid_units ** (1 / alpha))

    # compute dimension of the parameter vector
    d = layers.param_vector_dimension

    # get number of transformations (weights and biases)
    n_trans = layers.num_transforms

    # initialize the future vector of limit scales:
    # all its components are = 1 except for those corresponding to the weights
    # that need to be scaled by 1 / (n_l)^(1/alpha)
    limit_scaling = np.ones(d)

    # now we possibly need to change the scaling only for the entities of the vector limit_scaling
    # corresponding to the weights

    # get weights partitions
    w_ind = layers.get_weights_partition

    for i in range(n_trans):
        print(f"From index {w_ind[i]} to index {w_ind[i+1]}")
        limit_scaling[w_ind[i] : w_ind[i + 1]] = limit_scaling[i] * scale_factor[i]

    # plt.figure()
    # plt.plot(limit_scaling)
    # plt.show()

    return limit_scaling


class PartiallyStochastic:
    def __init__(self, layers: Layers, stoch_frac: float, theta_map):
        """
        Class initializer
        :param layers: sets the structure of the BNN
        :param stoch_frac: fraction of stochastic parameters
        """
        self.total_param_dim = layers.param_vector_dimension
        if stoch_frac < 0 or stoch_frac > 1:
            raise ValueError("Stochastic fraction should be between 0 and 1")
        else:
            self.stoch_frac = stoch_frac
        self.num_inferred = int(self.stoch_frac * self.total_param_dim)
        self.num_fixed = self.total_param_dim - self.num_inferred
        self.inferred_indices, self.fixed_indices = self.set_params_for_part_stochastic
        self.theta_fixed = theta_map[self.fixed_indices]

    @property
    def set_params_for_part_stochastic(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Defines (randomly) which parameters are going to be stochastic and which --- fixed
        :return: indices of the parameters that will be inferred  and indices of parameters that will be fixed to MAP
        """

        # define which indices are inferred
        inferred_indices = np.sort(
            np.random.choice(np.arange(0, self.total_param_dim), self.num_inferred, replace=False)
        )

        # set up the indices of the parameters that will be fixed
        # as the set difference between indices of all parameter and those that will be inferred
        all_indices = np.arange(0, self.total_param_dim)
        fixed_indices = np.setdiff1d(all_indices, inferred_indices)

        return inferred_indices, fixed_indices
