import enum
import numpy as np
import scipy.stats as sps
import pickle
import matplotlib.pyplot as plt
import os

import matplotlib

# to make LaTeX-style plots
matplotlib.rcParams.update({"font.size": 14})
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
matplotlib.rcParams["text.usetex"] = True

# to specify image quality
DPI = 300


class SignalType(str, enum.Enum):
    """
    Class containing different signal types:
    "datasky" = sharp (piecewise constant) signal,
    "smooth" = smooth signal
    """
    datasky = "datasky"
    smooth = "smooth"


class NoiseLevel(str, enum.Enum):
    """
    Class modeling different noise amount:
    "low" = 2% relative noise
    "med" = 5% relative noise
    """
    low = "low"
    med = "med"


NOISE_LEVEL = {
    NoiseLevel.low: 0.02,
    NoiseLevel.med: 0.05,
}


class ObservationData:
    """
    Class to store true signal, observation data, noise, forward matrix.
    Also allows plotting the problem graphically (see plot_problem function).

    signal_type: described by the SignalType class
    domain_discr: (size = problem_dim) domain discretization
    signal_true: (size = problem_dim) ground truth signal (either smooth or piecewise constant function): x
    signal_convolved: (size = problem_dim) result of 1d convolution operation applied to signal_true, data: y = A(x)
    data_discr: (size = n_data) discretization corresponding to the number of data points
    y_data: (size = n_data) data points (without noise):  y_noisy = P(Ax)
    y_data_noisy: (size = n_data) noisy data, signal_convolved corrupted by gaussian noise: y_noisy = P(Ax) + e
    sigma_obs: observational noise parameter
    system_matrix (size = n_data x problem_dim): forward operator matrix: A
    signal_norm: norm of signal_true
    lambda_obs: noise precision lambda_obs = 1 / sigma_obs^2
    problem_dim: number of discretization points
    n_data: number of data points

    """

    signal_type: SignalType
    domain_discr: np.ndarray
    signal_true: np.ndarray
    signal_convolved: np.ndarray
    data_discr: np.ndarray
    y_data: np.ndarray
    y_data_noisy: np.ndarray
    sigma_obs: float
    system_matrix: np.ndarray
    signal_norm: float
    lambda_obs: float
    problem_dim: int
    n_data: int

    def __init__(self, file_name: str, signal_type: SignalType) -> None:
        """
        Load data from the file and initialize the class fields
        :param file_name: name of the file containing the data
        :param signal_type: either sharp or smooth
        """
        with open(file_name, "rb") as f:
            loaded_data = pickle.load(f)

        if len(loaded_data) != 8:
            raise ValueError("Check the number of data items in the file: should be 7")
        (
            self.domain_discr,
            self.signal_true,
            self.signal_convolved,
            self.data_discr,
            self.y_data,
            self.y_data_noisy,
            self.sigma_obs,
            self.system_matrix,
        ) = loaded_data

        self.signal_norm = np.linalg.norm(self.signal_true)
        self.lambda_obs = 1 / (self.sigma_obs**2)  # noise precision
        self.problem_dim = self.domain_discr.shape[0]
        self.n_data = self.data_discr.shape[0]
        self.signal_type = signal_type

    def plot_problem(self, fig_path: str) -> None:
        """
        Makes plot of the ground truth signal and observation data
        :param fig_path:
        :return:
        """

        if self.signal_type == SignalType.smooth:
            y_lims = [-1.2, 8]
        elif self.signal_type == SignalType.datasky:
            y_lims = [-0.2, 1.7]
        else:
            raise ValueError("Wrong signal type: should be either 'datasky' or 'smooth'")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax_1, ax_2 = axes.flatten()

        # ground truth signal plot
        ax_1.plot(self.domain_discr, self.signal_true, "k-")
        ax_1.set_ylim(y_lims[0], y_lims[1])
        ax_1.set_title("True signal")

        # plot convolved signal together with data points (noisy)
        ax_2.plot(self.domain_discr, self.signal_convolved, "k-", label="convolved signal")
        ax_2.plot(self.data_discr, self.y_data_noisy, "b.", label="observations")
        ax_2.set_ylim(y_lims[0], y_lims[1])
        ax_2.set_title("Data")

        # create directory to save figures (if it does not exist)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        plt.savefig(fig_path + "deconv_problem.pdf", dpi=DPI)



