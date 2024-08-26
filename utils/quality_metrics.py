import numpy as np


def compute_rel_error(ref_reco: np.ndarray, reco: np.ndarray) -> float:

    """
    Function to compute relative reconstruction error (used as a quality metric) as
    eps_rel = ||img - ref_img||_2 / ||ref_img||_2,
    :param ref_reco: array containing the reference object, e.g. true signal/image
    :param reco: array representing solution/reconstruction
    :return: relative reconstruction error
    """

    return np.linalg.norm(reco - ref_reco) / np.linalg.norm(ref_reco)
