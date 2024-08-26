import numpy as np
from scipy.fftpack import fft


def compute_rel_error(ref_reco: np.ndarray, reco: np.ndarray) -> float:

    """
    Function to compute relative reconstruction error (used as a quality metric) as
    eps_rel = ||img - ref_img||_2 / ||ref_img||_2,
    :param ref_reco: array containing the reference object, e.g. true signal/image
    :param reco: array representing solution/reconstruction
    :return: relative reconstruction error
    """

    return np.linalg.norm(reco - ref_reco) / np.linalg.norm(ref_reco)


def compute_iact(chain):
    """
    Adaptation of the source code for pymcmcstat.chain.ChainStatistics <
    Computes the integrated autocorrelation time using Sokal's
    adaptive truncated periodogram estimator.

    :param chain: sampling chain
    :return: autocorrelation time (tau) and counter (m)
    """

    # number of samples in the chain
    chain_len = len(chain)

    # initialize arrays
    tau = 0  # iact value
    m = 0  # counter

    # apply FFT
    x = fft(chain, axis=0)

    # real part of transformed chain
    xr = np.real(x)

    # imaginary part of the transformed chain
    xi = np.imag(x)

    xmag = xr**2 + xi**2

    xmag[0] = 0.0

    xmag = np.real(fft(xmag, axis=0))
    var = xmag[0] / chain_len / (chain_len - 1)

    if var == 0:
        print("Warning: variance = 0")

    xmag = xmag / xmag[0]
    snum = - 1 / 3
    for ii in range(chain_len):
        snum = snum + xmag[ii] - 1 / 6
        if snum < 0:
            tau = 2 * (snum + ii / 6)
            m = ii + 1
            break

    return tau, m
