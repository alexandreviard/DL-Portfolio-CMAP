import pandas as pd
import numpy as np
import scipy.stats as sp
from scipy.stats.qmc import Halton, Sobol, LatinHypercube
from numbers import Number 


def simulate_normal_returns(mu, cov, N=1000, n_simul=1):
    """
    génère des returns à l'échelle déterminée par mu et cov 
    N done le nombre de time steps 
    n_simul donne le nombre de simulations 
    shape : [N, n_simul, d_assets] (si n_simul = 1 alors seulement 2 dimensions)
    """
    rets = sp.multivariate_normal(mu, cov).rvs((N, n_simul))
    return rets 

### simulation for alpha stable distributions

def alpha_stable_transformation(U, W, alpha=2, beta=0, gamma=1, delta=0):
    """implement the transformation"""
    if alpha == 1:
        y_bar = (2 / np.pi) * ((np.pi / 2 + beta * U) * np.tan(U) -
                               beta * np.log((np.pi / 2 * W * np.cos(U)) /
                                             (np.pi / 2 + beta * U)))
    else:
        S_a_b = (1 + beta ** 2 * (np.tan(np.pi * alpha / 2)) ** 2) ** (1 / (2 * alpha))
        B_a_b = (1 / alpha) * np.arctan(beta * np.tan(np.pi * alpha / 2))

        term1 = S_a_b * np.sin(alpha * (U + B_a_b)) / (np.cos(U) ** (1 / alpha))
        term2 = (np.cos(U - alpha * (U + B_a_b)) / W) ** ((1 - alpha) / alpha)

        y_bar = term1 * term2 

    y = gamma * y_bar + delta

    return y

def alpha_stable_sampler(alpha=2, beta=0, gamma=1, delta=0, size=None):
    """
    Generates a sample from a univariate α-stable distribution with specified parameters.

    Parameters:
    ----------
    alpha : float, optional
        Stability parameter, must be in the range (0, 2]. Default is 1.
    beta : float, optional
        Skewness parameter, must be in the range [-1, 1]. Default is 0.
    gamma : float, optional
        Scale parameter, must be positive. Default is 1.
    delta : float, optional
        Location parameter, can be any real number. Default is 0.
    size : int or tuple of ints, optional
        Output shape. Default is None, which returns a single sample.

    Returns:
    -------
    y : ndarray or scalar
        Samples from the specified α-stable distribution.
    """
    
    if isinstance(size, int):
        n = size
    elif isinstance(size, tuple):
        n = size[0] * size[1]
    else:
        raise TypeError(f"Expected 'size' to be int or tuple, got {type(size).__name__} instead")
      
    W = np.random.exponential(scale=1, size=size)
    U = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=size)

    return alpha_stable_transformation(U=U, W=W, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
    
def alpha_stable_sampler_QMC(alpha=2, beta=0, gamma=1, delta=0, size=None, sequence="LatinHypercube", dimension_QMC=1, randomized_QMC=False):
    """
    Generates samples from a univariate α-stable distribution using Quasi-Monte Carlo (QMC) sequences.

    Parameters:
    ----------
        Stability parameter, must be in the range (0, 2]. Default is 1.
    beta : float, optional
        Skewness parameter, must be in the range [-1, 1]. Default is 0.
    gamma : float, optional
        Scale parameter, must be positive. Default is 1.
    delta : float, optional
        Location parameter, can be any real number. Default is 0.
    size : int or tuple of int, optional
        Output shape. If int, returns a 1D array of that length. If tuple, returns an array with the given shape. Default is None, which returns a single sample.
    sequence : str, optional
        The QMC sequence to use. Options are "Sobol", "Halton", and "LatinHypercube". Default is "LatinHypercube".
    dimension_QMC : int, optional
        The dimension of the QMC sequence. Default is 1. Increases the number of samples 
    randomized_QMC : bool, optional
        Whether to scramble the QMC sequence (randomized QMC). Default is False. 

    Returns:
    -------
    y : ndarray or scalar
        Samples from the specified α-stable distribution. The shape is determined by the `size` parameter.

    Raises:
    ------
    TypeError
        If `size` is not an int or tuple of ints.
    KeyError
        If `sequence` is not one of the supported QMC sequences.

    Notes:
    -----
    The sampler is not efficient for the non default sequences 

    Examples:
    --------
    >>> alpha_stable_sampler_QMC(alpha=1.5, beta=0.5, gamma=2, delta=1, size=(3, 4))
    array([[ 1.843,  2.312,  1.267,  2.876],
           [ 0.967,  1.548,  2.104,  1.846],
           [ 2.457,  1.746,  2.134,  1.789]])
    """
    if size is None:
        n = 1
    elif isinstance(size, int):
        n = size
    elif isinstance(size, tuple):
        n = size[0] * size[1]
    else:
        raise TypeError(f"Expected 'size' to be int or tuple, got {type(size).__name__} instead")
    
    sequences = {"Sobol": Sobol, "Halton": Halton, "LatinHypercube": LatinHypercube}

    if sequence not in sequences:
        raise KeyError(f"Unsupported QMC sequence: {sequence}. Choose from {list(sequences.keys())}.")

    QMC = sequences[sequence](d=dimension_QMC, scramble=randomized_QMC)
    QMC_sequence = QMC.random(n).ravel()
    U = QMC_sequence * np.pi - np.pi / 2

    # reinitializing the sequence to obtain an independent sequence
    QMC = sequences[sequence](d=dimension_QMC, scramble=randomized_QMC)
    QMC_sequence = QMC.random(n).ravel()    
    W = -np.log(QMC_sequence)

    y = alpha_stable_transformation(U=U, W=W, alpha=alpha, beta=beta, gamma=gamma, delta=delta)

    if isinstance(size, tuple):
        if dimension_QMC > 1:
            y = y.reshape(size[0] * dimension_QMC, size[1])
        else:
            y = y.reshape(size)
    return y

if __name__ == "__main__":
    print(alpha_stable_sampler(size=(1000)))
