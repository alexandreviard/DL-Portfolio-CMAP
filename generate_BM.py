import pandas as pd
import numpy as np
from scipy.linalg import toeplitz



# def C(i,j,H):
#     return (1/2) * (np.abs(i) ** (2 * H) + np.abs(j) ** (2 * H) 
#     - np.abs(i - j) ** (2 * H))

def C_matrix(T, H):
    s = np.arange(1, T+1) ** (2 * H)
    d = np.arange(T) ** (2 * H)
    D = toeplitz(d)
    C_matrix = 0.5 * (s[:, None] + s[None, :] - D)
    return C_matrix

# def gen_fbm(T, H):
#     X = np.random.normal(0, 1, size=T)
#     matrix = C_matrix(T, H)
#     L = np.linalg.cholesky(matrix)
#     y = L @ X
#     return y / (T ** H)

def gen_N_fbm(T=1000, H=0.5, N=1):
    """
    Génère N Brownien si on prend H=0.5 sur T temps
    """
    X = np.random.normal(0, 1, size=(T, N))
    matrix = C_matrix(T, H)
    L = np.linalg.cholesky(matrix)
    y = L @ X
    return y / (T ** H)

def variogram(tau, df):
    """
    Il faut que les colonnes soient les réalisations 
    et la ligne le tps
    """
    return ((df - df.shift(tau)).dropna() ** 2).mean(axis=0).mean()