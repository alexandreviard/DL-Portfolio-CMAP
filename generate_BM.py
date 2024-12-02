import pandas as pd
import numpy as np
from scipy.linalg import toeplitz

####### méthode 1 #####

def gen_BM(T=1, n_increments=1000):
    """
    size (n_increments + 1)
    """
    if T == 0 or n_increments == 0: 
        return np.array([0])  
    
    dt = T / n_increments
    dW = np.sqrt(dt) * np.random.normal(size=n_increments)
    cumsum_incr = np.cumsum(dW)

    return np.concatenate((np.array([0]), cumsum_incr))

def Ito(f = lambda x : 1, T=1, N=1000):
    """
    cvg en proba vers l'intégrale de 0 à T de f contre W 
    si f suffisamment régulière
    """

    if N == 0:
        return 0 

    dt = T / N
    t_n = np.arange(N) * dt
    f_vector = np.array([f(t) for t in t_n])
    dW = np.sqrt(dt) * np.random.normal(size=N)
    return f_vector @ dW

### approx d'une diffusion avec le schema d'Euler 

def euler_maruyama(mu, sigma, x0, T, N):
    dt = T / N
    t = np.linspace(0, T, N + 1)  
    X = np.zeros(N + 1)           
    X[0] = x0                     
    for i in range(1, N + 1):
        dW = np.sqrt(dt) * np.random.randn()  
        X[i] = X[i - 1] + mu * X[i - 1] * dt + sigma * X[i - 1] * dW
    return t, X



####### méthode 2 - génére des FBM #####

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

