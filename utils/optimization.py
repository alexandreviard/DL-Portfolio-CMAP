import numpy as np 
from scipy.optimize import minimize, Bounds 


# fonctions qui calculent les poids de la strat 



def diversification_ratio(w, cov):
    sigma = np.sqrt(np.diag(cov))
    wT_sigma = w.T @ sigma 
    wT_cov_w = w.T @ cov @ w 
    return wT_sigma / (np.sqrt(wT_cov_w))



def get_weights_md(cov, bounds=None):
    """
    We get the weights for period t by minimizing the negative diversification ratio with cov matrix computed at time (t - 1)  
    """
    n_assets = cov.shape[0]
    w0 = np.ones(n_assets) / n_assets 
    
    # Defining the loss and the weight constraint
    def loss(w, cov):
        return - diversification_ratio(w, cov)
    
    def constraint(w):
        return w.sum() - 1 
    
    cons = {'type': 'eq', 'fun': constraint}
    res = minimize(
        fun = loss,
        x0 = w0,
        constraints=cons,
        args=cov,
        bounds=bounds,
        method='SLSQP'
    )
    return res.x  