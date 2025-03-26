import matplotlib.pyplot as plt
import numpy as np

def plot_markowitz(weights, returns, figsize=(15, 3), savefig=False, save_path="portfolio_markowitz_plot.png"):
    """
    Affiche les résultats d’un portefeuille à partir des poids donnés.

    Paramètres :
    - weights : np.ndarray ou tensor (n, nb_assets), les poids du portefeuille jour par jour
    - returns : np.ndarray ou tensor (n, nb_assets), les rendements des actifs

    Affiche :
    0) Graphique des poids dans le temps
    1) Rendements journaliers du portefeuille
    2) Performance cumulée
    """

    # Convertir si besoin en numpy
    if not isinstance(weights, np.ndarray):
        weights = weights.detach().cpu().numpy()
    if not isinstance(returns, np.ndarray):
        returns = returns.detach().cpu().numpy()

    # A decommenter si les returns sont pas issu de simulate_rets_portfolio

    # Calcul des rendements du portefeuille avec convention B (weights t-1 sur returns t)
    #n_obs = returns.shape[0]
    #portfolio_returns = np.zeros(n_obs)
    #portfolio_returns[0] = returns[0] @ weights[0]

    #for t in range(1, n_obs):
    #   portfolio_returns[t] = returns[t] @ weights[t - 1]
    # returns = portfolio_returns

    # === Affichage ===
    plt.figure(figsize=figsize)

    # 0. Poids dans le temps
    plt.subplot(131)
    plt.stackplot(np.arange(weights.shape[0]), weights.T)
    plt.title("Poids des actifs (stacked)")
    plt.xlabel("Jour")
    plt.ylabel("Poids")

    # 1. Rendements journaliers
    plt.subplot(132)
    plt.plot(returns)
    plt.title("Rendements journaliers du portefeuille")
    plt.xlabel("Jour")
    plt.ylabel("Rendement")

    # 2. Performance cumulée
    plt.subplot(133)
    plt.plot(np.cumprod(returns + 1))
    plt.title("Performance cumulée du portefeuille")
    plt.xlabel("Jour")
    plt.ylabel("Cumul (base 1.0)")

    plt.tight_layout()
    plt.show()

    # Sauvegarde
    if savefig:
        plt.savefig(save_path)
        print(f"Figure enregistrée sous : {save_path}")

    # Sharpe ratio annualisé
    compute_sharpe(returns)



def compute_sharpe(returns):
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    print("Sharpe ratio :", round(sharpe, 4))

    return sharpe
