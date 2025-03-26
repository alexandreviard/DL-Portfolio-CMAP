import matplotlib.pyplot as plt
import numpy as np
from dataset import FinancialDataset, DataHandler

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


def plot_price_simulations(raw_data :FinancialDataset, synthetic_dataloader : DataHandler):
    """
    Pour le moment les indices sont dans le dataloader à revoir après et les mettre dans le FinancialDataset directement
    """
    synthetic_data = raw_data.dataset
    # Convert synthetic data to NumPy (Shape: n_simul, n_dates, n_assets)
    synthetic_returns = synthetic_data.numpy()  # Shape: (n_simul, n_dates, n_assets)

    # Get the number of simulations, dates, and assets
    n_simul, n_dates, n_assets = synthetic_returns.shape

    # Get the date index from DataHandler
    date_range = synthetic_dataloader.date_range

    # Convert Returns to Prices for All Simulations (Assume Initial Price = 100)
    initial_price = 100
    synthetic_prices = initial_price * (1 + synthetic_returns).cumprod(axis=1)  # Cumprod over time (axis=1)

    # Plot the price series for each simulation and asset
    plt.figure(figsize=(12, 6))

    # Use different colors for each simulation and asset
    colors = plt.get_cmap("tab10", n_simul * n_assets)  # Generate colors

    for sim in range(n_simul):
        for asset_idx, ticker in enumerate(raw_data.tickers):
            plt.plot(date_range, synthetic_prices[sim, :, asset_idx], 
                    label=f"Sim {sim+1} - {ticker}", color=colors(sim * n_assets + asset_idx))

    # Formatting
    plt.xlabel("Date")
    plt.ylabel("Synthetic Price")
    plt.title("Synthetic Stock Price Simulation - Multiple Simulations")
    plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))  # Legend outside the plot
    plt.grid()
    plt.show()
