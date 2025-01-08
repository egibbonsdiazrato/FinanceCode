import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(4235243)  # Fix seed

if __name__ == '__main__':
    # TWO-ASSET EXAMPLE

    # Equally weighted example
    # Parameters
    N_days = 500
    mu_a, sigma_a = 0.045, 0.10
    mu_b, sigma_b = 0.045, 0.15

    ret_a = np.random.normal(mu_a, sigma_a, N_days)
    ret_b = np.random.normal(mu_b, sigma_b, N_days)

    weights = 0.5*np.ones(2)  # Equally weighted

    # Compute the portfolio expected rate of return and standard deviation
    expected_ret_assets = np.array([np.mean(ret_a), np.mean(ret_b)])
    expected_ret_portfolio = np.transpose(expected_ret_assets) @ weights

    portfolio_cov = np.cov(ret_a, ret_b)
    vol_portfolio = np.sqrt(np.transpose(weights) @ portfolio_cov @ weights)

    # Prints
    print('Covariance matrix: \n', portfolio_cov)
    print('Asset expected returns: \n', expected_ret_assets)
    print(f'Portfolio expected return and volatility: {expected_ret_portfolio} and {vol_portfolio}')

    # Plotting
    fig1 = plt.figure(figsize=(16, 9))

    # Top subplot
    ax1 = fig1.add_subplot(211)

    ax1.plot(ret_a, label='Asset a')
    ax1.plot(ret_b, label='Asset b')

    ax1.set_xlabel('Time (Days)')
    ax1.set_ylabel(r'$r$')

    ax1.grid(True, which='both', linewidth=0.5, alpha=0.75)

    # Bottom subplot
    ax2 = fig1.add_subplot(212)

    ax2.hist(ret_a, edgecolor='k', bins=10, alpha=0.75)
    ax2.hist(ret_b, edgecolor='k', bins=10, zorder=-1)

    ax2.set_xlabel(r'$r$')
    ax2.set_ylabel('Frequency')

    ax2.grid(True, which='both', linewidth=0.5, alpha=0.75)

    plt.show()

    # Exploring different weights
    w_a_arr = np.arange(0, 1.0 + 0.01, 0.01)

    # Placeholders
    expected_ret_portfolio_arr = np.array([])
    vol_portfolio_arr = np.array([])

    for w_a in w_a_arr:
        # Weight definition
        weights = np.array([w_a, 1 - w_a])

        # Compute expected return and volatility of the portfolio
        expected_ret_portfolio = np.transpose(expected_ret_assets) @ weights
        vol_portfolio = np.sqrt(np.transpose(weights) @ portfolio_cov @ weights)

        # Append values
        expected_ret_portfolio_arr = np.append(expected_ret_portfolio_arr, expected_ret_portfolio)
        vol_portfolio_arr = np.append(vol_portfolio_arr, vol_portfolio)

    # Plot
    fig2 = plt.figure(figsize=(16, 9))

    # Left subplot
    ax1 = fig2.add_subplot(121)

    ax1.plot(w_a_arr, expected_ret_portfolio_arr, label='Expected Returns')
    ax1.plot(w_a_arr, vol_portfolio_arr, label='Volatility')

    ax1.set_xlabel(r'$w_a$')
    ax1.legend(loc='upper right')

    # Right subplot
    ax2 = fig2.add_subplot(122)

    ax2.scatter(vol_portfolio_arr, expected_ret_portfolio_arr, s=4)

    ax2.set_xlabel('Portfolio Volatlity')
    ax2.set_ylabel('Portfolio Expected Returns')

    plt.show()

    # TODO: N-asset
    