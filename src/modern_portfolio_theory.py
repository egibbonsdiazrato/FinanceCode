import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    two_asset_example = False
    multi_asset_example = False
    real_example = True

    # TWO-ASSET EXAMPLE
    if two_asset_example:
        # Parameters
        N_days = 500
        mu_a, sigma_a = 0.045, 0.05
        mu_b, sigma_b = 0.045, 0.10

        ret_a = np.random.normal(mu_a, sigma_a, N_days)
        ret_b = np.random.normal(mu_b, sigma_b, N_days)

        weights = 0.5*np.ones(2)  # Equally weighted

        # Compute the portfolio expected rate of return and standard deviation
        expected_ret_assets = np.array([np.mean(ret_a), np.mean(ret_b)])
        expected_ret_portfolio = np.transpose(expected_ret_assets) @ weights

        portfolio_cov = np.cov(ret_a, ret_b)
        vol_portfolio = np.sqrt(np.transpose(weights) @ portfolio_cov @ weights)

        # Prints
        print('TWO-ASSET EXAMPLE')
        print('Covariance matrix: \n', portfolio_cov)
        print('Asset expected returns: \n', expected_ret_assets)
        print(f'Portfolio expected return and volatility: {expected_ret_portfolio} and {vol_portfolio}')

        # Plotting
        fig1 = plt.figure(figsize=(16, 9))

        # Top subplot
        ax1 = fig1.add_subplot(211)

        ax1.plot(100*ret_a, label='Asset a')
        ax1.plot(100*ret_b, label='Asset b')

        ax1.set_xlabel('Time (Days)')
        ax1.set_ylabel(r'$r$ (%)')
        ax1.legend(loc='upper right')

        ax1.grid(True, which='both', linewidth=0.5, alpha=0.75)

        # Bottom subplot
        ax2 = fig1.add_subplot(212)

        ax2.hist(100*ret_a, edgecolor='k', bins=10, alpha=0.75)
        ax2.hist(100*ret_b, edgecolor='k', bins=10, zorder=-1)

        ax2.set_xlabel(r'$r$ (%)')
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

        ax1.grid(True, which='both', linewidth=0.5, alpha=0.75)
        ax1.legend(loc='upper right')

        # Right subplot
        ax2 = fig2.add_subplot(122)

        ax2.scatter(vol_portfolio_arr, 100*expected_ret_portfolio_arr, s=4)

        ax2.set_xlabel('Portfolio Volatility')
        ax2.set_ylabel('Portfolio Expected Returns (%)')

        ax2.grid(True, which='both', linewidth=0.5, alpha=0.75)

        plt.show()

    # MULTI-ASSET EXAMPLE
    if multi_asset_example:
        # Parameters
        mu, sigma, N_assets, N_obs, N_portfolio = 0.05, 0.20, 10, 100, 10000

        # Generate the returns (matrix) of the assets, where each row is an asset and the columns are timesteps (or
        # observations)
        ret = np.random.normal(mu, sigma, (N_assets, N_obs))
        ret_mean = np.mean(ret, axis=1)

        # Storage arrays
        expected_ret_portfolio_arr = np.array([])
        vol_portfolio_arr = np.array([])

        for _ in range(0, N_portfolio):
            # Compute portfolio expected return
            weights_lims = np.array([0.0, 1.0])
            weights = np.random.uniform(low=weights_lims[0], high=weights_lims[1], size=N_assets)
            weights /= np.sum(weights)  # Normalise

            # Compute expected return and std. deviation of the portfolio
            exp_ret_port = np.transpose(ret_mean) @ weights
            port_cov = np.cov(ret)
            vol_port = np.sqrt(np.transpose(weights) @ port_cov @ weights)

            # Append the values to the arrays
            expected_ret_portfolio_arr = np.append(expected_ret_portfolio_arr, exp_ret_port)
            vol_portfolio_arr = np.append(vol_portfolio_arr, vol_port)

        # Plot
        fig3 = plt.figure(figsize=(16, 9))

        ax = fig3.add_subplot(111)

        ax.scatter(vol_portfolio_arr, 100*expected_ret_portfolio_arr, s=12)

        # Find the minimum risk point
        min_risk_index = np.argmin(vol_portfolio_arr)
        ax.scatter(vol_portfolio_arr[min_risk_index], 100*expected_ret_portfolio_arr[min_risk_index],
                   c='green', label='Min. vol', s=20)

        # Find the maximum returns point
        max_ret_index = np.argmax(expected_ret_portfolio_arr)
        ax.scatter(vol_portfolio_arr[max_ret_index], 100*expected_ret_portfolio_arr[max_ret_index],
                   c='red', label='Max. ret', s=20)

        ax.set_xlabel('Portfolio Volatility')
        ax.set_ylabel('Portfolio Returns (%)')

        ax.grid(True, which='both', linewidth=0.5, alpha=0.75)
        ax.legend(loc='upper right')

        plt.show()