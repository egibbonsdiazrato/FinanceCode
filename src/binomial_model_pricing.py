import numpy as np

from modules.DerivativeBinomialTreeModel import Market, Stock, DerivativeBTM


def binary_option_payoff(S_T: np.ndarray) -> np.ndarray:
    """
    Produces payoffs for a custom function which only pays the payoff_val if stock price at maturity is greater
    than or equal to S_limit.

    Args:
        S_T: the last column of the matrix which holds the values of the stock at maturity.

    Returns:
        payoff: the payoff of S_T for the option.
    """
    S_limit, payoff_val = 100, 100  # Parameters
    payoff = np.where(~np.isnan(S_T), np.where(S_T < S_limit, 0, payoff_val), np.nan)
    return payoff


if __name__ == '__main__':
    # Set print options to print the entire matrix
    np.set_printoptions(threshold=10**5, linewidth=10**5)

    # Initial set up
    market_1 = Market(r=0, T=3)
    market_2 = Market(r=0.01/100, T=9)

    stock_A = Stock(S_0=100, DeltaS=20, DeltaS_type='abs')
    stock_B = Stock(S_0=100, DeltaS=1.1, DeltaS_type='rel')

    # Option simulations
    option1 = DerivativeBTM(payoff_func=DerivativeBTM.EUR_call_option_strike100_payoff,
                            payoff_func_desc='This derivative is a EUR call option with strike 100.')
    option1.simulate_price_and_replication(stock=stock_A, market=market_1, verbose=True)
    option1.generate_filtration_table(['down', 'up', 'down'], market_1.T)

    option2 = DerivativeBTM(payoff_func=DerivativeBTM.EUR_call_option_strike100_payoff,
                            payoff_func_desc='This derivative is a EUR call option with strike 100.')
    option2.simulate_price_and_replication(stock=stock_B, market=market_2, verbose=True)
    option2.generate_filtration_table(['down', 'up', 'down', 'up', 'up', 'down', 'up', 'down', 'up'], market_2.T)

    option3 = DerivativeBTM(payoff_func=binary_option_payoff,
                            payoff_func_desc='This derivative pays 100 if the stock price at maturity is greater than'
                                             '\nor equal to 100.'
                            )
    option3.simulate_price_and_replication(stock=stock_A, market=market_1, verbose=True)
    option3.generate_filtration_table(['down', 'up', 'down'], market_1.T)
