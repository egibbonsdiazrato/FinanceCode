import numpy as np

from modules.DerivativeBinomialTreeModel import DerivativeBTM


def custom_option_payoff(S_T: np.ndarray) -> np.ndarray:
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

    # Parameters
    S_0 = 100
    DeltaS = 20
    DeltaS_type = 'abs'
    r = 0 / 100
    T = 4
    payoff_func = DerivativeBTM.EUR_call_option_strike100_payoff

    option1 = DerivativeBTM(S_0,
                            DeltaS,
                            DeltaS_type,
                            r,
                            T,
                            payoff_func,
                            payoff_func_desc='This derivative is a EUR call option with strike 100.')
    option1.simulate()
    option1.generate_filtration_table(['down', 'up', 'down'])

    option1_alt = DerivativeBTM(S_0,
                                DeltaS,
                                DeltaS_type,
                                0.01 / 100,
                                T,
                                payoff_func,
                                payoff_func_desc='This derivative is a EUR call option with strike 100.')
    option1_alt.simulate()
    option1_alt.generate_filtration_table(['down', 'up', 'down'])

    # Parameters
    S_0 = 100
    DeltaS = 1.1
    DeltaS_type = 'rel'
    r = 0.01 / 100
    T = 10

    option2 = DerivativeBTM(S_0,
                            DeltaS,
                            DeltaS_type,
                            r,
                            T,
                            payoff_func,
                            payoff_func_desc='This derivative is a EUR call option with strike 100.')
    option2.simulate()
    option2.generate_filtration_table(['down', 'up', 'down', 'up', 'up',
                                       'down', 'up', 'down', 'up'])

    # Parameters
    S_0 = 100
    DeltaS = 20
    DeltaS_type = 'abs'
    r = 0 / 100
    T = 4
    payoff_func = custom_option_payoff

    option3 = DerivativeBTM(S_0,
                            DeltaS,
                            DeltaS_type,
                            r,
                            T,
                            payoff_func,
                            payoff_func_desc='This derivative pays 100 if the stock price at maturity is greater than'
                                             '\nor equal to 100.')
    option3.simulate()
    option3.generate_filtration_table(['up', 'up', 'up'])
