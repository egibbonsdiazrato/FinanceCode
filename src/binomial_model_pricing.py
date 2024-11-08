from modules.DerivativeBinomialTreeModel import DerivativeBTM


if __name__ == '__main__':
    # Parameters
    (S_0, delta_S, T, payoff_func) = (100, 20, 4, DerivativeBTM.EUR_call_optn_strike100_payoff)

    option = DerivativeBTM(S_0,
                           delta_S,
                           T,
                           payoff_func,
                           payoff_func_desc='This derivative is a EUR call option with strike 100.')
    option.simulate()
    option.generate_filtration_table(['down', 'up', 'down'])

    # Parameters
    (S_0, delta_S, T, payoff_func) = (100, 20, 4, DerivativeBTM.custom_option_payoff)

    option = DerivativeBTM(S_0,
                           delta_S,
                           T,
                           payoff_func,
                           payoff_func_desc='This derivative pays 100 if the stock price at maturity is greater than \n'
                                            'the starting stock price.')
    option.simulate()
    option.generate_filtration_table(['up', 'up', 'up'])
