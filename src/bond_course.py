from colorama import Fore, Style
import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf
import pandas as pd

if __name__ == '__main__':
    # Standalone questions
    print(Fore.BLUE
          + 'You want to start investing and find a bond paying a 3% annual rate compounded monthly. If you invest '
            'USD 100 per month, how much will you have after five years?'
          + Style.RESET_ALL)
    total_savings = npf.fv(rate=0.03/12, nper=12*5, pmt=-100, pv=0)
    print(Fore.GREEN
          + f'The future value of this investment is USD {total_savings:.2f}.\n'
          + Style.RESET_ALL)

    print(Fore.BLUE
          + 'You are saving for a deposit on a house.You need to save up USD 15,000 and have saved USD 5,000 so far.'
            'If you save USD 250 per month, how many months until you reach your goal?'
          + Style.RESET_ALL)
    time_to_goal = npf.nper(rate=0.05 / 12, pmt=-250, pv=-5000, fv=15000)
    print(Fore.GREEN
          + f'The number of months is {time_to_goal:.2f} or {time_to_goal//12} years and {np.ceil(time_to_goal % 12)} '
            f'months.\n'
          + Style.RESET_ALL)

    print(Fore.BLUE
          + 'You are 22 years old and want to retire at 65 with USD 1 million in the bank. If you can grow your '
            'money at 5% per year, how much should you invest per month?\n'
          + Style.RESET_ALL)
    monthly_contributions = npf.pmt(rate=0.05/12, nper=(65-22)*12, pv=0, fv=1000000)
    print(Fore.GREEN
          + f'The monthly contributions should be USD {-monthly_contributions:.2f}.\n'
          + Style.RESET_ALL)

    print(Fore.BLUE
          + 'Imagine you want to have saved USD 11,000 after 7 years and consider three different investments, each '
            'paying 3%, 4%, and 5% annual interest compounded monthly. If you contribute USD 100 per month, '
            'what initial lump sum should you also invest to achieve this goal?'
          + Style.RESET_ALL)
    for ann_rate in [0.03, 0.04, 0.05]:
        initial_lump_sum = npf.pv(rate=ann_rate / 12, nper=12 * 7, pmt=-100, fv=11000)
        print(Fore.GREEN
              + f'For an annualised rate of {ann_rate*100:.2f}%, the initial contribution should be '
                f'USD {-initial_lump_sum}.'
              + Style.RESET_ALL)
    print()

    for yield_val, maturity in zip([0.05, 0.05, 0.07], [3, 10, 3]):
        ZCB_price = 100/((1 + yield_val)**maturity)
        print(Fore.BLUE
              + f'The price of a ZC bond with face value USD 100, yield {yield_val*100:.2f}% and maturity '
                f'{maturity}Y is '
              + Fore.GREEN
              + f'USD {ZCB_price:.2f}.'
              + Style.RESET_ALL)

