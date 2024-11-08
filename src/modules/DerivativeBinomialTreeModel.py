import numpy as np
import pandas as pd
from typing import Callable


class DerivativeBTM:
    """
    This class find the price of a derivative by using the Binomial Tree Model.

    In this implementation, the following assumptions are made:
    - r = 0: the time horizon is considered to be short enough that no interest would be paid over this period.
    - q = 1 / 2.
    - deltat = 1: the timesteps are integer timesteps.

    For details of the mathematics refer to doc/BinomialTreeModel.md.
    """
    def __init__(self,
                 S_0: int | float,
                 DeltaS: int | float,
                 DeltaS_type: str,
                 T: int,
                 payoff_func: Callable[[np.ndarray], np.ndarray],
                 payoff_func_desc: str | bool = None,
                 verbose: bool = True) -> None:
        """
        Constructor.

        Args:
            S_0: Starting price of the underlying stock.
            DeltaS: Absolute or fractional symmetric movement of the underlying. Provide a positive or a value greater
            than 1 for abs and fractional as the down movement will be either - DeltaS or 1/DeltaS.
            DeltaS_type: either abs or frac to specify what DeltaS is.
            T: Number of time discrete time periods to maturity.
            payoff_func: Function which calculates the payoff at maturity of the derivative to be modelled.
            payoff_func_desc: Description of the payoff_func, which defaults to None.
            verbose: Flag which controls verbosity, which defaults to True.
        """
        # Save inputs attributes
        self.S_0 = S_0
        self.DeltaS = DeltaS
        self.T = T
        self.payoff_func = payoff_func
        self.payoff_func_desc = payoff_func_desc
        self.verbose = verbose

        # Derived attributes
        self.N_rows = 2 ** (self.T - 1) - 1  # Rows required for binomial tree model matrix
        self.initial_inds = np.array([int(self.N_rows // 2), 0])  # The index pair for point at t = 0

        # Flags
        # Delta type flag
        self.DeltaS_abs = False
        self.DeltaS_frac = False
        if DeltaS_type == 'abs':
            self.DeltaS_abs = True
        elif DeltaS_type == 'frac':
            self.DeltaS_frac = True
        else:
            raise Exception(f'The DeltaS type provided, {DeltaS_type}, has to be either abs or frac')
        self.simulated = False

        # Placeholder attributes
        self.stock_tree = None
        self.deriv_tree = None
        self.hedge_tree = None
        self.borrow_tree = None

    def _verbose_header(self) -> None:
        """
        Prints a header to the console.
        """
        print()
        print(f'{"Binomial Tree Model Instance":=^80}')
        if self.payoff_func_desc is not None:
            print(f'{self.payoff_func_desc}')
            print(f'This is modelled with initial stock price of S_0 = {self.S_0} and a maturity of '
                  f'T = {self.T} \n timesteps')
    
    def _calc_up_stock_price(self, S_now: np.floating) -> int | float:
        """
        Calculates the stock up price for the timestep after S_now.
        Args:
            S_now: The price of the stock now.

        Returns:
            S_up: The price of the stock if it goes up the following timestep.
        """

        if self.DeltaS_abs:
            S_up = S_now + self.DeltaS
            return S_up

        if self.DeltaS_frac:
            S_up = S_now*self.DeltaS
            return S_up

    def _calc_down_stock_price(self, S_now: np.floating) -> int | float:
        """
        Calculates the stock down price for the timestep after S_now.
        Args:
            S_now: The price of the stock now.

        Returns:
            S_down: The price of the stock if it goes down the following timestep.
        """

        if self.DeltaS_abs:
            S_down = S_now - self.DeltaS
            return S_down

        if self.DeltaS_frac:
            S_down = S_now*(1/self.DeltaS)
            return S_down
        
    def _gen_stock_tree(self) -> None:
        """
        Generates the stock tree in matrix form where empty cells are filled with nans. The start, t = 0, is shown
        as the leftmost column of the matrix. Whereas, the end, t = T - 1, is shown as the rightmost column of the
        matrix.
        """
        # Initialise matrix
        stock_tree = np.full((self.N_rows, self.T), np.nan)

        # Add initial stock price
        stock_tree[self.initial_inds[0], self.initial_inds[1]] = self.S_0

        # Generate the up and down values for each timestep by looking at non-zero values in the previous timestep
        for t in range(1, self.T):
            # Indices for which there are prices in the previous timesteps
            inds = np.where(~np.isnan(stock_tree[:, t - 1]))[0]

            # Generate an up and down price for each non-zero price of the previous timestep
            for ind_now in inds:
                S_now = stock_tree[ind_now, t - 1]
                ind_up, ind_down = ind_now - 1, ind_now + 1  # Note up and down refers to stock price move
                stock_tree[ind_up, t] = self._calc_up_stock_price(S_now)  # Up move
                stock_tree[ind_down, t] = self._calc_down_stock_price(S_now)  # Down move

        # Save as attributes
        self.stock_tree = stock_tree

        # Verbose prints
        if self.verbose:
            print()
            print('The stock tree:')
            print(self.stock_tree)

    def _calc_derivative_tree(self) -> None:
        """
        Calculates the derivative tree in matrix through backpropagation form where empty cells are filled with nans.
        The start, t = 0, is shown as the leftmost column of the matrix. Whereas, the end, t = T - 1, is shown
        as the rightmost column of the matrix.
        """
        # Initialise the matrix and find the last column by applying the payoff func
        deriv_tree = np.full_like(self.stock_tree, np.nan)
        stock_mat = self.stock_tree[:, -1]
        deriv_payoff_mat = self.payoff_func(stock_mat).reshape(-1)

        # Add the maturity row for derivative
        deriv_tree[:, -1] = deriv_payoff_mat

        # Compute backpropagation
        t_backprop_steps = np.arange(1, self.T)[::-1]  # Reversed array
        for t in t_backprop_steps:
            # Indices for which there are prices in the previous timesteps.
            inds = np.where(~np.isnan(deriv_tree[:, t]))[0]

            # Generate the backpropagation by looking at pairs of indices
            for ind_up, ind_down in zip(inds[:-1], inds[1:]):
                # The row below of up ind (equivalent to above the down ind) is that of the previous timestep
                ind_now = ind_up + 1

                # Calculate q probabilities
                q_num = self.stock_tree[ind_now, t - 1] - self.stock_tree[ind_down, t]
                q_denom = self.stock_tree[ind_up, t] - self.stock_tree[ind_down, t]
                q = q_num / q_denom

                # Get derivative up and down values and calculate derivative now value
                deriv_up = deriv_tree[ind_up, t]
                deriv_down = deriv_tree[ind_down, t]
                deriv_now = q*deriv_up + (1 - q)*deriv_down

                deriv_tree[ind_now, t - 1] = deriv_now

        # Save as attributes
        self.deriv_tree = deriv_tree

        # Verbose prints
        if self.verbose:
            print('The derivative tree:')
            print(self.deriv_tree)

    def _calc_hedges(self) -> None:
        """
        Calculates the hedge tree using the stock and derivative tree in matrix form where empty cells are
        filled with nans. The start, t = 0, is shown as the leftmost column of the matrix. Whereas, the end,
        t = T - 1, is shown as the rightmost column of the matrix.
        """
        # Initialise the hedge tree
        hedge_tree = np.full_like(self.stock_tree, np.nan)

        # Compute backpropagation
        t_backprop_steps = np.arange(1, self.T)[::-1]  # Reversed array
        for t in t_backprop_steps:
            # Indices for which there are prices in the previous timesteps.
            inds = np.where(~np.isnan(self.deriv_tree[:, t]))[0]

            # Generate the backpropagation by looking at pairs of indices
            for ind_up, ind_down in zip(inds[:-1], inds[1:]):
                # The row below of up ind (equivalent to above the down ind) is that of the previous timestep
                ind_now = ind_up + 1

                # Get stock and derivative up and down values to calculate hedge now value
                stock_up = self.stock_tree[ind_up, t]
                stock_down = self.stock_tree[ind_down, t]
                deriv_up = self.deriv_tree[ind_up, t]
                deriv_down = self.deriv_tree[ind_down, t]

                hedge_tree[ind_now, t - 1] = (deriv_up - deriv_down) / (stock_up - stock_down)

        # Save as attributes
        self.hedge_tree = hedge_tree

        # Verbose prints
        if self.verbose:
            print('The hedge tree:')
            print(self.hedge_tree)

    def _calc_borrow(self) -> None:
        """
        Calculates the borrowing tree using the stock and derivative tree in matrix form where empty cells are
        filled with nans. The start, t = 0, is shown as the leftmost column of the matrix. Whereas, the end,
        t = T - 1, is shown as the rightmost column of the matrix.
        """
        # Calculate borrowing tree
        self.borrow_tree = self.deriv_tree - self.hedge_tree*self.stock_tree

        # Verbose prints
        if self.verbose:
            print('The bond holding tree:')
            print(self.borrow_tree)

    def simulate(self) -> None:
        """
        Performs the generation of the four trees required: stock, derivative, hedge and borrowing.
        """
        # Ensures simulation only performed once
        if not self.simulated:

            # Verbose prints
            if self.verbose:
                self._verbose_header()

            # Calculation of all trees
            self._gen_stock_tree()
            self._calc_derivative_tree()
            self._calc_hedges()
            self._calc_borrow()

            # Verbose prints
            if self.verbose:
                deriv_PV = self.deriv_tree[int(self.N_rows//2), 0]
                print(f'\n The PV of this derivative is {deriv_PV}')

            # Change flag
            self.simulated = True

    def _seq_to_inds(self, seq: list[str]) -> np.ndarray:
        """
        Translates the up and down sequence into an array which contains all the pairs of indices to follow a
        specific filtration.

        Args:
            seq: list which contains either 'up' or 'down'.

        Returns:
            inds: np.ndarray of integers with the indices.
        """
        row_prefactor = {'up': -1, 'down': 1}

        inds = np.zeros((self.T, 2))
        inds[0, :] = self.initial_inds

        for counter, move in enumerate(seq, 1):
            inds[counter, :] = inds[counter - 1, :] + np.array([row_prefactor[move], 1])

        return inds.astype(int)

    def generate_filtration_table(self, seq: list[str], return_flag: bool = False) -> None | pd.DataFrame:
        """
        Generate the filtration and print it to console with the value of each tree for each timestep in pandas form.

        Args:
            seq: list which contains either 'up' or 'down' to be passed to _seq_to_inds.
            return_flag: flag to control whether to return values or not.

        Returns:
            filtration_table: optional return if the return_flag is True.
        """

        # Raise error if tree not generated
        if not self.simulated:
            raise Exception('The trees have not been generated.')

        # Raise error if sequence is not the correct length
        if len(seq) != self.T - 1:
            raise Exception(f'The sequence is {len(seq)=} and not {self.T}')

        # Get the indices for the filtration
        inds_path = self._seq_to_inds(seq)

        # Produce table
        filtration_table = pd.DataFrame()
        filtration_table['t_i'] = np.arange(0, self.T)
        filtration_table.set_index(keys=['t_i'], inplace=True)  # Make time the index
        filtration_table['Prev. Jump'] = ['-'] + seq
        filtration_table['S_i'] = self.stock_tree[inds_path[:, 0], inds_path[:, 1]]
        filtration_table['V_i'] = self.deriv_tree[inds_path[:, 0], inds_path[:, 1]]
        filtration_table[r'\phi_i'] = np.concatenate((np.array([np.nan]),
                                                      self.hedge_tree[inds_path[:-1, 0], inds_path[:-1, 1]]))
        filtration_table[r'\psi_i'] = np.concatenate((np.array([np.nan]),
                                                      self.borrow_tree[inds_path[:-1, 0], inds_path[:-1, 1]]))

        # Verbose prints
        if self.verbose:
            print()
            print(f'The filtration table for the sequence {seq}:')
            print(filtration_table.to_string())

        if return_flag:
            return filtration_table

    @staticmethod
    def EUR_call_option_strike100_payoff(S_T: np.ndarray) -> np.ndarray:
        """
        Produces in-built European call option with strike 100 payoffs for stock prices at maturity.

        Args:
            S_T: the last column of the matrix which holds the values of the stock at maturity.

        Returns:
            payoff: the payoff of S_T for the option.
        """
        k = 100  # Strike value
        payoff = np.where(~np.isnan(S_T), np.where(S_T < k, 0, S_T - k), np.nan)
        return payoff

    @staticmethod
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
