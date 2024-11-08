import numpy as np
import pandas as pd
from typing import Callable


class DerivativeBTM:
    """
    This class find the price of a derivative by using the Binomial Tree Model.

    In this implementation, the following assumptions are made:
    - r = 0: the time horizon is considered to be short enough that no interest would be paid over this period.
    - q = 1 / 2.
    - S_delta only tested for +/- 20.

    For details of the mathematics refer to doc/BinomialTreeModel.md.
    """
    def __init__(self,
                 S_0: int,
                 delta_S: int,
                 T: int,
                 payoff_func: Callable[[np.ndarray], np.ndarray],
                 payoff_func_desc: str | bool = None,
                 verbose: bool = True) -> None:
        """
        Constructor.

        Args:
            S_0: Starting price of the underlying stock.
            delta_S: Size of movement of the underlying (assumed to be the same for up and down).
            T: Number of time discrete time periods to maturity.
            payoff_func: Function which calculates the payoff at maturity of the derivative to be modelled.
            payoff_func_desc: Description of the payoff_func, which defaults to None.
            verbose: Flag which controls verbosity, which defaults to True.
        """
        # Save inputs attributes
        self.S_0 = S_0
        self.delta_S = delta_S
        self.T = T
        self.payoff_func = payoff_func
        self.payoff_func_desc = payoff_func_desc
        self.verbose = verbose

        # Derived attributes
        self.N_rows = 2 ** (self.T - 1) - 1  # Rows required for binomial tree model matrix
        self.initial_inds = np.array([int(self.N_rows // 2), 0])  # The index pair for point at t = 0

        # Flags
        self.simulated = False

        # Placeholder attributes
        self.stock_tree = None
        self.deriv_tree = None
        self.hedge_tree = None
        self.bond_tree = None

    def _verbose_header(self) -> None:
        """
        Prints a header to the console if then verbose flag is active.
        """
        # Verbose prints
        if self.verbose:
            print()
            print(f'{"Binomial Tree Model Instance":=^80}')
            if self.payoff_func_desc is not None:
                print(f'{self.payoff_func_desc}')
                print(f'This is modelled with initial stock price of S_0 = {self.S_0} and maturity '
                      f'T = {self.T} timesteps')

    def _generate_stock_tree(self) -> None:
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
            inds_prices = np.where(~np.isnan(stock_tree[:, t - 1]))[0]

            # Generate an up and down price for each non-zero price of the previous timestep
            for ind_price in inds_prices:
                # Up move
                stock_tree[ind_price - 1, t] = stock_tree[ind_price, t - 1] + 20

                # Down move
                stock_tree[ind_price + 1, t] = stock_tree[ind_price, t - 1] - 20

        # Save as attributes
        self.stock_tree = stock_tree

        # Verbose prints
        if self.verbose:
            print()
            print('The stock tree:')
            print(self.stock_tree)

    def _calculate_derivative_tree(self) -> None:
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
        t_backprop_steps = np.arange(1, self.T)[::-1]
        for t in t_backprop_steps:
            # Indices for which there are prices in the previous timesteps.
            inds_prices = np.where(~np.isnan(deriv_tree[:, t]))[0]

            # Generate the backpropagation by looking at paris of indices
            for ind_price_top, ind_price_bot in zip(inds_prices[1:], inds_prices[:-1]):
                ind_price_backprop = int((ind_price_bot + ind_price_top)/2)
                # Get value
                deriv_tree[ind_price_backprop, t - 1] = 0.5*(deriv_tree[ind_price_bot, t]
                                                             + deriv_tree[ind_price_top, t])

        # Save as attributes
        self.deriv_tree = deriv_tree

        # Verbose prints
        if self.verbose:
            print('The derivative tree:')
            print(self.deriv_tree)

    def _calculate_hedges(self) -> None:
        """
        Calculates the hedge tree using the stock and derivative tree in matrix form where empty cells are
        filled with nans. The start, t = 0, is shown as the leftmost column of the matrix. Whereas, the end,
        t = T - 1, is shown as the rightmost column of the matrix.
        """
        # Initialise the hedge tree
        hedge_tree = np.full_like(self.stock_tree, np.nan)

        # Compute backpropagation
        t_backprop_steps = np.arange(1, self.T)[::-1]
        for t in t_backprop_steps:
            # Indices for which there are prices in the previous timesteps.
            inds_prices = np.where(~np.isnan(self.deriv_tree[:, t]))[0]

            # Generate the backpropagation by looking at paris of indices
            for ind_price_top, ind_price_bot in zip(inds_prices[1:], inds_prices[:-1]):
                ind_price_backprop = int((ind_price_bot + ind_price_top)/2)
                # Get value
                deriv_diff = self.deriv_tree[ind_price_bot, t] - self.deriv_tree[ind_price_top, t]
                stock_diff = self.stock_tree[ind_price_bot, t] - self.stock_tree[ind_price_top, t]
                hedge_tree[ind_price_backprop, t - 1] = deriv_diff / stock_diff

        # Save as attributes
        self.hedge_tree = hedge_tree

        # Verbose prints
        if self.verbose:
            print('The hedge tree:')
            print(self.hedge_tree)

    def _calculate_borrowing(self) -> None:
        """
        Calculates the borrowing tree using the stock and derivative tree in matrix form where empty cells are
        filled with nans. The start, t = 0, is shown as the leftmost column of the matrix. Whereas, the end,
        t = T - 1, is shown as the rightmost column of the matrix.
        """
        # Calculate borrowing tree
        self.bond_tree = self.deriv_tree - self.hedge_tree*self.stock_tree

        # Verbose prints
        if self.verbose:
            print('The bond holding tree:')
            print(self.bond_tree)

    def simulate(self) -> None:
        """
        Performs the generation of the four trees required: stock, derivative, hedge and borrowing.
        """
        # Prevents resimulation
        if not self.simulated:

            self._verbose_header()

            # Calculation of all trees
            self._generate_stock_tree()
            self._calculate_derivative_tree()
            self._calculate_hedges()
            self._calculate_borrowing()

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
                                                      self.bond_tree[inds_path[:-1, 0], inds_path[:-1, 1]]))

        # Verbose prints
        if self.verbose:
            print()
            print(f'The filtration table for the sequence {seq}:')
            print(filtration_table.to_string())

        if return_flag:
            return filtration_table

    @staticmethod
    def EUR_call_optn_strike100_payoff(S_T: np.ndarray) -> np.ndarray:
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
        S_limit, payoff_val = 100, 100
        payoff = np.where(~np.isnan(S_T), np.where(S_T < S_limit, 0, payoff_val), np.nan)
        return payoff
