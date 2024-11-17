import numpy as np
from unittest import TestCase

from src.modules.DerivativeBinomialTreeModel import DerivativeBTM


class TestDerivativeBTM(TestCase):
    """
    A unit test for a simple EUR call option with strike at 100 which has been solved manually.
    """
    def setUp(self) -> None:
        """
        Set up function to instantiate class to be tested and save expected results as attributes.
        """
        # Initialise and perform simulation
        S_0 = 100
        DeltaS = 20
        DeltaS_type = 'abs'
        r = 0 / 100
        T = 3
        payoff_func = DerivativeBTM.EUR_call_option_strike100_payoff
        self.option = DerivativeBTM(S_0, DeltaS, DeltaS_type, r, T, payoff_func, verbose=False)
        self.option.simulate()

        # Expected results
        self.exp_stock_tree = np.array([[np.nan, np.nan, np.nan, 160.],
                                        [np.nan, np.nan, 140., np.nan],
                                        [np.nan, 120., np.nan, 120.],
                                        [100., np.nan, 100., np.nan],
                                        [np.nan,  80., np.nan,  80.],
                                        [np.nan, np.nan, 60., np.nan],
                                        [np.nan, np.nan, np.nan,  40.]])
        self.exp_deriv_tree = np.array([[np.nan, np.nan, np.nan, 60.],
                                        [np.nan, np.nan, 40., np.nan],
                                        [np.nan, 25., np.nan, 20.],
                                        [15., np.nan, 10., np.nan],
                                        [np.nan, 5., np.nan, 0.],
                                        [np.nan, np.nan, 0., np.nan],
                                        [np.nan, np.nan, np.nan, 0.]])
        self.exp_hedge_tree = np.array([[np.nan, np.nan, np.nan, np.nan],
                                        [np.nan, np.nan, 1., np.nan],
                                        [np.nan, 0.75, np.nan, np.nan],
                                        [0.5, np.nan, 0.5, np.nan],
                                        [np.nan, 0.25, np.nan, np.nan],
                                        [np.nan, np.nan, 0., np.nan],
                                        [np.nan, np.nan, np.nan, np.nan]])
        self.exp_borrow_tree = np.array([[np.nan, np.nan, np.nan, np.nan],
                                         [np.nan, np.nan, -100., np.nan],
                                         [np.nan, -65., np.nan, np.nan],
                                         [-35., np.nan, -40., np.nan],
                                         [np.nan, -15., np.nan, np.nan],
                                         [np.nan, np.nan, 0., np.nan],
                                         [np.nan, np.nan, np.nan, np.nan]])

    def test_stock_tree(self) -> None:
        """
        Test to ensure matching stock trees.
        """
        self.assertTrue(np.allclose(self.option.stock_tree, self.exp_stock_tree,
                                    atol=1e-14, rtol=1e-14, equal_nan=True))

    def test_deriv_tree(self) -> None:
        """
        Test to ensure matching stock trees.
        """
        self.assertTrue(np.allclose(self.option.deriv_tree, self.exp_deriv_tree,
                                    atol=1e-14, rtol=1e-14, equal_nan=True))

    def test_hedge_tree(self) -> None:
        """
        Test to ensure matching stock trees.
        """
        self.assertTrue(np.allclose(self.option.hedge_tree, self.exp_hedge_tree,
                                    atol=1e-14, rtol=1e-14, equal_nan=True))

    def test_borrow_tree(self) -> None:
        """
        Test to ensure matching stock trees.
        """
        self.assertTrue(np.allclose(self.option.borrow_tree, self.exp_borrow_tree,
                                    atol=1e-14, rtol=1e-14, equal_nan=True))
