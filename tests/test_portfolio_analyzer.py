import unittest
import os
import numpy as np
import pandas as pd

from portfolio_analyzer import PortfolioAnalyzer, DEPOSIT_DESCRIPTION_LIST


class TestPortfolioAnalyzer(unittest.TestCase):

    def setUp(self) -> None:
        self.portfolio_value_df = \
            pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), 'fixtures', 'chart.csv')))
        self.transaction_df = \
            pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), 'fixtures', 'transactions_2022.csv')))

    def test_calculate_sliding_correlation(self):
        x = [1, 5, -1, 10, 1, 0]
        y = [0, 1, 2, 1, 0, -1]
        window_length = 4
        sliding_corr = np.zeros(len(x) - window_length + 1)
        for i in range(len(x) - window_length + 1):
            meanx = np.mean(x[i:window_length - 1 + i])
            meany = np.mean(y[i:window_length - 1 + i])
            sliding_corr[i] = np.sum((x[i:window_length - 1 + i] - meanx) * (y[i:window_length - 1 + i]))
            sliding_corr[i] /= np.sqrt(np.sum((x[i:window_length - 1 + i] - meanx) ** 2) *
                                       np.sum((y[i:window_length - 1 + i] - meany) ** 2))
        np.testing.assert_array_almost_equal(sliding_corr, PortfolioAnalyzer.calculate_sliding_correlation(x, y, 4))

    def test_calculate_daily_returns(self):
        cum_daily_return_array = np.array([0,
                                           0.1371151796,
                                           -1.364033684,
                                           -2.988407239,
                                           -3.0519008,
                                           -2.559795194,
                                           -3.266013617,
                                           -1.847556881,
                                           -0.7723761926,
                                           -1.169627869,
                                           -3.731395797,
                                           -4.571963505,
                                           -5.440963973,
                                           -5.124304499,
                                           -3.693109306])

        test_portfolio_value_df = PortfolioAnalyzer.calculate_daily_returns(self.transaction_df,
                                                                            self.portfolio_value_df)
        test_cum_daily_return_array = test_portfolio_value_df["cumulative_return_pct"].values
        np.testing.assert_array_almost_equal(cum_daily_return_array, test_cum_daily_return_array, 2)


if __name__ == '__main__':
    unittest.main()
