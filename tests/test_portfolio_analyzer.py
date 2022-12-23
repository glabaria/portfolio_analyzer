import unittest
import numpy as np

from portfolio_analyzer import PortfolioAnalyzer


class TestPortfolioAnalyzer(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
