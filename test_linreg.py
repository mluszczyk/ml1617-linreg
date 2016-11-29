from unittest import TestCase

from linreg import rmse_partial_derivative


class Test(TestCase):
    def test_partial_derivative_two_points(self):
        x = [(1, 1), (1, 2)]
        y = [2, 2]
        d0 = rmse_partial_derivative(y, (2, 0), x, 0)
        self.assertEqual(d0, 0)
        d1 = rmse_partial_derivative(y, (2, 0), x, 1)
        self.assertEqual(d1, 0)