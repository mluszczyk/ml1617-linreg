from unittest import TestCase

import numpy

from linreg import rmse_partial_derivative, logistic


class TestRMSE(TestCase):
    def test_derivative(self):
        x = numpy.asarray([[1., 2., 2.], [1., 0., 2.], [1., 2., 0.], [1., 0., 0.]])
        y = numpy.asarray([1., -1., 3., 1.])
        w = numpy.asarray([0., 0., 0.])

        e = rmse_partial_derivative(0., y, w, x, 0)
        self.assertAlmostEqual(e, -2, places=2)


class TestLogistic(TestCase):
    def test_zero(self):
        self.assertAlmostEqual(logistic(0.), 0.5, places=5)

    def test_minus_inf(self):
        self.assertAlmostEqual(logistic(-100), 0., places=5)
