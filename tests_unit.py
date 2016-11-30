from unittest import TestCase

import numpy

from linreg import rmse, rmse_partial_derivative


class TestRMSE(TestCase):
    def test_value(self):
        x = numpy.asarray([[1., 2., 2.], [1., 0., 2.], [1., 2., 0.], [1., 0., 0.]])
        y = numpy.asarray([1., -1., 3., 1.])
        w = numpy.asarray([0., 0., 0.])
        e = rmse(w, x, y)
        self.assertAlmostEqual(e, 3., places=2)

    def test_derivative(self):
        x = numpy.asarray([[1., 2., 2.], [1., 0., 2.], [1., 2., 0.], [1., 0., 0.]])
        y = numpy.asarray([1., -1., 3., 1.])
        w = numpy.asarray([0., 0., 0.])

        e = rmse_partial_derivative(0., y, w, x, 0)
        print(e)
