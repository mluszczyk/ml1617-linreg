from unittest import TestCase

import numpy

from linreg import rmse_partial_derivative, logistic
from mylogisticregression import MyLogisticRegression


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

    def test_get_params(self):
        estimator = MyLogisticRegression(l2=0.003)
        params = {'batch_size': None,
                  'decay': 1.0,
                  'holdout_size': 0.0,
                  'l2': 0.003,
                  'learning_rate': 0.1,
                  'n_epochs': 100,
                  'shuffle': False,
                  'standardize': False}
        self.assertEqual(estimator.get_params(), params)

    def test_set_params(self):
        estimator = MyLogisticRegression()
        estimator.set_params(l2=0.04)
        self.assertEqual(estimator.l2, 0.04)
