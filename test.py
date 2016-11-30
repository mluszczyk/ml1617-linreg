from unittest import TestCase

import numpy
from numpy.random import seed
from numpy.testing import assert_array_equal
from sklearn.metrics import accuracy_score

from linreg import rmse_partial_derivative, logistic
from mylinearregression import MyLinearRegression
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

    def test_example(self):
        numpy.random.seed(2)

        a = 0.3
        b = -0.2
        c = 0.001

        def lin_rule(x, noise=0.):
            return a * x[0] + b * x[1] + c + noise < 0.

        n = 100
        range_points = 1

        sigma = 0.05

        X = range_points * 2 * (numpy.random.rand(n, 2) - 0.5)
        y = [lin_rule(x, sigma * numpy.random.normal()) for x in X]

        estimator = MyLogisticRegression(learning_rate=0.05)
        estimator.fit(X, y)
        y_pred = estimator.predict(X)

        accuracy = accuracy_score(y, y_pred)

        print(accuracy)
        self.assertGreaterEqual(accuracy, 0.9)


class TestLinear(TestCase):
    def test_zero_epochs(self):
        estimator = MyLinearRegression(n_epochs=0)
        X = numpy.asarray([[1., 2.], [3., 4.]])
        y = numpy.asarray([1., 3.])
        estimator.fit(X, y)
        assert_array_equal(estimator.w, [0., 0., 0.])

    def test_holdout(self):
        estimator = MyLinearRegression(holdout_size=0.5)
        v = numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8])
        estimator.fit(v.reshape(8, 1), v)
        assert_array_equal(estimator.validation[1], [1, 2, 3, 4])

    def test_holdout_shuffle(self):
        seed(34)
        estimator = MyLinearRegression(holdout_size=0.5, shuffle=True)
        v = numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8])
        estimator.fit(v.reshape(8, 1), v)
        self.assertTrue((estimator.validation[1] != numpy.asarray([1, 2, 3, 4])).all())

    def test_example_two_features(self):
        X = numpy.asarray([[2., 2.], [0., 2.], [2., 0.], [0., 0.]])
        y = numpy.asarray([1., -1., 3., 1.])

        estimator = MyLinearRegression()
        estimator.fit(X, y)

        self.assertAlmostEqual(estimator.w[0], 1, places=1)
        self.assertAlmostEqual(estimator.w[1], 1, places=1)
        self.assertAlmostEqual(estimator.w[2], -1, places=1)

        y_pred = estimator.predict(X)
        self.assertAlmostEqual(y_pred[0], 1., places=1)