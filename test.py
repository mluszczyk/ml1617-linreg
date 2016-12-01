from unittest import TestCase

import numpy
from numpy.random import seed
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.special._ufuncs import expit
from sklearn.metrics import accuracy_score

from linreg import rmse_gradient
from mylinearregression import MyLinearRegression
from mylogisticregression import MyLogisticRegression


class TestRMSE(TestCase):
    def test_derivative(self):
        x = numpy.asarray([[1., 2., 2.], [1., 0., 2.], [1., 2., 0.], [1., 0., 0.]])
        y = numpy.asarray([1., -1., 3., 1.])
        w = numpy.asarray([0., 0., 0.])

        e = rmse_gradient(w, x, y)[0]
        self.assertAlmostEqual(e, -1, places=2)


class TestLogisticRegression(TestCase):
    def test_zero(self):
        self.assertAlmostEqual(expit(0.), 0.5, places=5)

    def test_minus_inf(self):
        self.assertAlmostEqual(expit(-100), 0., places=5)

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

        estimator = MyLogisticRegression()
        estimator.fit(X, y)
        y_pred = estimator.predict(X)

        accuracy = accuracy_score(y, y_pred)

        self.assertGreaterEqual(accuracy, 0.9)

        proba_pred = 1 - estimator.predict_proba(X)[:, 0]
        assert_array_equal(numpy.round_(proba_pred), y_pred)

        proba_pred = estimator.predict_proba(X)[:, 1]
        assert_array_equal(numpy.round_(proba_pred), y_pred)

    def test_two_y(self):
        estimator = MyLinearRegression()
        with self.assertRaises(ValueError):
            estimator.fit(numpy.asarray([[1]]), numpy.asarray([[1, 1]]))


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

        estimator = MyLinearRegression(learning_rate=0.2)
        estimator.fit(X, y)

        self.assertAlmostEqual(estimator.w[0], 1, places=1)
        self.assertAlmostEqual(estimator.w[1], 1, places=1)
        self.assertAlmostEqual(estimator.w[2], -1, places=1)

        y_pred = estimator.predict(X)
        self.assertAlmostEqual(y_pred[0], 1., places=1)

    def test_standardization_is_used(self):
        X = numpy.asarray([[2., 3.], [0., 1.], [2., 0.], [0., 0.]])
        y = numpy.asarray([1., 5., 3., 1.])

        estimator = MyLinearRegression(standardize=True)
        estimator.fit(X, y)

        scaler = estimator.standard_scaler
        self.assertIsNotNone(scaler)

        assert_array_equal(scaler.mean_, [1., 1.])

        y_pred = estimator.predict(X)

        estimator.standard_scaler = None
        y_pred_unscaled = estimator.predict(X)
        self.assertNotEqual(list(y_pred), list(y_pred_unscaled))

    def test_standardize_y(self):
        X = numpy.asarray([[1.63295], [-1.63295], [0.]])
        y = numpy.asarray([2., -2., 0.])

        estimator = MyLinearRegression(standardize=True)
        estimator.fit(X, y)

        assert_array_almost_equal(estimator.w, [0., 1.], decimal=3)
        assert_array_almost_equal(estimator.predict(X), [2., -2., 0.], decimal=3)
