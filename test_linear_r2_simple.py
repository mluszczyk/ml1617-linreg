from unittest import TestCase

import numpy

from mylinearregression import MyLinearRegression


class LinearRegressionR2Simple(TestCase):
    def test_call(self):
        X = numpy.asarray([[2., 2.], [0., 2.], [2., 0.], [0., 0.]])
        y = numpy.asarray([1., -1., 3., 1.])

        estimator = MyLinearRegression()
        estimator.fit(X, y)

        self.assertAlmostEqual(estimator.w[0], 1, places=1)
        self.assertAlmostEqual(estimator.w[1], 1, places=1)
        self.assertAlmostEqual(estimator.w[2], -1, places=1)

        y_pred = estimator.predict(X)
        self.assertAlmostEqual(y_pred[0], 1., places=1)
