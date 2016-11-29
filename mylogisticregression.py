import numpy

from linreg import partial_derivative_logistic
from mylinearregression import MyLinearRegression


class MyLogisticRegression(MyLinearRegression):
    def __init__(self, *args, **kwargs):
        super(MyLogisticRegression, self).__init__(*args, **kwargs)
        self.partial_derivative = partial_derivative_logistic

    @staticmethod
    def bools_to_signs(y):
        f = numpy.vectorize(lambda b: 1. if b else -1, otypes=[numpy.float])
        return f(y)

    @staticmethod
    def signs_to_bools(y):
        return y > 0.

    def fit(self, X, y):
        y = self.bools_to_signs(y)

        super().fit(X, y)

        # for compatibility with sklearn LogisticRegression
        self.coef_ = numpy.asarray([self.w[1:]])
        self.intercept_ = numpy.asarray([self.w[0]])

        return self

    def predict(self, X):
        y = super().predict(X)

        return self.signs_to_bools(y)
