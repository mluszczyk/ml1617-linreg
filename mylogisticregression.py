import numpy
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from linreg import partial_derivative_logistic, predict_logistic
from mylinearregression import MyLinearRegression


class MyLogisticRegression(BaseEstimator):
    def __init__(self, *args, **kwargs):
        self.linear_regression = MyLinearRegression(*args, **kwargs)
        # assert self.linear_regression.partial_derivative is not None
        # self.linear_regression.partial_derivative = partial_derivative_logistic
        # assert self.linear_regression.predict_func is not None
        # self.linear_regression.predict_func = predict_logistic

    @staticmethod
    def bools_to_floats(y):
        return numpy.asfarray(y)

    @staticmethod
    def floats_to_bools(y):
        return y > 0.5

    def fit(self, X, y):
        y = self.bools_to_floats(y)

        from matplotlib import pyplot
        pyplot.hist(y)

        self.linear_regression.fit(X, y)

        # for compatibility with sklearn LogisticRegression
        # self.coef_ = numpy.asarray([self.w[1:]])
        # self.intercept_ = numpy.asarray([self.w[0]])

        return self

    def predict(self, X):
        y = self.linear_regression.predict(X)

        self.y_ = y

        return self.floats_to_bools(y)
