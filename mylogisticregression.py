import numpy
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from linreg import partial_derivative_logistic, predict_logistic, predict_logistic_bool, adjust
from mylinearregression import MyLinearRegression


class MyLogisticRegression(BaseEstimator):
    def __init__(self, *args, **kwargs):
        self.linear_regression = MyLinearRegression(*args, **kwargs)
        assert self.linear_regression.partial_derivative is not None
        self.linear_regression.partial_derivative = partial_derivative_logistic
        assert self.linear_regression.predict_func is not None
        self.linear_regression.predict_func = predict_logistic_bool

    def fit(self, X, y):
        from matplotlib import pyplot
        pyplot.hist(y)

        self.linear_regression.fit(X, y)
        return self

    def predict(self, X):
        return self.linear_regression.predict(X)

    def predict_log(self, X):
        X = check_array(X)
        if self.linear_regression.standard_scaler is not None:
            X = self.linear_regression.standard_scaler.transform(X)

        return predict_logistic(self.linear_regression.w, adjust(X))
