import numpy
from numpy import concatenate
from scipy.special import expit
from scipy.special._ufuncs import expit

from mybaseregression import MyBaseRegression


class MyLogisticRegression(MyBaseRegression):
    def __init__(self,
                 batch_size=None, n_epochs=100, shuffle=False,
                 holdout_size=0., l2=0., learning_rate=.1, decay=1.0,
                 standardize=False):
        super().__init__(logistic_gradient,
                         batch_size, n_epochs, shuffle,
                         holdout_size, l2, learning_rate, decay,
                         standardize)

    def predict(self, X):
        return self.predict_wrapper(X, lambda w, x: x @ w > 0.)

    def predict_proba(self, X):
        pred = self.predict_wrapper(X, lambda w, x: expit(x @ w))
        col2 = pred.reshape(-1, 1)
        col1 = 1. - col2
        return concatenate((col1, col2), axis=1)


def logistic_gradient(w, x, y):
    return numpy.mean([
        (expit(numpy.inner(w, xk)) - yk) * xk for xk, yk in zip(x, y)
    ], axis=0)