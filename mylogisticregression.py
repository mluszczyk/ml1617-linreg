from numpy import concatenate

from linreg import predict_logistic, predict_logistic_bool, logistic_gradient
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
        return self.predict_wrapper(X, predict_logistic_bool)

    def predict_proba(self, X):
        col1 = self.predict_wrapper(X, predict_logistic).reshape(-1, 1)
        col2 = 1. - col1
        return concatenate((col1, col2), axis=1)
