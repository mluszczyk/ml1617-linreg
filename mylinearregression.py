import numpy
from sklearn.preprocessing import StandardScaler

from mybaseregression import MyBaseRegression
from noopscaler import NoOpScaler


class MyLinearRegression(MyBaseRegression):
    def __init__(self,
                 batch_size=None, n_epochs=100, shuffle=False,
                 holdout_size=0., l2=0., learning_rate=.1, decay=1.0,
                 standardize=False):
        super().__init__(rmse_gradient,
                         batch_size, n_epochs, shuffle,
                         holdout_size, l2, learning_rate, decay,
                         standardize)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        return self.predict_wrapper(X, lambda w, x: x @ w)

    @staticmethod
    def get_scaler_y(standardize: bool):
        if standardize:
            return StandardScaler()
        else:
            return NoOpScaler()


def rmse_gradient(w, x, y):
    return -numpy.mean([
        (yk - numpy.inner(w, xk)) * xk for yk, xk in zip(y, x)
    ], axis=0)
