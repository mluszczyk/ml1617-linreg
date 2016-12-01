from sklearn.preprocessing import StandardScaler

from mybaseregression import MyBaseRegression
from linreg import predict, rmse_gradient


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
        return self.predict_wrapper(X, predict)

    def fit_transform_y(self, y):
        if self.standardize:
            self.standard_scaler_y = StandardScaler()
            return self.standard_scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)
        else:
            self.standard_scaler_y = None
            return y

    def inverse_transform_y(self, y):
        if self.standard_scaler_y is None:
            return y
        else:
            return self.standard_scaler_y.inverse_transform(y.reshape(-1, 1)).reshape(-1)
