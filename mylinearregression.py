from mybaseregression import MyBaseRegression
from linreg import rmse_partial_derivative, predict


class MyLinearRegression(MyBaseRegression):
    """ A template estimator to be used as a reference implementation .
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(self,
                 batch_size=None, n_epochs=100, shuffle = False,
                 holdout_size = 0., l2=0., learning_rate=.1, decay=1.0,
                 standardize = False):
        super().__init__(rmse_partial_derivative,
                         batch_size, n_epochs, shuffle,
                         holdout_size, l2, learning_rate, decay,
                         standardize)

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
