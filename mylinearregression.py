from mybaseregression import MyBaseRegression
from linreg import rmse_partial_derivative, predict


class MyLinearRegression(MyBaseRegression):
    """ A template estimator to be used as a reference implementation .
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(rmse_partial_derivative, *args, **kwargs)

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
