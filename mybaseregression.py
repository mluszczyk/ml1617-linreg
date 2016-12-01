from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle, check_X_y, check_array

from gradientdescent import gradient_descent, adjust
from noopscaler import NoOpScaler


class MyBaseRegression(BaseEstimator):
    def __init__(self, loss_gradient,
                 batch_size=None, n_epochs=100, shuffle: bool=False,
                 holdout_size: float=0., l2=0., learning_rate=.1, decay=1.0,
                 standardize: bool=False):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.holdout_size = holdout_size
        self.l2 = l2
        self.learning_rate = learning_rate
        self.decay = decay
        self.standardize = standardize

        self.standard_scaler_x = self.get_scaler_x(self.standardize)
        self.standard_scaler_y = self.get_scaler_y(self.standardize)

        self.w = None
        self.validation = None

        self.loss_gradient = loss_gradient

    @staticmethod
    def get_scaler_x(standardize: bool):
        if standardize:
            return StandardScaler()
        else:
            return NoOpScaler()

    @staticmethod
    def get_scaler_y(standardize: bool):
        return NoOpScaler()

    def holdout(self, X, y):
        holdout_num = int(round(self.holdout_size * X.shape[0]))
        if self.holdout_size:
            if self.shuffle:
                X, y = shuffle(X, y)
            self.validation = X[:holdout_num, :], y[:holdout_num]
            X, y = X[holdout_num:], y[holdout_num:]
        return X, y

    def fit(self, X, y):
        """A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        X, y = self.holdout(X, y)
        X = self.standard_scaler_x.fit_transform(X)
        y = self.standard_scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)
        batch_size = X.shape[0] if self.batch_size is None else self.batch_size
        self.w = gradient_descent(
            self.loss_gradient,
            adjust(X), y, batch_size, self.n_epochs, self.shuffle,
            self.l2, self.learning_rate, self.decay)
        # Return the estimator
        return self

    def predict_wrapper(self, X, predict_func):
        X = check_array(X)
        X = self.standard_scaler_x.transform(X)
        predicted = predict_func(self.w, adjust(X))
        return self.standard_scaler_y.inverse_transform(predicted.reshape(-1, 1)).reshape(-1)

    def fit_transform_y(self, y):
        return y

    def inverse_transform_y(self, y):
        return y
