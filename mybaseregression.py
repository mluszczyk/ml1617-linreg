from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle, check_X_y, check_array

from linreg import predict, linreg, adjust


class MyBaseRegression(BaseEstimator):
    def __init__(self, loss_partial_derivative,
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

        self.w = None
        self.validation = None
        self.standard_scaler = None

        self.partial_derivative = loss_partial_derivative
        self.predict_func = predict

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
        if self.standardize:
            self.standard_scaler = StandardScaler()
            X = self.standard_scaler.fit_transform(X)
        batch_size = X.shape[0] if self.batch_size is None else self.batch_size
        self.w = linreg(
            self.partial_derivative,
            adjust(X), y, batch_size, self.n_epochs, self.shuffle,
            self.l2, self.learning_rate, self.decay)
        # Return the estimator
        return self

    def predict_wrapper(self, X, predict_func):
        X = check_array(X)
        if self.standard_scaler is not None:
            X = self.standard_scaler.transform(X)
        return predict_func(self.w, adjust(X))