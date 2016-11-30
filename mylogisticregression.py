from linreg import partial_derivative_logistic, predict_logistic, predict_logistic_bool
from mybaseregression import MyBaseRegression


class MyLogisticRegression(MyBaseRegression):
    def __init__(self,
                 batch_size=None, n_epochs=100, shuffle = False,
                 holdout_size = 0., l2=0., learning_rate=.1, decay=1.0,
                 standardize = False):
        super().__init__(partial_derivative_logistic,
                         batch_size, n_epochs, shuffle,
                         holdout_size, l2, learning_rate, decay,
                         standardize)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X):
        return self.predict_wrapper(X, predict_logistic_bool)

    def predict_log(self, X):
        return self.predict_wrapper(X, predict_logistic)
