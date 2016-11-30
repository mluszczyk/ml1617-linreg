from linreg import partial_derivative_logistic, predict_logistic, predict_logistic_bool
from mybaseregression import MyBaseRegression


class MyLogisticRegression(MyBaseRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(partial_derivative_logistic, *args, **kwargs)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X):
        return self.predict_wrapper(X, predict_logistic_bool)

    def predict_log(self, X):
        return self.predict_wrapper(X, predict_logistic)
