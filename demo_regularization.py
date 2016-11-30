from unittest import TestCase

import numpy
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from mylinearregression import MyLinearRegression

train_n = 60
test_n = 60

X_full = numpy.random.rand(train_n + test_n, 30)
y_full = X_full[:, 0] + X_full[:, 1] + 1

X_train, y_train = X_full[:train_n], y_full[:train_n]
X_test, y_test = X_full[train_n:], y_full[train_n:]


def main():
    ridge_estimator = Ridge(alpha=0.1)
    ridge_estimator.fit(X_train, y_train)
    print(ridge_estimator.intercept_, ridge_estimator.coef_)

    lasso_estimator = Lasso(alpha=0.01)
    lasso_estimator.fit(X_train, y_train)
    print(lasso_estimator.intercept_, lasso_estimator.coef_)

    sklin_estimator = LinearRegression()
    sklin_estimator.fit(X_train, y_train)
    print(sklin_estimator.intercept_, sklin_estimator.coef_)

    my_estimator = MyLinearRegression(n_epochs=500)
    my_estimator.fit(X_train, y_train)
    print(my_estimator.w)

    myl2_estimator = MyLinearRegression(n_epochs=500, l2=0.01)
    myl2_estimator.fit(X_train, y_train)
    print(myl2_estimator.w)

    print("train scores")
    print(r2_score(y_train, ridge_estimator.predict(X_train)))
    print(r2_score(y_train, lasso_estimator.predict(X_train)))
    print(r2_score(y_train, sklin_estimator.predict(X_train)))
    print(r2_score(y_train, my_estimator.predict(X_train)))
    print(r2_score(y_train, myl2_estimator.predict(X_train)))

    print("test scores")
    print(r2_score(y_test, ridge_estimator.predict(X_test)))
    print(r2_score(y_test, lasso_estimator.predict(X_test)))
    print(r2_score(y_test, sklin_estimator.predict(X_test)))
    print(r2_score(y_test, my_estimator.predict(X_test)))
    print(r2_score(y_test, myl2_estimator.predict(X_test)))

if __name__ == '__main__':
    main()
