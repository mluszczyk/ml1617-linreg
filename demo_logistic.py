"""Demonstrates logistic regression.
Shows histograms of predict_proba from sklearn and predict_log from my regression.
"""

import numpy as np

from matplotlib import pyplot

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mylogisticregression import MyLogisticRegression

np.random.seed(123)

a = 0.3
b = -0.2
c = 0.001


def lin_rule(x, noise=0.):
    return a * x[0] + b * x[1] + c + noise < 0.


def get_y_fun(a, b, c):
    def y(x):
        return - x * a / b - c / b
    return y


lin_fun = get_y_fun(a, b, c)

n = 1000
range_points = 1
range_plot = 1.1

sigma = 0.05

X = range_points * 2 * (np.random.rand(n, 2) - 0.5)

y = [lin_rule(x, sigma * np.random.normal()) for x in X]

print(X[:10])
print(y[:10])


estimators = [
    LogisticRegression(),
    MyLogisticRegression(learning_rate=1.)
]

for estimator in estimators:
    estimator.fit(X, y)

    y_pred = estimator.predict(X)

    print('Accuracy: {}'.format(accuracy_score(y, y_pred)))

pyplot.subplot(2, 1, 1)
pyplot.hist(estimators[0].predict_proba(X))
pyplot.subplot(2, 1, 2)
pyplot.hist(estimators[1].predict_proba(X))

pyplot.show()
