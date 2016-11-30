import numpy

from mylinearregression import MyLinearRegression

X = numpy.asarray([[2., 2.], [0., 2.], [2., 0.], [0., 0.]])
y = numpy.asarray([1., -1., 3., 1.])

estimator = MyLinearRegression()
estimator.fit(X, y)

print(estimator.w)

y_pred = estimator.predict(X)
print(y_pred)