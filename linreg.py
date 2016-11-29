import math
import numpy
from sklearn.utils import shuffle


def predict(w, x):
    return numpy.asarray([numpy.inner(w, xi) for xi in x])


def predict_logistic(w, x):
    return 1 / (1 + math.e ** - predict(w, x))


def l2_loss(ys, ps):
    assert len(ys) == len(ps)
    return sum((y - p) ** 2 for y, p in zip(ys, ps)) / len(ys)


def rmse(w, x, y):
    return l2_loss(y, predict(w, x))


def rmse_partial_derivative(l2, y, w, x, i):
    n = len(y)
    return (
        -2. / n * sum((y[k] - numpy.inner(w, x[k])) * x[k][i] for k, _ in enumerate(y)) +
        l2 * numpy.sum(w)
    )


def partial_derivative_logistic(l2, y, w, x, i):
    rmse_ = rmse(w, x, y)
    rmse_partial_derivative_ = rmse_partial_derivative(l2, y, w, x, i)
    return math.e ** rmse_ * rmse_partial_derivative_ / (1 + math.e ** rmse_)


def gradient(partial_derivative, l2, w, x, y):
    dim = len(w)
    return numpy.asarray(tuple(partial_derivative(l2, y, w, x, i) for i in range(dim)))


def adjust(x):
    """Prepends row filled with 1s to the vector, so that constant of a polynomial
    is evaluated easily.
    """
    return numpy.insert(x, 0, 1, axis=1)


def linreg(partial_loss, x, y, batch_size, n_epochs, shuffle_: bool, l2, learning_rate, decay):
    start = numpy.zeros((x.shape[1],))

    w = start
    for num in range(n_epochs):
        if shuffle_:
            x, y = shuffle(x, y)
        batch_iterator = (
            (x[start:start + batch_size], y[start:start + batch_size])
            for start in range(0, x.shape[0], batch_size)
        )

        for bx, by in batch_iterator:
            grad = gradient(partial_loss, l2, w, bx, by)
            w += -learning_rate * grad
            learning_rate *= decay
    return w
