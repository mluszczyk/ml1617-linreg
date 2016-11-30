import math
import numpy
from sklearn.utils import shuffle


def predict(w, x):
    return numpy.asarray([numpy.inner(w, xi) for xi in x])


def logistic(u):
    return 1 / (1 + math.e ** -u)


def predict_logistic(w, x):
    return logistic(predict(w, x))


def predict_logistic_bool(w, x):
    """Equivalent to predict_logistics(w, x) > 0.5
    Source: course materials, lesson 4, slide 6.
    """
    return predict(w, x) > 0.


def l2_loss(ys: numpy.ndarray, ps: numpy.ndarray):
    return numpy.mean((ys - ps) ** 2)


def rmse_partial_derivative(l2, y, w, x, i) -> float:
    """But there's no square and the constant of 2 is removed."""
    n = len(y)
    return (
        -1. / n * sum((yk - numpy.inner(w, xk)) * xk[i] for yk, xk in zip(y, x)) +
        ((2 * l2 * w[i]) if (i > 0) else 0)
    )


def partial_derivative_logistic(l2, y, w, x, i) -> float:
    """Source: course materials, presentation for lesson 4, s 11."""
    return (
        sum((logistic(numpy.inner(w, xk)) - yk) * xk[i] for xk, yk in zip(x, y)) +
        ((2 * l2 * w[i]) if (i > 0) else 0)
    )


def gradient(partial_derivative, l2, w, x, y):
    return numpy.asarray(tuple(partial_derivative(l2, y, w, x, i) for i, _ in enumerate(w)))


def adjust(x):
    """Prepends row filled with 1s to the vector, so that constant of a polynomial
    is evaluated easily.
    """
    return numpy.insert(x, 0, 1, axis=1)


def linreg(loss_partial_derivative, x, y, batch_size, n_epochs, shuffle_: bool, l2, learning_rate, decay):
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
            grad = gradient(loss_partial_derivative, l2, w, bx, by)
            w += -learning_rate * grad
            learning_rate *= decay
    return w
