import numpy
from scipy.special import expit
from sklearn.utils import shuffle


def predict(w, x):
    return numpy.asarray([numpy.inner(w, xi) for xi in x])


def predict_logistic(w, x):
    return expit(predict(w, x))


def predict_logistic_bool(w, x):
    """Equivalent to predict_logistics(w, x) > 0.5
    Source: course materials, lesson 4, slide 6.
    """
    return predict(w, x) > 0.


def l2_regularization_gradient(l2, w):
    g = w.copy()
    g[0] = 0.  # don't penalize intercept
    return 2 * g * l2


def rmse_gradient(w, x, y):
    return -numpy.mean([
        (yk - numpy.inner(w, xk)) * xk for yk, xk in zip(y, x)
    ], axis=0)


def logistic_gradient(w, x, y):
    return numpy.mean([
        (expit(numpy.inner(w, xk)) - yk) * xk for xk, yk in zip(x, y)
    ], axis=0)


def adjust(x):
    """Prepends row filled with 1s to the vector, so that constant of a polynomial
    is evaluated easily.
    """
    return numpy.insert(x, 0, 1, axis=1)


def gradient_descent(loss_gradient,
                     x, y, batch_size, n_epochs, shuffle_: bool,
                     l2, learning_rate, decay):
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
            grad = loss_gradient(w, bx, by) + l2_regularization_gradient(l2, w)
            w += -learning_rate * grad
            learning_rate *= decay
    return w
