import numpy


def predict(w, x):
    return numpy.asarray([numpy.inner(w, xi) for xi in x])


def evaluate(w, x, y):
    return l2_loss(y, predict(w, x))


def l2_loss(ys, ps):
    assert len(ys) == len(ps)
    return sum((y - p) ** 2 for y, p in zip(ys, ps)) / len(ys)


def partial_derivative(y, w, x, i):
    n = len(y)
    return -2. / n * sum((y[k] - numpy.inner(w, x[k])) * x[k][i] for k, _ in enumerate(y))


def calculate_grad(w, x, y):
    dim = len(w)
    return numpy.asarray(tuple(partial_derivative(y, w, x, i) for i in range(dim)))


def distance(vector):
    return numpy.inner(vector, vector)


def descent(start, get_batch_iterator, gradient, n_epochs, learning_rate, decay):
    w = start
    for num in range(n_epochs):
        for x, y in get_batch_iterator():
            grad = gradient(w, x, y)
            w += -learning_rate * grad
            learning_rate *= decay
    return w


def adjust(x):
    return numpy.insert(x, 0, 1, axis=1)


def linreg(x, y, batch_size, n_epochs, learning_rate, decay):
    start = numpy.zeros((x.shape[1],))

    def get_batch_iterator():
        return (
            (x[start:start + batch_size], y[start:start + batch_size])
            for start in range(0, x.shape[0], batch_size)
        )

    return descent(start, get_batch_iterator, calculate_grad, n_epochs, learning_rate, decay)
