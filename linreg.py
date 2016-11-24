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


def descent(start, gradient, evaluate, eps, gamma, tries):
    w = start
    for num in range(tries):
        grad = gradient(w)
        if distance(grad) < eps:
            return w
        else:
            w += -gamma * grad
        print("after num", num)
        print("w", w)
        print("grad", grad)
        print("eval", evaluate(w))

    raise RuntimeError("Not converged")


def adjust(x):
    return numpy.insert(x, 0, 1, axis=1)


def linreg(x, y, eps, gamma, tries):
    start = numpy.zeros((x.shape[1],))

    def gradient(w):
        return calculate_grad(w, x, y)

    return descent(start, gradient, lambda w: evaluate(w, x, y), eps, gamma, tries)


def main():
    xs = [0.528229010724395, 0.9027745539321911, 0.3118521458689988, 0.11704848570256954, 0.4766089600561566,
     0.8668141447194354, 0.769038647282098, 0.8098811544095684, 0.05375849568209068, 0.18397805336513817,
     0.36641542412849015, 0.9195363797475026, 0.30861831282417374, 0.03593311622663642, 0.603159475302856,
     0.37447220715639884, 0.8874675593178811, 0.5968691534877782, 0.8327689006487441, 0.0693988278626102,
     0.3047020662736677, 0.02446020472845012, 0.8102245130153789, 0.9376997208167048, 0.22331195497364453,
     0.6877732863395257, 0.04309004460198451, 0.7699576194773077, 0.9251029071041337, 0.6497096323562148,
     0.39043786139643566, 0.5658964998386322, 0.9560865232041085, 0.12240076362169139, 0.663259319266864,
     0.6439028501152504, 0.3795501715840256, 0.8723756163652371, 0.42850106230142115, 0.9069686142431905,
     0.5555111846118262, 0.2719987530642608, 0.24824896777883598, 0.6422411069726882, 0.7615255784890255,
     0.5212169118238551, 0.6632230125941745, 0.05272145788444327, 0.39472218571003914, 0.9361550756020353]
    ys = [0.6660301081666091, 0.7880311929162538, 0.6209767354689372, 0.49839473536150647, 0.6359144631450441,
     0.775198404343493, 0.7459093307043013, 0.7133720991350466, 0.4591466030107432, 0.5530876060405034,
     0.6523828434598912, 0.7974052361256054, 0.6038908262366895, 0.5654206237724303, 0.6900629303151626,
     0.6046792549694412, 0.7531726605001798, 0.6524867803417727, 0.7524037054223885, 0.5206761046700723,
     0.5801936133423045, 0.5211079298372449, 0.7610907666606702, 0.7650513166070351, 0.5430171847213586,
     0.6838572594876159, 0.5257790543690498, 0.749720166273106, 0.7417230143992304, 0.7170392536658252,
     0.6054873002426322, 0.6494912378903008, 0.7836036247312703, 0.5660181206880279, 0.6611925537901934,
     0.677418202867866, 0.6212332444014119, 0.755211481708266, 0.6574209206631763, 0.7961329669363469,
     0.6780963740303935, 0.5777549618568788, 0.6029694914018995, 0.7055898454746773, 0.7615936253567677,
     0.6509820273225375, 0.6921296101111231, 0.4965640242465189, 0.5969105590529076, 0.7922304500021877]
    Xs_orig = numpy.asarray(xs).reshape((len(xs), 1))
    Ys = numpy.asarray(ys)

    Xs = adjust(Xs_orig)

    output = linreg(Xs, Ys, 0.0000001, 0.25, 70)
    print([round(f, 2) for f in output])


if __name__ == '__main__':
    main()

