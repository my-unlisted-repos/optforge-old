import numpy as np

def estimate_grad_forward2(f, x:np.ndarray, h = 1e-4):
    grad = np.empty_like(x)
    fx = f(x)

    vec = x.flat
    jac = grad.flat
    for i in range(x.size):

        vec[i] += h
        jac[i] = (f(x) - fx) / h
        vec[i] -= h

    return grad

def estimate_grad_backward2(f, x:np.ndarray, h = 1e-4):
    grad = np.empty_like(x)
    fx = f(x)

    vec = x.flat
    jac = grad.flat
    for i in range(x.size):

        vec[i] -= h
        jac[i] = (fx - f(x)) / h
        vec[i] += h

    return grad

def estimate_grad_central2(f, x:np.ndarray, h = 1e-4):
    pos = np.empty_like(x)
    neg = np.empty_like(x)

    vec = x.flat
    posv = pos.flat
    negv = neg.flat
    for i in range(x.size):

        vec[i] += h
        posv[i] = f(x)

        vec[i] -= 2*h
        negv[i] = f(x)

        vec[i] -= h

    grad = (pos - neg) / 2 * h
    return grad

def estimate_grad_forward3(f, x:np.ndarray, h = 1e-4):
    fh = np.empty_like(x)
    f2h = np.empty_like(x)

    vec = x.flat
    fhv = fh.flat
    f2hv = f2h.flat
    for i in range(x.size):

        vec[i] += h
        fhv[i] = f(x)

        vec[i] += h
        f2hv[i] = f(x)

        vec[i] -= 2*h

    grad = (-3*f(x) + 4*fh - f2h) / 2 * h
    return grad

def estimate_grad_backward3(f, x:np.ndarray, h = 1e-4):
    fh = np.empty_like(x)
    f2h = np.empty_like(x)

    vec = x.flat
    fhv = fh.flat
    f2hv = f2h.flat
    for i in range(x.size):

        vec[i] -= h
        fhv[i] = f(x)

        vec[i] -= h
        f2hv[i] = f(x)

        vec[i] += 2*h

    grad = (f2h - 4*fh + 3*f(x)) / 2 * h
    return grad


# 2 point
# 3 point
# spsa
# fdsa
# stochastic forward
# stochastic backward