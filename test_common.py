import numpy as np

def fun_zero(t, y):
    return np.zeros_like(y)

def func_spiral(t, y):
    return [np.cos(t) - y[1], np.sin(t) + y[0]]