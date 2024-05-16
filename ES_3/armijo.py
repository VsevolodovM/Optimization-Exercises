import numpy as np
import types

def armijo(f:types.FunctionType,gradient,beta:float, gamma:float, x:np.array, s:np.array):
    """This function returns parameter satisfying the conditions of Armijo"""
    sigma = 1
    grad_s = gradient(x) @ s
    if grad_s >= 0:
        raise ValueError("s is not direction of descent")
    else:
        while f(x + sigma * s) - f(x) > gamma * sigma * grad_s:
            sigma *= beta
        return sigma
