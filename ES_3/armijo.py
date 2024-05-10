import numpy as np
import types

def armijo(f:types.FunctionType, x:np.array, s:np.array, gradient:list, gamma:float, beta:float):
    """This function returns parameter satisfying the conditions of Armijo"""
    sigma = 1
    grad = gip(gradient, x) @ s
    while f(x + sigma * s) - f(x) > gamma * sigma * grad:
        sigma *= beta
    return sigma


def gip(gradient, point): #gradient in point
    """This function returns value of gradient at the give point"""
    return np.array([f(point) for f in gradient])