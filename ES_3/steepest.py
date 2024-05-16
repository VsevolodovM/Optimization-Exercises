from armijo import *
from numpy import linalg as LA

def steepestdescent(funct:types.FunctionType, gradient, xinit:np.array, tol:float, maxit:int):
    """This function is an implementation of Steepest Descent method"""
    beta = 0.5
    gamma = 10 ** (-2)
    k = 0
    x_k = xinit
    s_k = np.zeros_like(xinit)
    sigma_k = 0

    while LA.norm(gradient(x_k)) > tol and k < maxit:
        s_k = -gradient(x_k) 

        sigma_k = armijo(funct,gradient, beta, gamma,  x_k, s_k)

        x_k = x_k + sigma_k * s_k
        k += 1

    if LA.norm(gradient(x_k)) <= tol:
        return [k, x_k]
    elif k >= maxit:
        return [k, x_k]