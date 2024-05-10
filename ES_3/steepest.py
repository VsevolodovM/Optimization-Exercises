from armijo import *
from numpy import linalg as LA

def steepest(funct:types.FunctionType, gradient:np.array, xinit:np.array, gamma:float, beta:float, epsilon:float, maxit:int):  #Algo 8.1(Geiger)
    """This function is an implementation of Steepest Descent method"""
    case = 0
    k = 0
    x_k = xinit
    s_k = np.zeros_like(xinit)
    sigma_k = 0

    while LA.norm(gip(gradient, x_k)) > epsilon and k < maxit:
        s_k = -gip(gradient, x_k) / LA.norm(gip(gradient, x_k))

        sigma_k = armijo(funct, x_k, s_k, gradient, gamma, beta)

        x_k = x_k + sigma_k * s_k
        k += 1

    if LA.norm(gip(gradient, x_k)) <= epsilon:
        return [x_k, 0]
    elif k >= maxit:
        return [x_k, 1]