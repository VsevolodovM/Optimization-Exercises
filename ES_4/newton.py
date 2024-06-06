import numpy as np


def newton(f, grad_f, hess_f, x0, tol, max_iter):
    x = x0
    for k in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Abbruchbedingung
        if np.linalg.norm(grad, 2) <= tol:
            return k, x
        
        # Lösung des linearen Gleichungssystems H(x_k) p_k = -grad_f(x_k)
        s = np.linalg.solve(hess, -grad)
        
        # Aktualisierung
        x = x + s
    
    return max_iter, x


## Test 1 (of 3) of the newton function
## test using the Rosenbrock function
## https://en.wikipedia.org/wiki/Rosenbrock_function
## f(x,y) = 100(y-x^2)^2 + (1-x)^2
## the function call rosenbrock() must return:
## n = 5
## x_n = [1., 1.]

def rosenbrock():
    f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    grad_f = lambda x: np.array([400*x[0]*(x[0]**2-x[1])
                                 + 2*(x[0]-1),
                                 200*(x[1]-x[0]**2)
                                 ])
    hess_f = lambda x: np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],[-400 * x[0], 200]])
    # iterations setting
    x0 = np.array([-50, 30.5])
    tol = 1e-6
    max_iter = 10**6

    n, x = newton(f, grad_f, hess_f, x0, tol, max_iter)
    print('Result of the test with Rosenbrock function:')
    print('The iteration index is', n)
    print('x_n =', x)

rosenbrock()


# Test 2 (of 3) of the newton function
# test using the Himmelblau function
# https://en.wikipedia.org/wiki/Himmelblau%27s_function
# f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
# the function call himmelblau() must return:
# n = 10
# x_n = [3.58442834, -1.84812654]

def himmelblau():
    f = lambda x: (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2
    grad_f = lambda x: np.array([4*x[0]*(x[0]**2+x[1]-11)
                                 + 2*(x[0]+x[1]**2-7),
                                 2*(x[0]**2+x[1]-11)
                                 + 4*x[1]*(x[0]+x[1]**2-7)
                                 ])
    hess_f = lambda x: np.array([[12 * x[0]**2 + 4 * x[1] - 40, 4 * x[0] + 4 * x[1]],
                                [4 * x[0] + 4 * x[1], 4 * x[0] + 12 * x[1]**2 - 26]])
    # iterations setting
    x0 = np.array([15, -22])
    tol = 1e-6
    max_iter = 10**6

    n, x = newton(f, grad_f, hess_f, x0, tol, max_iter)
    print('Result of the test with Himmelblau function:')
    print('The iteration index is', n)
    print('x_n =', x)

himmelblau()


## Test 3 (of 3) for the newton function
## test using the Bazaraa/Shetty function
## f(x,y) = (x-2)^4 + (x-2y)^2
## the function call bazaraa() must return:
## n = 18
## x_n = [2.00541312, 1.00270656]

def bazaraa():
    f = lambda x: (x[0]-2)**4 + (x[0]-2*x[1])**2
    grad_f = lambda x: np.array([4*(x[0]-2)**3 + 2*(x[0]-2*x[1]),
                                 4*(2*x[1]-x[0])
                                 ])
    hess_f = lambda x: np.array([[12 * (x[0] + x[1])**2 + 2, -4],
                                [-4, 8]])
    # iterations setting
    x0 = np.array([10, -20])
    tol = 1e-6
    max_iter = 10**6

    n, x = newton(f, grad_f, hess_f, x0, tol, max_iter)
    print('Result of the test with Bazaraa function:')
    print('The iteration index is', n)
    print('x_n =', x)

bazaraa()
