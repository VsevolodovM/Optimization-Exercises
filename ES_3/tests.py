## Test 1 (of 2) for the Armijo-rule
## the function call test_armijo() must return (in this order):
## 0.26804671691687404
## 0.010644075786444315
## 0.26358518435595346
## 0.010613705445787398
import armijo
import numpy as np
from steepest import *

def test_armijo():
    f = lambda x: x[0]**2 + 3*x[0]*x[1] + 12
    grad_f = lambda x: np.array([2*x[0] + 3*x[1], 3*x[0]])

    x = np.array([1, 2])
    s = -grad_f(x)
    betas = [0.99, 0.98]
    gammas = [0.5, 0.98]

    for beta in betas:
        for gamma in gammas:
            ans = armijo(f, grad_f, beta, gamma, x, s)
            print("when beta =", beta, "and gamma =", gamma,
                  "we get sigma = ", ans)


# Test 2 (of 2) for the Armijo-rule
# the function call test_armijo_2() must return an error
# (in this example, s is not a descent direction of f at x)

def test_armijo_2():
    f = lambda x: x[0]**2 + 3*x[0]*x[1] + 12
    grad_f = lambda x: np.array([2*x[0] + 3*x[1], 3*x[0]])

    x = np.array([1, 2])
    s = grad_f(x)
    beta = 0.5
    gamma = 1e-2

    sigma = armijo(f, grad_f, beta, gamma, x, s)


## Test 1 (of 3) for the steepestdescent function
## test steepestdescent() using the Rosenbrock function
## https://en.wikipedia.org/wiki/Rosenbrock_function
## f(x,y) = 100(y-x^2)^2 + (1-x)^2
## the function call rosenbrock() must return:
## n = 56383
## x_n = [1.00000784, 1.00001569]

def rosenbrock():
    f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    grad_f = lambda x: np.array([400*x[0]*(x[0]**2-x[1])
                                 + 2*(x[0]-1),
                                 200*(x[1]-x[0]**2)
                                 ])
    # iterations setting
    x0 = np.array([-0.5, 20])
    tol = 1e-5
    max_iter = 10**6

    n, x = steepestdescent(f, grad_f, x0, tol, max_iter)
    print('The iteration index is', n)
    print('x_n =', x)


## Test 2 (of 3) for the steepestdescent function
## test steepestdescent using the Himmelblau function
## https://en.wikipedia.org/wiki/Himmelblau%27s_function
## f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
## the function call himmelblau() must return:
## n = 45
## x_n = [3.58442841, -1.84812652]

def himmelblau():
    f = lambda x: (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2
    grad_f = lambda x: np.array([4*x[0]*(x[0]**2+x[1]-11)
                                 + 2*(x[0]+x[1]**2-7),
                                 2*(x[0]**2+x[1]-11)
                                 + 4*x[1]*(x[0]+x[1]**2-7)
                                 ])

    # iterations setting
    x0 = np.array([5, -8])
    tol = 1e-5
    max_iter = 10**6

    n, x = steepestdescent(f, grad_f, x0, tol, max_iter)
    print('The iteration index is', n)
    print('x_n =', x)


## Test 3 (of 3) for the steepestdescent function
## test steepestdescent using the Bazaraa/Shetty function
## f(x,y) = (x-2)^4 + (x-2y)^2
## the function call bazaraa() must return:
## n = 2992
## x_n = [2.01303834, 1.00651901]

def bazaraa():
    f = lambda x: (x[0]-2)**4 + (x[0]-2*x[1])**2
    grad_f = lambda x: np.array([4*(x[0]-2)**3 + 2*(x[0]-2*x[1]),
                                 4*(2*x[1]-x[0])
                                 ])

    # iterations setting
    x0 = np.array([4, 2])
    tol = 1e-5
    max_iter = 10**6

    n, x = steepestdescent(f, grad_f, x0, tol, max_iter)
    print('The iteration index is', n)
    print('x_n =', x)



bazaraa()