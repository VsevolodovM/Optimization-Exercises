import numpy as np
import pickle

def armijo(f,gradient,beta:float, gamma:float, x:np.array, s:np.array):
    """This function returns parameter satisfying the conditions of Armijo"""
    sigma = 1
    grad_s = gradient(x) @ s
    if grad_s >= 0:
        raise ValueError("s is not direction of descent")
    else:
        while f(x + sigma * s) - f(x) > gamma * sigma * grad_s:
            sigma *= beta
        return sigma

def mkq(A, b, x0, tol, max_iter):
    x = x0
    s = -A.T @ (A @ x - b)
    f = lambda x: 0.5 * np.linalg.norm(A @ x - b)**2
    g = lambda x: A.T @ (A @ x - b)
    beta = 0.5
    gamma = 10 ** (-2)
    for _ in range(max_iter):
        if np.linalg.norm(-s, 2) < tol:
            break
        
        sigma = armijo(f, g, beta, gamma, x, s)
        
        
        x = x + sigma * s
        s = -A.T @ (A @ x - b)
        
    return x






# read the data
with open('tests.dat','rb') as file:
    data = pickle.load(file)

# number of tests
test_no = data['test_no']

# test data
A = data['A']
b = data['b']
# this parameter is used for determining whether your solution is
# "reasonable" or not
Pb = data['Pb']

# tol and max_iter setup
tol = 1e-6
max_iter = 10**7

for i in range(test_no):
    x0 = np.zeros(np.shape(A[i])[1])
    x = mkq(A[i], b[i], x0, tol, max_iter)
    print("the result of test number", i+1, "is\n",
          np.linalg.norm(A[i]@x-Pb[i]))

# after running this script, you must get, in this order:
# 2.213698984764086e-06
# 2.9099109647652636e-06
# 7.994830285471546e-06
# 2.74971129275074e-06
# 6.418113365990214e-07
# 7.88704137923301e-07
# 6.179706716428063e-07
# 6.912763563566402e-07
# 1.9544450967002434e-06
# 4.026004960229904e-07
