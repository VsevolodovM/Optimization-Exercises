import numpy as np



def QuadraticGradient(A:np.array, b:np.array, x_0:np.array, eps:float, max_iter:int) -> tuple[int, np.array]:
    n = 0
    x = x_0
    while n < max_iter:
        s_n = - A @ x - b
        norm_s_n = np.linalg.norm(s_n)
        if norm_s_n < eps:
            break
        else:
            sigma = norm_s_n**2 / (s_n @ A @ s_n)
            x = x + sigma * s_n
        n += 1
    return n, x 



if __name__ == "__main__":
    A = np.array([[4, 0],[0, 2]])
    b = np.array([-4,-2])
    x_0 = np.array([5, -5])
    eps = 10 ** (-4)
    max_iter = 10 ** 4
    print("1)  N:",QuadraticGradient(A,b,x_0,eps,max_iter)[0], "Xn:", QuadraticGradient(A,b,x_0,eps,max_iter)[1])

    A = np.array([[4, 0, 0, 0, 0],[0, 4, 0, 0, 0],[0, 0, 2, 0, 0],[0, 0, 0, 2, 0],[0, 0, 0, 0, 1]])
    b = np.array([-4,-4,-2,-2,-1])
    x_0 = np.array([0, 0, 0, 0, 0])
    print("2)  N:",QuadraticGradient(A,b,x_0,eps,max_iter)[0], "Xn:", QuadraticGradient(A,b,x_0,eps,max_iter)[1])