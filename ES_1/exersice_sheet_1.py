import numpy as np
import matplotlib.pyplot as plt


#First exercise: Implement the function which returns Hilbert Matrix of order N

def defineHilbert(N):
    H = np.zeros((N,N))
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            H[i - 1, j -1] = 1 / (i + j - 1)
    return H


print(defineHilbert(5))




#Second exercise:
# 1) Define an array x with equidistant values between -2pi and 2pi, with step pi/50
x = np.arange(-2 * np.pi, 2 * np.pi, np.pi/50)
# 2) Define y1 and y2 as the sin and cos of x
y1 = np.sin(x)
y2 = np.cos(x)
# 3) Plot x agains y1 and y2 on the same coordinate system with the following specifications:
#      i)  The legend for the (x, y1) should be "Sin" and for the (x, y2) should be "Cos"
#      ii) (x, y1) displayed as a solid blue line
#      iii)(x, y2) as a dashed red line

plt.figure(figsize=(10, 5))
plt.plot(x, y1, label='Sin', color='blue')  
plt.plot(x, y2, label='Cos', color='red', linestyle='dashed')  
plt.title('Sin- and Cos-curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()



