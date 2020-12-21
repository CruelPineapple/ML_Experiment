import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

arr1 = np.array([2, 3])
arr2 = np.array([4, 5])
print(np.dot(arr1, arr2))

arr3 = np.array([[2, 3], [4, 5]])
arr4 = np.array([[6, 7], [8, 9]])
print(np.dot(arr3, arr4))

x, y, r = sp.symbols('x,y,r')
print(2 * (r*r-x**2), (x, -r, r))

f = (1+2*x)*x**2
print(sp.expand(f))
print(f.subs(x, y))

f = x**2 - 1
print(sp.solve(f))

f = x**2 + y**3
print(sp.diff(f, x))

a = np.arange(10)
plt.plot(a, a*1.5, a, a*2.5, a, a*3.5, a, a*4.5)
plt.show()