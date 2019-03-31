import numpy as np

a = np.identity(4)
b = np.identity(4)
a[0][3] = -3
b[0][3] = 3
print(a,'\n',b)
print(a*b)
print(np.matmul(a,b))

