import random
import numpy as np
quad = lambda x: (np.sum(x ** 2), x * 2)
print "Running sanity checks..."
#gradcheck_naive(quad, np.array(123.456))      # scalar test
#gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
arr = np.array([1,2,3])
print(quad(arr)) 

print(np.array([[1,2,3],[4,5,6]])*np.array([[1,2,3],[4,5,6]]))
arr = np.array([[1,2,3],[4,5,6]])
print(len(arr.shape))
print(arr**2)
print(np.sqrt(2))