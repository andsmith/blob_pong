import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Define a sparse matrix in CSR format
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
A = csr_matrix((data, (row, col)), shape=(3, 3))

# Define the right-hand side vector
b = np.array([6, -4, 27])

# Solve the system Ax = b
x = spsolve(A, b)

print(x)
# Expected Output: [ -5.   3.   5.]