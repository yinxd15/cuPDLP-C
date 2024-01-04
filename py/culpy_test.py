import culpy
import numpy as np
import scipy.sparse as sp


# min x + y + z
# s.t.
# x + y >= 1
# y + z >= 2
# z + x >= 3

A = sp.csc_array([
    [1, 1, 1],
    [1, 0, 0],
    [-1, 0, 0],
], dtype="double")
nRows, nCols = A.shape
nnz = A.nnz
nEqs = 1

colMatBeg = A.indptr
colMatIdx = A.indices
colMetElem = A.data
b = np.array([1, 0, -1], dtype="double")
c = np.array([1, 0, 0], dtype="double")
print(colMatBeg, colMatIdx, colMetElem)
print(b.dtype, c.dtype, colMetElem.dtype)
x = culpy.solve(nRows, nCols, nnz, nEqs, colMatBeg, colMatIdx, colMetElem, b, c, 
    {"nIterLim": 1000, "dTimeLim": 3000.11}
)
print(x)
# print(y)