import numpy as np
cimport numpy as cnp
import cython
from cython.parallel import prange
ctypedef cnp.double_t cDOUBLE
DOUBLE = np.float64
cimport openmp
cdef int num_threads

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def fillcube(cnp.ndarray[int, ndim=2] idx, cnp.ndarray[cDOUBLE, ndim=2] table,int nt):
    cdef cnp.ndarray[cDOUBLE, ndim = 3] a3D
    cdef int i, M

    print(cnp.max(idx[1,:]), cnp.max(idx[2,:]), table.shape[1])
    a3D = np.zeros((cnp.max(idx[1,:]), cnp.max(idx[2,:]), table.shape[1]), dtype=DOUBLE)
    M = table.shape[0]

    for i in prange(M, num_threads=nt, nogil=True):
        a3D[idx[1,i], idx[2,i]]=table[idx[i],:]
    return a3D


