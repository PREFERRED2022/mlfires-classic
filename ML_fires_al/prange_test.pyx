import numpy as np
cimport numpy as cnp
import cython
from cython.parallel import prange
ctypedef cnp.double_t cDOUBLE
DOUBLE = np.float64
cimport openmp
cdef int num_threads
cnp.import_array()

#openmp.omp_set_dynamic(1)
def mydot(cnp.ndarray[cDOUBLE, ndim=2] a, cnp.ndarray[cDOUBLE, ndim=2] b, int nt):
    cdef cnp.ndarray[cDOUBLE, ndim=2] c
    cdef int i, M, N, K

    c = np.zeros((a.shape[0], b.shape[1]), dtype=DOUBLE)
    M = a.shape[0]
    N = a.shape[1]
    K = b.shape[1]

    for i in prange(M, num_threads=nt, nogil=True):
        multiply(&a[i,0], &b[0,0], &c[i,0], N, K)
    return c

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void multiply(double *a, double *b, double *c, int N, int K) nogil:
    cdef int j, k
    for j in range(N):
        for k in range(K):
            c[k] += a[j]*b[k+j*K]


